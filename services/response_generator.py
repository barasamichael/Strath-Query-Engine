import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytz
from openai import OpenAI

from services.embeddings import EmbeddingService
from services.intent_recognizer import IntentRecognizer, IntentType
from services.vector_db import VectorDBService
from services.tavily_service import TavilyService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_generator")

# Intents that warrant checking real-time data only when explicit temporal
# signals are also present in the query.  Schedule queries are always
# real-time because the structured DB is local and cheap.
_ALWAYS_REAL_TIME = {IntentType.SCHEDULE_QUERY}

# Words/phrases that signal the user needs current-as-of-today data.
_TEMPORAL_SIGNALS = {
    "current", "latest", "recent", "today", "now", "right now",
    "this semester", "current semester", "this year",
    "deadline", "upcoming", "announcement", "news", "update",
    "how much", "exact amount", "specific",
}


class _ResponseCache:
    """
    Semantic response cache backed by embedding similarity.
    Entries expire after `ttl_seconds`; capacity is capped at `max_size`.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.97,
        ttl_seconds: int = 3600,
        max_size: int = 128,
    ):
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._max_size = max_size
        # Each entry: (embedding: np.ndarray, response: dict, ts: float, intent: IntentType)
        self._entries: List[Tuple] = []

    def get(
        self, query_embedding: np.ndarray, intent_type: IntentType
    ) -> Optional[Dict]:
        now = time.time()
        for emb, response, ts, cached_intent in self._entries:
            if now - ts > self._ttl:
                continue
            if cached_intent != intent_type:
                continue
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(emb)
            if norm_q == 0 or norm_e == 0:
                continue
            sim = float(np.dot(query_embedding, emb) / (norm_q * norm_e))
            if sim >= self._threshold:
                logger.info("Cache hit (similarity=%.3f)", sim)
                return response
        return None

    def put(
        self,
        query_embedding: np.ndarray,
        response: Dict,
        intent_type: IntentType,
    ) -> None:
        now = time.time()
        # Evict expired entries first
        self._entries = [
            e for e in self._entries if now - e[2] < self._ttl
        ]
        if len(self._entries) >= self._max_size:
            self._entries.pop(0)
        self._entries.append((query_embedding, response, now, intent_type))


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters for English text."""
    return max(1, len(text) // 4)


# Maximum tokens we allow the context block to consume (leaves room for
# the system prompt, user prompt wrapper, and the completion itself).
_CONTEXT_TOKEN_BUDGET = 80_000


class ResponseGenerator:
    """
    Generates answers by orchestrating intent recognition, structured
    schedule lookup, Tavily real-time search, and vector retrieval,
    then calling an LLM to synthesise a grounded, cited response.
    """

    def __init__(
        self,
        vector_db_service: Optional[VectorDBService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        tavily_service: Optional[TavilyService] = None,
        structured_storage=None,
        intent_recognizer: Optional[IntentRecognizer] = None,
    ):
        self.model = "gpt-4o-mini"
        self.temperature = 0.1
        self.max_tokens = 4096

        self.vector_db = vector_db_service or VectorDBService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.tavily_service = tavily_service
        self.structured_storage = structured_storage
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self._cache = _ResponseCache()

        logger.info(
            "ResponseGenerator initialised — real-time: %s",
            "ACTIVE" if self.tavily_service else "DISABLED",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_kenya_time(self) -> Tuple[str, str]:
        try:
            kenya_tz = pytz.timezone("Africa/Nairobi")
            now = datetime.now(kenya_tz)
            return now.strftime("%Y-%m-%d %H:%M:%S EAT"), now.isoformat()
        except Exception:
            now = datetime.utcnow()
            return now.strftime("%Y-%m-%d %H:%M:%S UTC"), now.isoformat()

    def generate_response(
        self,
        query: str,
        context_info: Optional[Dict[str, Any]] = None,
        use_real_time: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded, cited response for `query`.

        Args:
            query: The user's question.
            context_info: Optional memory/context metadata from MemoryProcessor.
            use_real_time: Allow Tavily queries (can be disabled in tests).
            conversation_history: Prior turns as [{"role": "user"|"assistant", "content": "..."}].
        """
        try:
            start = time.time()

            # 1. Intent classification via the shared IntentRecognizer
            intent_info = self.intent_recognizer.recognize_intent(query)
            intent_type: IntentType = intent_info["intent_type"]
            intent_confidence: float = intent_info["confidence"]

            # 2. Cache lookup (requires a query embedding)
            query_embedding = self.embedding_service.embed_query(query)
            if query_embedding is not None:
                cached = self._cache.get(query_embedding, intent_type)
                if cached is not None:
                    cached["cache_hit"] = True
                    return cached

            # 3. Structured schedule lookup (highest-priority source)
            structured_schedule = None
            if intent_type == IntentType.SCHEDULE_QUERY:
                structured_schedule = self._query_schedule_structured(query)

            # 4. Real-time data via Tavily (only when genuinely needed)
            real_time_data = None
            if use_real_time and self.tavily_service:
                if self._needs_real_time(query, intent_type):
                    # Skip if structured schedule already covers this query
                    if not (
                        intent_type == IntentType.SCHEDULE_QUERY
                        and structured_schedule
                    ):
                        real_time_data = self._fetch_real_time_parallel(
                            query, intent_type
                        )

            # 5. Vector context — shrink limit when richer sources exist
            if structured_schedule:
                ctx_limit = 5
            elif real_time_data:
                ctx_limit = 10
            else:
                ctx_limit = 15
            vector_context = self._get_vector_context(
                query, intent_type, ctx_limit
            )

            # 6. LLM synthesis
            response_content = self._generate_direct_response(
                query=query,
                intent_type=intent_type,
                vector_context=vector_context,
                real_time_data=real_time_data,
                structured_schedule=structured_schedule,
                context_info=context_info,
                conversation_history=conversation_history or [],
            )

            processing_time = time.time() - start

            approach = "vector_only"
            if structured_schedule and real_time_data:
                approach = "structured_schedule_with_real_time"
            elif structured_schedule:
                approach = "structured_schedule"
            elif real_time_data:
                approach = "vector_with_real_time"

            result = {
                "response": response_content["content"],
                "intent_type": intent_type.value,
                "intent_confidence": intent_confidence,
                "context_sources": len(vector_context),
                "real_time_used": real_time_data is not None,
                "real_time_results": (
                    len(real_time_data.get("results", [])) if real_time_data else 0
                ),
                "structured_schedule_used": structured_schedule is not None,
                "structured_schedule_entries": (
                    structured_schedule.get("result_count", 0)
                    if structured_schedule
                    else 0
                ),
                "processing_time": processing_time,
                "token_usage": response_content.get("token_usage", {}),
                "approach": approach,
                "cache_hit": False,
                "timestamp": datetime.now(
                    pytz.timezone("Africa/Nairobi")
                ).strftime("%Y-%m-%d %H:%M:%S EAT"),
            }

            if real_time_data:
                result["real_time_sources"] = [
                    r.get("url", "") for r in real_time_data.get("results", [])
                ]

            # Store in cache
            if query_embedding is not None:
                self._cache.put(query_embedding, result, intent_type)

            logger.info(
                "Response generated in %.2fs — approach=%s real_time=%s",
                processing_time,
                approach,
                real_time_data is not None,
            )
            return result

        except Exception as e:
            logger.error("Error generating response: %s", e)
            return {
                "response": (
                    "I encountered an error processing your question. "
                    "Please try rephrasing or ask about something else."
                ),
                "error": str(e),
                "intent_type": "error",
                "real_time_used": False,
                "cache_hit": False,
            }

    # ------------------------------------------------------------------
    # Real-time decision logic
    # ------------------------------------------------------------------

    def _needs_real_time(self, query: str, intent_type: IntentType) -> bool:
        """
        Return True only when the query genuinely requires current data.

        Schedule queries always hit the structured DB (not Tavily) so they
        are excluded here.  Fees and admission queries only trigger Tavily
        when the user's phrasing contains explicit temporal signals.
        """
        if intent_type in _ALWAYS_REAL_TIME:
            return False  # handled via structured_storage, not Tavily

        query_lower = query.lower()

        # Explicit temporal signal in the query text
        if any(signal in query_lower for signal in _TEMPORAL_SIGNALS):
            return True

        # Current year mentioned
        if str(datetime.now().year) in query:
            return True

        return False

    # ------------------------------------------------------------------
    # Data-fetching helpers
    # ------------------------------------------------------------------

    def _fetch_real_time_parallel(
        self, query: str, intent_type: IntentType
    ) -> Optional[Dict]:
        """Build search queries and execute them in parallel via Tavily."""
        if not self.tavily_service:
            return None

        base_query = self._enhance_query_for_real_time(query, intent_type)
        current_year = str(datetime.now().year)

        searches = [base_query]
        if intent_type == IntentType.FEES_QUERY:
            searches.append(
                f"Strathmore University tuition fees {current_year}"
            )
        elif intent_type == IntentType.ADMISSION_QUERY:
            searches.append(
                f"Strathmore University admission requirements {current_year}"
            )

        all_results: List[Dict] = []

        def _search(q: str) -> List[Dict]:
            try:
                result = self.tavily_service.search(
                    query=q,
                    max_results=4,
                    search_depth="basic",
                    include_answer=True,
                )
                return result.get("results", [])
            except Exception as exc:
                logger.warning("Tavily search failed for '%s': %s", q, exc)
                return []

        with ThreadPoolExecutor(max_workers=len(searches)) as pool:
            futures = {pool.submit(_search, q): q for q in searches[:2]}
            for future in as_completed(futures):
                all_results.extend(future.result())

        if not all_results:
            return None

        # Deduplicate by URL, sort by relevance
        seen: Dict[str, Dict] = {}
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen:
                seen[url] = r

        sorted_results = sorted(
            seen.values(),
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )

        return {"results": sorted_results[:5], "query": base_query}

    def _enhance_query_for_real_time(
        self, query: str, intent_type: IntentType
    ) -> str:
        query_lower = query.lower()
        if "strathmore" not in query_lower:
            query = f"Strathmore University {query}"
        current_year = str(datetime.now().year)
        if current_year not in query and str(datetime.now().year - 1) not in query:
            query = f"{query} {current_year}"
        return query

    def _get_vector_context(
        self, query: str, intent_type: IntentType, limit: int
    ) -> List[Dict]:
        try:
            params: Dict[str, Any] = {
                "query": query,
                "top_k": limit,
                "use_hybrid": True,
                "include_real_time": True,
            }
            if intent_type == IntentType.SCHEDULE_QUERY:
                params["filter_metadata"] = {"doc_type": "schedule"}
            return self.vector_db.search(**params)
        except Exception as e:
            logger.error("Vector context retrieval failed: %s", e)
            return []

    def _query_schedule_structured(self, query: str) -> Optional[Dict]:
        if not self.structured_storage:
            return None
        try:
            result = self.structured_storage.query_with_natural_language(query)
            if result.get("success") and result.get("result_count", 0) > 0:
                logger.info(
                    "Structured schedule returned %d entries",
                    result["result_count"],
                )
                return result
        except Exception as e:
            logger.warning("Structured schedule query failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    def _generate_direct_response(
        self,
        query: str,
        intent_type: IntentType,
        vector_context: List[Dict],
        real_time_data: Optional[Dict],
        structured_schedule: Optional[Dict] = None,
        context_info: Optional[Dict] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:

        system_instruction = """You are an expert assistant for Strathmore University students, \
staff, and prospective students.

RESPONSE RULES:
- Answer directly and completely — no preamble or filler phrases.
- If the provided sources contain the answer, use them and cite the source inline \
  (e.g., "According to [Source 2]…" or "The schedule database shows…").
- If the sources do NOT contain sufficient information to answer the question, say \
  "I don't have reliable information about that" — do NOT invent fees, dates, names, \
  or policies.
- Prioritise sources in this order: SCHEDULE DATABASE > CURRENT/REAL-TIME > UNIVERSITY DOCS.
- Include specific details (costs, deadlines, room numbers, contact info) when present in sources.
- Use bullet points or numbered lists for procedures and multi-part answers.
- Bold key facts (amounts, deadlines, room numbers).

CITATION FORMAT: Reference sources inline as [Source N] or [Schedule DB] or [Real-time].
"""

        # --- Build context block with token budget enforcement ---
        context_parts: List[str] = []
        token_budget = _CONTEXT_TOKEN_BUDGET

        if structured_schedule and structured_schedule.get("results"):
            header = (
                f"=== SCHEDULE DATABASE (AUTHORITATIVE) ===\n"
                f"SQL: {structured_schedule.get('sql_query', 'direct query')}\n"
                f"Entries found: {structured_schedule['result_count']}\n"
            )
            rows: List[str] = []
            for entry in structured_schedule["results"][:30]:
                parts = [
                    entry.get("class_group", ""),
                    entry.get("subject", ""),
                ]
                if entry.get("unit_code"):
                    parts.append(f"({entry['unit_code']})")
                parts += [
                    entry.get("day", ""),
                    f"{entry.get('start_time', '')}–{entry.get('end_time', '')}",
                    f"Room: {entry.get('room', '')}",
                    f"Instructor: {entry.get('instructor', '')}",
                    f"Type: {entry.get('session_type', '')}",
                    f"Semester: {entry.get('semester', '')}",
                ]
                rows.append(" | ".join(p for p in parts if p.strip()))
            block = header + "\n".join(rows)
            cost = _estimate_tokens(block)
            if cost <= token_budget:
                context_parts.append(block)
                token_budget -= cost

        if real_time_data and real_time_data.get("results"):
            header = "=== CURRENT/REAL-TIME INFORMATION ===\n"
            snippets: List[str] = []
            for i, r in enumerate(real_time_data["results"][:4], 1):
                snippet = (
                    f"[Source {i}] {r.get('title', '')}\n"
                    f"Published: {r.get('published_date', 'unknown')}\n"
                    f"{r.get('content', '')[:600]}\n"
                    f"URL: {r.get('url', '')}"
                )
                snippets.append(snippet)
            block = header + "\n\n".join(snippets)
            cost = _estimate_tokens(block)
            if cost <= token_budget:
                context_parts.append(block)
                token_budget -= cost

        if vector_context:
            header = "=== UNIVERSITY DOCUMENTATION ===\n"
            chunks: List[str] = []
            for i, ctx in enumerate(vector_context[:12], 1):
                chunk = (
                    f"[Source {i}] (relevance={ctx.get('score', 0):.2f}, "
                    f"type={ctx.get('metadata', {}).get('doc_type', 'doc')})\n"
                    f"File: {ctx.get('source', 'unknown')}\n"
                    f"{ctx.get('text', '')[:500]}"
                )
                cost = _estimate_tokens(chunk)
                if cost > token_budget:
                    break
                chunks.append(chunk)
                token_budget -= cost
            if chunks:
                context_parts.append(header + "\n\n".join(chunks))

        context_text = (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant sources found in the knowledge base for this query."
        )

        # --- Intent-specific coverage checklist ---
        intent_requirements = {
            IntentType.FEES_QUERY: (
                "Cover: exact amounts (if in sources), payment deadlines, "
                "payment methods, scholarships/financial aid, late-payment policy, "
                "contact for fee queries."
            ),
            IntentType.ADMISSION_QUERY: (
                "Cover: entry requirements, application deadlines, required documents, "
                "application fees, selection criteria, key dates, admissions contact."
            ),
            IntentType.SCHEDULE_QUERY: (
                "Cover: specific class times, room numbers, instructor names. "
                "List every matching entry from the schedule database."
            ),
            IntentType.PROCEDURAL_QUERY: (
                "Cover: numbered step-by-step process, required documents, "
                "where to go, timelines, costs, what to do if problems arise."
            ),
            IntentType.NAVIGATION_QUERY: (
                "Cover: building name, floor/room number, directions from a landmark, "
                "opening hours if relevant."
            ),
            IntentType.FACTUAL_QUERY: (
                "Cover: complete answer with context, related useful information, "
                "contact details for follow-up."
            ),
        }

        user_parts = [
            f"QUESTION: {query}",
            f"INTENT: {intent_type.value}",
        ]
        if intent_type in intent_requirements:
            user_parts.append(f"REQUIRED COVERAGE: {intent_requirements[intent_type]}")
        user_parts += ["", "SOURCES:", context_text]

        user_prompt = "\n".join(user_parts)

        # --- Build message list, injecting conversation history ---
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]
        if conversation_history:
            # Keep last 10 turns to stay within budget
            messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return {
                "content": content,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Batch / stats helpers (unchanged interface, preserved for callers)
    # ------------------------------------------------------------------

    def force_real_time_response(self, query: str) -> Dict[str, Any]:
        if not self.tavily_service:
            return {
                "response": "Real-time service not available. Configure TAVILY_API_KEY.",
                "error": "No real-time service",
                "real_time_used": False,
            }
        try:
            intent_info = self.intent_recognizer.recognize_intent(query)
            intent_type: IntentType = intent_info["intent_type"]
            real_time_data = self._fetch_real_time_parallel(query, intent_type)
            if not real_time_data:
                return {
                    "response": f"No current information found for: {query}",
                    "real_time_used": True,
                    "real_time_results": 0,
                }
            response_content = self._generate_direct_response(
                query=query,
                intent_type=intent_type,
                vector_context=[],
                real_time_data=real_time_data,
            )
            return {
                "response": response_content["content"],
                "real_time_used": True,
                "real_time_results": len(real_time_data.get("results", [])),
                "intent_type": intent_type.value,
                "approach": "real_time_only",
                "token_usage": response_content.get("token_usage", {}),
            }
        except Exception as e:
            return {
                "response": f"Error in real-time processing: {e}",
                "error": str(e),
                "real_time_used": False,
            }

    def batch_process_queries(
        self, queries: List[str], use_real_time: bool = True
    ) -> List[Dict[str, Any]]:
        results = []
        for i, query in enumerate(queries, 1):
            logger.info("Processing query %d/%d: %s…", i, len(queries), query[:50])
            try:
                result = self.generate_response(query, use_real_time=use_real_time)
                result["query_index"] = i
                result["query"] = query
                results.append(result)
            except Exception as e:
                logger.error("Failed to process query %d: %s", i, e)
                results.append({
                    "query_index": i,
                    "query": query,
                    "response": f"Processing failed: {e}",
                    "error": str(e),
                    "real_time_used": False,
                })
        return results

    def get_response_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache_entries": len(self._cache._entries),
            "services": {
                "vector_db": self.vector_db is not None,
                "embedding": self.embedding_service is not None,
                "tavily": self.tavily_service is not None,
                "intent_recognizer": self.intent_recognizer is not None,
            },
        }
        if self.vector_db:
            try:
                stats["vector_db_stats"] = self.vector_db.get_collection_stats()
            except Exception:
                pass
        if self.embedding_service:
            try:
                stats["embedding_stats"] = self.embedding_service.get_embedding_stats()
            except Exception:
                pass
        if self.tavily_service:
            try:
                stats["tavily_stats"] = self.tavily_service.get_cache_stats()
            except Exception:
                pass
        return stats

    def get_real_time_integration_stats(self) -> Dict[str, Any]:
        return {
            "real_time_service_available": self.tavily_service is not None,
            "temporal_signals": len(_TEMPORAL_SIGNALS),
            "always_real_time_intents": [i.value for i in _ALWAYS_REAL_TIME],
            "real_time_model": "Tavily API",
            "parallel_searches": True,
            "semantic_cache_enabled": True,
            "cache_size": len(self._cache._entries),
        }
