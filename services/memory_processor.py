import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import spacy
import tiktoken
from openai import OpenAI
from spacy.matcher import Matcher

logger = logging.getLogger("memory_processor")

# ---------------------------------------------------------------------------
# spaCy label → domain category
# ORG is handled separately: some ORGs are course codes / class groups.
# ---------------------------------------------------------------------------
_LABEL_MAP: Dict[str, str] = {
    "PERSON": "person",
    "GPE": "place",
    "FAC": "place",
    "LOC": "place",
    "DATE": "time",
    "TIME": "time",
    "EVENT": "event",
    "ORG": "org",
}

# Regex to identify ORG tokens that are actually academic codes
_COURSE_CODE_RE = re.compile(r"^[A-Z]{2,5}\s*[\d][\d.]*[A-Z]?$")
_CLASS_GROUP_RE = re.compile(r"^[A-Z]{2,5}\s*\d[A-Z]$")

# Pronouns and vague references that signal a coreference need
_PRONOUNS = frozenset({
    "he", "she", "they", "him", "her", "them",
    "his", "hers", "their", "it", "its",
})
_VAGUE_REFS = frozenset({
    "this", "that", "these", "those",
    "predecessor", "successor", "same", "similar",
})
_FOLLOWUP_PHRASES = (
    "what about",
    "tell me more about",
    "more about",
    "elaborate on",
)


@dataclass
class ExtractedEntity:
    text: str
    category: str       # person / place / academic / time / event / org
    spacy_label: str
    message_index: int
    context_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.text,
            "entity_type": self.category,
            "message_index": self.message_index,
            "context_snippet": self.context_snippet,
        }


class MemoryProcessor:
    """
    Conversation memory processor.

    Entity extraction: spaCy en_core_web_sm NER + custom Matcher for
    Strathmore-specific tokens (course codes like ICS 3104, class groups like BICS 3A).

    Coreference resolution: gpt-4o-mini JSON call that resolves pronouns and
    vague references ("his", "it", "that class") to named entities from history.

    Context window: tiktoken-aware, respecting max_context_tokens.
    """

    def __init__(
        self,
        max_context_tokens: int = 2000,
        memory_window: int = 15,
        ambiguity_threshold: float = 0.15,
    ):
        self.max_context_tokens = max_context_tokens
        self.memory_window = memory_window
        self.ambiguity_threshold = ambiguity_threshold

        self._nlp = spacy.load("en_core_web_sm")
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._encoder = tiktoken.encoding_for_model("gpt-4o-mini")

        self._matcher = Matcher(self._nlp.vocab)
        self._register_domain_patterns()

        logger.info("MemoryProcessor initialised (spaCy NER + OpenAI coreference + tiktoken)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_conversation_context(
        self,
        current_query: str,
        messages: List[Dict[str, Any]],
        max_history_messages: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Analyse conversation history and resolve references in current_query.

        Returns a dict compatible with the ContextInfo schema used by the API.
        """
        try:
            if not messages:
                return self._minimal_context()

            window = messages[-(max_history_messages or self.memory_window):]

            # 1. Extract named entities from conversation history via spaCy
            entities = self._extract_entities(window)

            # 2. Check if the query contains references worth resolving
            ref_words = self._detect_reference_words(current_query)

            if not ref_words:
                if self._is_followup(current_query):
                    return self._recent_context(window, reason="Follow-up pattern detected")
                return self._minimal_context()

            # 3. No history entities → can't resolve references, return recent context
            if not entities:
                return self._recent_context(
                    window, reason="References detected but no prior entities found"
                )

            # 4. Resolve coreferences via OpenAI
            resolution = self._resolve_coreferences(current_query, window, entities)

            if resolution.get("needs_clarification"):
                return {
                    "has_context": True,
                    "context_is_relevant": True,
                    "needs_clarification": True,
                    "clarification_prompt": resolution.get("clarification_prompt", ""),
                    "relevance_score": 0.8,
                    "relevance_reasons": ["Ambiguous reference — clarification needed"],
                    "resolved_references": resolution.get("candidates", []),
                    "context_instructions": resolution.get("clarification_prompt", ""),
                    "memory_available": True,
                    "comprehensive_context_used": True,
                }

            if resolution.get("resolved") and resolution.get("resolutions"):
                return self._build_resolved_context(window, entities, resolution)

            # Resolution found nothing → fall back to recent context
            return self._recent_context(
                window, reason="References detected but could not be resolved"
            )

        except Exception as e:
            logger.error("MemoryProcessor error: %s", e, exc_info=True)
            return self._minimal_context(error=str(e))

    # ------------------------------------------------------------------
    # Entity extraction — spaCy NER + domain Matcher
    # ------------------------------------------------------------------

    def _register_domain_patterns(self) -> None:
        """Register Strathmore-specific token patterns with the spaCy Matcher."""
        # Course codes: ICS 3104, MATH 101, HED 2201, ICS 3.1E
        self._matcher.add("COURSE_CODE", [
            [
                {"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}},
                {"TEXT": {"REGEX": r"^\d[\d.]*[A-Z]?$"}},
            ],
        ])
        # Class groups: BICS 3A, BICS 2B
        self._matcher.add("CLASS_GROUP", [
            [
                {"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}},
                {"TEXT": {"REGEX": r"^\d[A-Z]$"}},
            ],
        ])

    def _extract_entities(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, ExtractedEntity]:
        """
        Run spaCy NER and domain Matcher on all messages.

        Returns a dict keyed by lowercased entity text. Later mentions overwrite
        earlier ones so the most recent context wins on duplicates.
        """
        entity_map: Dict[str, ExtractedEntity] = {}

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not content:
                continue

            doc = self._nlp(content)

            # Standard named entities
            for ent in doc.ents:
                category = self._map_label(ent.label_, ent.text)
                if category is None:
                    continue
                snippet = content[max(0, ent.start_char - 30): ent.end_char + 30].strip()
                entity_map[ent.text.lower()] = ExtractedEntity(
                    text=ent.text,
                    category=category,
                    spacy_label=ent.label_,
                    message_index=i,
                    context_snippet=snippet,
                )

            # Strathmore-specific patterns (course codes, class groups)
            for match_id, start, end in self._matcher(doc):
                span = doc[start:end]
                label = self._nlp.vocab.strings[match_id]
                snippet = content[max(0, span.start_char - 30): span.end_char + 30].strip()
                entity_map[span.text.lower()] = ExtractedEntity(
                    text=span.text,
                    category="academic",
                    spacy_label=label,
                    message_index=i,
                    context_snippet=snippet,
                )

        return entity_map

    def _find_entity(
        self, resolves_to: str, entities: Dict[str, "ExtractedEntity"]
    ) -> Optional["ExtractedEntity"]:
        """
        Look up an entity by the name the LLM returned.

        Tries three passes so title prefixes and partial names don't cause misses:
          1. Exact lowercase key match  ("ruth kiraka" → "ruth kiraka")
          2. Map key is contained in the LLM name  ("ruth kiraka" ⊂ "dr. ruth kiraka")
          3. LLM name is contained in the map key  ("john" ⊂ "john mbugua")
        Returns the highest-confidence (most recently updated) match.
        """
        needle = resolves_to.lower().strip()
        if not needle:
            return None

        # Pass 1: exact
        if needle in entities:
            return entities[needle]

        # Pass 2 & 3: substring containment — collect all matches, take the
        # one whose map key is longest (most specific)
        candidates = []
        for key, entity in entities.items():
            if key in needle or needle in key:
                candidates.append((len(key), entity))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        return None

    @staticmethod
    def _map_label(label: str, text: str) -> Optional[str]:
        """Map a spaCy label to a domain category; returns None to skip the entity."""
        if label == "ORG":
            normalised = re.sub(r"\s+", " ", text.strip())
            if _COURSE_CODE_RE.match(normalised) or _CLASS_GROUP_RE.match(normalised):
                return "academic"
            return "org"
        return _LABEL_MAP.get(label)

    # ------------------------------------------------------------------
    # Reference detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_reference_words(query: str) -> List[str]:
        """Return pronoun/vague-reference tokens found in the query."""
        words = set(query.lower().split())
        found = list(words & (_PRONOUNS | _VAGUE_REFS))
        for phrase in _FOLLOWUP_PHRASES:
            if phrase in query.lower():
                found.append(phrase)
        return found

    @staticmethod
    def _is_followup(query: str) -> bool:
        q = query.lower()
        return any(phrase in q for phrase in _FOLLOWUP_PHRASES) or q.startswith("also ")

    # ------------------------------------------------------------------
    # Coreference resolution — OpenAI
    # ------------------------------------------------------------------

    def _resolve_coreferences(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        entities: Dict[str, ExtractedEntity],
    ) -> Dict[str, Any]:
        """
        Ask gpt-4o-mini to resolve what each pronoun / vague reference in query
        points to, given conversation history and the extracted entity list.
        """
        history_lines = []
        for msg in messages[-8:]:
            role = "Student" if msg.get("isUserMessage", True) else "Assistant"
            content = msg.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")

        entity_list = [
            f"{e.text} ({e.category})" for e in entities.values()
        ]

        prompt = (
            "You resolve references in a conversation about Strathmore University.\n\n"
            "CONVERSATION (most recent last):\n"
            + "\n".join(history_lines)
            + "\n\nENTITIES MENTIONED: "
            + (", ".join(entity_list) if entity_list else "none")
            + f'\n\nCURRENT QUERY: "{query}"\n\n'
            "Identify every reference in the query (pronouns, 'it', 'that', 'this', "
            "'predecessor', 'same', etc.) and determine what each refers to from the "
            "conversation above.\n\n"
            "Return JSON only:\n"
            "{\n"
            '  "resolved": true/false,\n'
            '  "resolutions": [\n'
            '    {"reference": "his", "resolves_to": "Dr. Ruth Kiraka", '
            '"entity_type": "person", "confidence": 0.9}\n'
            "  ],\n"
            '  "needs_clarification": true/false,\n'
            '  "clarification_prompt": "Are you asking about X or Y?",\n'
            '  "candidates": [\n'
            '    {"entity": "Dr. Ruth Kiraka", "entity_type": "person", "message_index": 1}\n'
            "  ],\n"
            '  "reasoning": "one sentence"\n'
            "}\n\n"
            "Set needs_clarification=true only when two equally plausible and recent "
            "candidates exist for the same reference.\n"
            "If one entity is clearly more recent or contextually relevant, resolve directly.\n"
            "If no resolvable references exist, set resolved=false."
        )

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=350,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.warning("Coreference resolution call failed: %s", e)
            return {"resolved": False, "needs_clarification": False, "resolutions": []}

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_resolved_context(
        self,
        messages: List[Dict[str, Any]],
        entities: Dict[str, ExtractedEntity],
        resolution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a full context dict after successful coreference resolution."""
        resolutions = resolution.get("resolutions", [])

        # Collect message indices for all resolved entities
        relevant_indices: set = set()
        for res in resolutions:
            entity = self._find_entity(res.get("resolves_to", ""), entities)
            if entity:
                idx = entity.message_index
                relevant_indices.update(
                    range(max(0, idx - 1), min(len(messages), idx + 2))
                )

        # Always include the last 3 messages for conversational continuity
        relevant_indices.update(range(max(0, len(messages) - 3), len(messages)))

        selected = [messages[i] for i in sorted(relevant_indices) if i < len(messages)]
        context_messages = self._build_token_window(selected)

        resolved_refs = []
        for res in resolutions:
            entity = self._find_entity(res.get("resolves_to", ""), entities)
            resolved_refs.append({
                "entity": res.get("resolves_to", ""),
                "entity_type": res.get("entity_type", ""),
                "reference": res.get("reference", ""),
                "confidence": float(res.get("confidence", 0.7)),
                "message_index": entity.message_index if entity else 0,
                "context_snippet": entity.context_snippet if entity else "",
            })

        instruction_parts = [
            f"'{r['reference']}' refers to {r['entity']} ({r['entity_type']})"
            for r in resolved_refs
            if r["entity"]
        ]
        instructions = ". ".join(instruction_parts)
        if instructions:
            instructions += ". Use this context to answer the query."
        else:
            instructions = "Use resolved context to answer the query."

        top_confidence = max(
            (r["confidence"] for r in resolved_refs), default=0.7
        )

        return {
            "has_context": True,
            "context_is_relevant": True,
            "needs_clarification": False,
            "relevance_score": top_confidence,
            "relevance_reasons": [
                f"Resolved: '{r['reference']}' → {r['entity']}" for r in resolved_refs
            ],
            "context_messages": context_messages,
            "formatted_context": self._format_context(context_messages, resolved_refs),
            "resolved_references": resolved_refs,
            "context_instructions": instructions,
            "memory_available": True,
            "comprehensive_context_used": True,
            "total_messages": len(messages),
            "context_message_count": len(context_messages),
        }

    def _recent_context(
        self, messages: List[Dict[str, Any]], reason: str = ""
    ) -> Dict[str, Any]:
        """Lightweight context from the most recent messages (follow-ups)."""
        context_messages = self._build_token_window(messages[-4:])
        return {
            "has_context": True,
            "context_is_relevant": True,
            "needs_clarification": False,
            "relevance_score": 0.6,
            "relevance_reasons": [reason] if reason else ["Recent context included"],
            "context_messages": context_messages,
            "formatted_context": self._format_context(context_messages, []),
            "resolved_references": [],
            "context_instructions": "Use recent conversation as context for the follow-up query.",
            "memory_available": True,
            "comprehensive_context_used": False,
            "total_messages": len(messages),
            "context_message_count": len(context_messages),
        }

    def _minimal_context(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return the minimal context structure for independent queries."""
        result: Dict[str, Any] = {
            "has_context": False,
            "context_is_relevant": False,
            "needs_clarification": False,
            "relevance_score": 0.0,
            "relevance_reasons": [],
            "context_messages": [],
            "formatted_context": "",
            "resolved_references": [],
            "context_instructions": "Independent query — no prior context needed.",
            "memory_available": False,
            "comprehensive_context_used": False,
            "total_messages": 0,
            "context_message_count": 0,
        }
        if error:
            result["error"] = error
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _build_token_window(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select as many messages as fit within max_context_tokens,
        picking from newest backwards.
        """
        result: List[Dict[str, Any]] = []
        budget = self.max_context_tokens

        for msg in reversed(messages):
            content = msg.get("content", "")
            cost = len(self._encoder.encode(content))
            if cost > budget:
                break
            budget -= cost
            result.insert(0, {
                "role": "user" if msg.get("isUserMessage", True) else "assistant",
                "content": content,
            })

        return result

    @staticmethod
    def _format_context(
        messages: List[Dict[str, Any]],
        resolved_refs: List[Dict[str, Any]],
    ) -> str:
        parts = []

        if resolved_refs:
            summaries = [
                f"{r['reference']} → {r['entity']} ({r['entity_type']})"
                for r in resolved_refs
                if r.get("entity")
            ]
            if summaries:
                parts.append("Resolved references: " + ", ".join(summaries))

        if messages:
            parts.append("Conversation history:")
            for msg in messages:
                role = "Student" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                if len(content) > 150:
                    content = content[:150] + "..."
                parts.append(f"{role}: {content}")

        return "\n".join(parts)
