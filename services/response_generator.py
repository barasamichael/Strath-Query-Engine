import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pytz

from openai import OpenAI
from services.embeddings import EmbeddingService, IntentType
from services.vector_db import VectorDBService
from services.tavily_service import TavilyService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_generator")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ResponseGenerator:
    """
    Comprehensive response generator with aggressive real-time integration.
    Direct and thorough - provides complete answers without fluff.
    """

    def __init__(
        self,
        vector_db_service: Optional[VectorDBService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        tavily_service: Optional[TavilyService] = None,
    ):
        self.model = "gpt-4o-mini"
        self.temperature = 0.1
        self.max_tokens = 4096

        # Initialize services
        self.vector_db = vector_db_service or VectorDBService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.tavily_service = tavily_service

        # Initialize intent recognition
        if hasattr(self.embedding_service, "initialize_intent_recognition"):
            self.embedding_service.initialize_intent_recognition()

        # Enhanced real-time indicators - more comprehensive
        self.real_time_triggers = [
            "current",
            "latest",
            "recent",
            "today",
            "now",
            "this year",
            "2024",
            "2025",
            "deadline",
            "announcement",
            "news",
            "update",
            "fee",
            "tuition",
            "cost",
            "price",
            "schedule",
            "timetable",
            "admission",
            "application",
            "registration",
            "enrollment",
            "when is",
            "what time",
            "how much",
            "status",
            "available",
        ]

        # Expanded time-sensitive topics that always need real-time data
        self.always_real_time_topics = [
            "fees",
            "tuition",
            "admission",
            "deadlines",
            "schedules",
            "announcements",
            "events",
            "registration",
            "applications",
            "results",
            "grades",
            "timetables",
            "payments",
            "scholarships",
            "bursaries",
            "accommodation",
            "orientation",
        ]

        logger.info(
            f"Enhanced ResponseGenerator initialized - Comprehensive mode: ACTIVE, "
            f"Real-time mode: {'ACTIVE' if self.tavily_service else 'DISABLED'}"
        )

    def get_current_kenya_time(self) -> Tuple[str, str]:
        """Get current time in Kenyan timezone."""
        try:
            import pytz
            from datetime import datetime

            kenya_tz = pytz.timezone("Africa/Nairobi")
            current_time = datetime.now(kenya_tz)

            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S EAT")
            iso_time = current_time.isoformat()

            return formatted_time, iso_time

        except Exception as e:
            logger.error(f"Error getting Kenya time: {e}")
            # Fallback to UTC
            from datetime import datetime

            utc_time = datetime.utcnow()
            formatted_time = utc_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            iso_time = utc_time.isoformat()
            return formatted_time, iso_time

    def generate_response(
        self,
        query: str,
        context_info: Optional[Dict[str, Any]] = None,
        use_real_time: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive, direct response with aggressive real-time integration.
        """
        try:
            start_time = datetime.now()

            # Step 1: Intent recognition (fast)
            intent_type, intent_confidence = self._quick_intent_recognition(
                query
            )

            # Step 2: Determine if we need real-time data AGGRESSIVELY
            needs_real_time = False and self._aggressive_real_time_check(
                query, intent_type
            )

            # Step 3: Parallel data retrieval
            retrieved_context = []
            real_time_data = None

            if needs_real_time and self.tavily_service and use_real_time:
                # AGGRESSIVE real-time first
                real_time_data = self._get_real_time_data(query, intent_type)

            # Always get comprehensive vector database context
            context_limit = 10 if real_time_data else 15
            vector_context = self._get_vector_context(
                query, intent_type, context_limit
            )
            retrieved_context.extend(vector_context)

            # Step 4: Generate comprehensive response
            response_content = self._generate_direct_response(
                query=query,
                intent_type=intent_type,
                vector_context=retrieved_context,
                real_time_data=real_time_data,
                context_info=context_info,
            )

            # Step 5: Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            print(processing_time)

            result = {
                "response": response_content["content"],
                "intent_type": intent_type.value,
                "intent_confidence": intent_confidence,
                "context_sources": len(retrieved_context),
                "real_time_used": real_time_data is not None,
                "real_time_results": (
                    len(real_time_data.get("results", []))
                    if real_time_data
                    else 0
                ),
                "processing_time": processing_time,
                "token_usage": response_content.get("token_usage", {}),
                "approach": (
                    "comprehensive_with_real_time"
                    if real_time_data
                    else "comprehensive_vector_only"
                ),
                "timestamp": datetime.now(
                    pytz.timezone("Africa/Nairobi")
                ).strftime("%Y-%m-%d %H:%M:%S EAT"),
            }

            if real_time_data:
                result["real_time_sources"] = [
                    r.get("url", "") for r in real_time_data.get("results", [])
                ]

            logger.info(
                f"Comprehensive response generated in {processing_time:.2f}s - Real-time: {'YES' if real_time_data else 'NO'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"I encountered an error processing your question about {query}. Please try rephrasing or ask about something else.",
                "error": str(e),
                "intent_type": "error",
                "real_time_used": False,
            }

    def _quick_intent_recognition(self, query: str) -> tuple[IntentType, float]:
        """Fast intent recognition - pattern matching first, embedding fallback."""
        try:
            # Quick pattern-based recognition
            query_lower = query.lower()

            # Schedule queries
            if any(
                word in query_lower
                for word in [
                    "class",
                    "schedule",
                    "timetable",
                    "when",
                    "what time",
                ]
            ):
                if any(
                    word in query_lower
                    for word in [
                        "bics",
                        "ics",
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ]
                ):
                    return IntentType.SCHEDULE_QUERY, 0.9

            # Fee/cost queries
            if any(
                word in query_lower
                for word in [
                    "fee",
                    "cost",
                    "tuition",
                    "pay",
                    "price",
                    "scholarship",
                ]
            ):
                return IntentType.FEES_QUERY, 0.9

            # Admission queries
            if any(
                word in query_lower
                for word in [
                    "admission",
                    "apply",
                    "application",
                    "entry",
                    "requirement",
                ]
            ):
                return IntentType.ADMISSION_QUERY, 0.9

            # Procedural queries
            if any(
                phrase in query_lower
                for phrase in [
                    "how do i",
                    "how to",
                    "process",
                    "procedure",
                    "steps",
                ]
            ):
                return IntentType.PROCEDURAL_QUERY, 0.8

            # Factual queries (default)
            return IntentType.FACTUAL_QUERY, 0.7

        except Exception:
            return IntentType.FACTUAL_QUERY, 0.5

    def _aggressive_real_time_check(
        self, query: str, intent_type: IntentType
    ) -> bool:
        """Aggressively determine if query needs real-time data."""
        query_lower = query.lower()

        # Always use real-time for certain intents
        if intent_type in [
            IntentType.FEES_QUERY,
            IntentType.ADMISSION_QUERY,
            IntentType.SCHEDULE_QUERY,
        ]:
            return True

        # Always use real-time for certain topics
        if any(topic in query_lower for topic in self.always_real_time_topics):
            return True

        # Check for explicit time indicators
        if any(trigger in query_lower for trigger in self.real_time_triggers):
            return True

        # Check for current year or time references
        current_year = str(datetime.now().year)
        if (
            current_year in query
            or "this semester" in query_lower
            or "current semester" in query_lower
        ):
            return True

        return False

    def _get_real_time_data(
        self, query: str, intent_type: IntentType
    ) -> Optional[Dict]:
        """Get real-time data from Tavily with enhanced query strategies."""
        if not self.tavily_service:
            return None

        try:
            # Enhance query based on intent
            enhanced_query = self._enhance_query_for_real_time(
                query, intent_type
            )

            # Use multiple search strategies
            searches = [enhanced_query]

            # Add specific searches based on intent
            if intent_type == IntentType.FEES_QUERY:
                searches.append(
                    f"Strathmore University tuition fees {datetime.now().year}"
                )
                searches.append("Strathmore University fee structure payment")

            elif intent_type == IntentType.ADMISSION_QUERY:
                searches.append(
                    f"Strathmore University admission {datetime.now().year}"
                )
                searches.append(
                    "Strathmore University application requirements"
                )

            elif intent_type == IntentType.SCHEDULE_QUERY:
                searches.append(
                    "Strathmore University academic calendar timetable"
                )

            # Execute searches and combine results
            all_results = []
            for search_query in searches[
                :2
            ]:  # Limit to 2 searches for cost control
                try:
                    result = self.tavily_service.search(
                        query=search_query,
                        max_results=4,  # Increased for more comprehensive data
                        search_depth="basic",
                        include_answer=True,
                    )
                    if result.get("results"):
                        all_results.extend(result["results"])
                except Exception as e:
                    logger.warning(
                        f"Tavily search failed for '{search_query}': {e}"
                    )
                    continue

            if all_results:
                # Deduplicate and rank results
                unique_results = {}
                for result in all_results:
                    url = result.get("url", "")
                    if url not in unique_results:
                        unique_results[url] = result

                # Sort by relevance score
                sorted_results = sorted(
                    unique_results.values(),
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )

                return {
                    "results": sorted_results[:5],  # Increased to top 5 results
                    "query": enhanced_query,
                    "search_count": len(searches),
                }

        except Exception as e:
            logger.error(f"Real-time data retrieval failed: {e}")

        return None

    def _enhance_query_for_real_time(
        self, query: str, intent_type: IntentType
    ) -> str:
        """Enhance query for better real-time search results."""
        query_lower = query.lower()

        # Add Strathmore context if missing
        if "strathmore" not in query_lower:
            query = f"Strathmore University {query}"

        # Add current year for relevance
        current_year = str(datetime.now().year)
        if (
            current_year not in query
            and str(datetime.now().year - 1) not in query
        ):
            query = f"{query} {current_year}"

        # Add intent-specific terms
        intent_enhancements = {
            IntentType.FEES_QUERY: "tuition fees cost payment",
            IntentType.ADMISSION_QUERY: "admission requirements application",
            IntentType.SCHEDULE_QUERY: "timetable schedule academic calendar",
            IntentType.PROCEDURAL_QUERY: "procedure process steps how to",
            IntentType.FACTUAL_QUERY: "information details",
        }

        if intent_type in intent_enhancements:
            enhancement = intent_enhancements[intent_type]
            if not any(term in query_lower for term in enhancement.split()):
                query = f"{query} {enhancement}"

        return query

    def _get_vector_context(
        self, query: str, intent_type: IntentType, limit: int
    ) -> List[Dict]:
        """Get context from vector database with intent-based optimization."""
        try:
            # Adjust search strategy based on intent
            search_params = {
                "query": query,
                "top_k": limit,
                "use_hybrid": True,
                "include_real_time": True,
            }

            # Intent-specific adjustments
            if intent_type == IntentType.SCHEDULE_QUERY:
                search_params["filter_metadata"] = {"doc_type": "schedule"}

            return self.vector_db.search(**search_params)

        except Exception as e:
            logger.error(f"Vector context retrieval failed: {e}")
            return []

    def _generate_direct_response(
        self,
        query: str,
        intent_type: IntentType,
        vector_context: List[Dict],
        real_time_data: Optional[Dict],
        context_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive, direct response without fluff."""

        # Enhanced system instruction - COMPREHENSIVE BUT DIRECT
        system_instruction = """You are a comprehensive, expert assistant for Strathmore University students, staff, and prospective students.

RESPONSE PHILOSOPHY:
- Provide complete, thorough answers that anticipate follow-up questions
- Be direct and get straight to the point - no fluff or introductory phrases
- Give users everything they need to know in one response
- Structure information logically from most important to supporting details
- Include specific details, procedures, requirements, deadlines, and contact information when relevant
- When you have current/real-time information, clearly indicate this and prioritize it

COMPREHENSIVE COVERAGE:
- Answer the main question fully
- Include related important information the user likely needs
- Provide step-by-step procedures when applicable
- Include relevant deadlines, costs, requirements, or prerequisites
- Mention contact information, office locations, or next steps when helpful
- Add context that helps users understand the bigger picture
- Include alternative options or related services when relevant

INFORMATION HIERARCHY:
1. Direct answer to the main question (with current data if available)
2. Essential details (costs, deadlines, requirements, procedures)
3. Step-by-step processes if applicable
4. Contact information and locations
5. Related information and alternatives
6. Important notes, warnings, or exceptions

FORMATTING:
- Use clear section breaks for different aspects of information
- Use bullet points or numbered lists for procedures and requirements
- Bold important information like deadlines, costs, and contact details
- Organize complex information in logical sections

Be thorough, informative, and complete while maintaining directness and clarity."""

        # Enhanced context formatting with better organization
        context_parts = []

        # Real-time information first (clearly marked and prioritized)
        if real_time_data and real_time_data.get("results"):
            context_parts.append("=== CURRENT/LATEST INFORMATION ===")
            for i, result in enumerate(real_time_data["results"][:4], 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")[
                    :600
                ]  # Increased content length
                url = result.get("url", "No URL")
                published = result.get("published_date", "")
                context_parts.append(f"CURRENT SOURCE {i}: {title}")
                if published:
                    context_parts.append(f"Published: {published}")
                context_parts.append(f"Content: {content}")
                context_parts.append(f"URL: {url}")
                context_parts.append("")

        # Vector database context (organized by relevance)
        if vector_context:
            context_parts.append(
                "=== UNIVERSITY DOCUMENTATION & KNOWLEDGE BASE ==="
            )
            for i, ctx in enumerate(
                vector_context[:12], 1
            ):  # Increased context
                score = ctx.get("score", 0)
                text = ctx.get("text", "")[:500]  # Increased text length
                source = ctx.get("source", "Unknown source")
                doc_type = ctx.get("metadata", {}).get("doc_type", "document")

                context_parts.append(
                    f"SOURCE {i} (relevance: {score:.2f}, type: {doc_type})"
                )
                context_parts.append(f"Source: {source}")
                context_parts.append(f"Content: {text}")
                context_parts.append("")

        context_text = (
            "\n".join(context_parts)
            if context_parts
            else "Limited information available - provide general guidance."
        )

        # Enhanced user prompt with more specific instructions
        user_prompt_parts = [
            f"QUESTION: {query}",
            f"INTENT TYPE: {intent_type.value}",
            f"CONTEXT SOURCES: {len(vector_context)} university documents",
        ]

        if real_time_data:
            user_prompt_parts.append(
                f"REAL-TIME SOURCES: {len(real_time_data.get('results', []))} current sources available"
            )

        # Add intent-specific response requirements
        intent_requirements = {
            IntentType.FEES_QUERY: """
REQUIRED COVERAGE FOR FEES:
- Exact fee amounts (current year)
- Payment deadlines and schedules
- Payment methods and procedures
- Available scholarships or financial aid
- Late payment consequences
- Contact information for fee inquiries""",
            IntentType.ADMISSION_QUERY: """
REQUIRED COVERAGE FOR ADMISSIONS:
- Entry requirements and qualifications
- Application deadlines and procedures
- Required documents and how to submit
- Application fees and payment methods
- Selection criteria and process
- Important dates and timelines
- Contact information for admissions office""",
            IntentType.SCHEDULE_QUERY: """
REQUIRED COVERAGE FOR SCHEDULES:
- Specific class times and locations
- Academic calendar dates
- Registration periods
- Exam schedules
- Important academic deadlines
- How to access personal timetables""",
            IntentType.PROCEDURAL_QUERY: """
REQUIRED COVERAGE FOR PROCEDURES:
- Complete step-by-step process
- Required documents or prerequisites
- Where to go and who to contact
- Timelines and deadlines
- Costs involved (if any)
- Alternative methods or options
- What to do if problems arise""",
            IntentType.FACTUAL_QUERY: """
REQUIRED COVERAGE FOR FACTUAL QUESTIONS:
- Complete answer with context
- Related important information
- Practical implications for the user
- Where to find more detailed information
- Contact details for further assistance""",
        }

        if intent_type in intent_requirements:
            user_prompt_parts.append(intent_requirements[intent_type])

        user_prompt_parts.extend(
            [
                "",
                "AVAILABLE INFORMATION:",
                context_text,
                "",
                """INSTRUCTIONS:
Provide a comprehensive, complete response that covers all aspects the user needs to know. 
Include specific details, procedures, costs, deadlines, and contact information when available.
Structure the response clearly with sections if needed.
If you have current/real-time information, clearly indicate this.
Anticipate and answer likely follow-up questions.
Be thorough but maintain directness - no introductory fluff.""",
            ]
        )

        user_prompt = "\n".join(user_prompt_parts)

        # Increase max tokens for more comprehensive responses
        max_tokens_for_response = min(
            self.max_tokens * 2, 4000
        )  # Allow longer responses

        # Generate response with enhanced parameters
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens_for_response,  # Increased token limit
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
            logger.error(f"Response generation failed: {e}")
            raise

    def get_real_time_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about real-time integration usage."""
        return {
            "real_time_service_available": self.tavily_service is not None,
            "real_time_triggers": len(self.real_time_triggers),
            "always_real_time_topics": len(self.always_real_time_topics),
            "real_time_model": "Tavily API",
            "cache_enabled": (
                hasattr(self.tavily_service, "search_cache")
                if self.tavily_service
                else False
            ),
            "cache_ttl": (
                getattr(self.tavily_service, "cache_ttl", 0)
                if self.tavily_service
                else 0
            ),
        }

    def force_real_time_response(self, query: str) -> Dict[str, Any]:
        """Force a response using only real-time data - for testing."""
        if not self.tavily_service:
            return {
                "response": "Real-time service not available. Please configure TAVILY_API_KEY.",
                "error": "No real-time service",
                "real_time_used": False,
            }

        try:
            intent_type, confidence = self._quick_intent_recognition(query)
            real_time_data = self._get_real_time_data(query, intent_type)

            if not real_time_data:
                return {
                    "response": f"No current information found for: {query}",
                    "real_time_used": True,
                    "real_time_results": 0,
                }

            response_content = self._generate_direct_response(
                query=query,
                intent_type=intent_type,
                vector_context=[],  # No vector context - pure real-time
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
                "response": f"Error in real-time processing: {str(e)}",
                "error": str(e),
                "real_time_used": False,
            }

    def batch_process_queries(
        self, queries: List[str], use_real_time: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently."""
        results = []

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")

            try:
                result = self.generate_response(
                    query, use_real_time=use_real_time
                )
                result["query_index"] = i
                result["query"] = query
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to process query {i}: {e}")
                results.append(
                    {
                        "query_index": i,
                        "query": query,
                        "response": f"Processing failed: {str(e)}",
                        "error": str(e),
                        "real_time_used": False,
                    }
                )

        return results

    def get_response_stats(self) -> Dict[str, Any]:
        """Get comprehensive response generator statistics."""
        stats = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "services": {
                "vector_db_available": self.vector_db is not None,
                "embedding_service_available": self.embedding_service
                is not None,
                "tavily_service_available": self.tavily_service is not None,
            },
            "real_time_config": {
                "triggers": self.real_time_triggers,
                "always_real_time_topics": self.always_real_time_topics,
                "aggressive_mode": True,
            },
            "response_style": "comprehensive_direct",
        }

        # Add service-specific stats
        if self.vector_db:
            try:
                vector_stats = self.vector_db.get_collection_stats()
                stats["vector_db_stats"] = vector_stats
            except:
                pass

        if self.embedding_service:
            try:
                embedding_stats = self.embedding_service.get_embedding_stats()
                stats["embedding_stats"] = embedding_stats
            except:
                pass

        if self.tavily_service:
            try:
                tavily_stats = self.tavily_service.get_cache_stats()
                stats["tavily_stats"] = tavily_stats
            except:
                pass

        return stats
