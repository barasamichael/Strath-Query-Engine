"""
Smart Query Router for KnowStrath RAG System

This service intelligently routes queries between structured schedule queries
and semantic search based on query analysis and intent recognition.
"""

import re
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from services.intent_recognizer import IntentType

logger = logging.getLogger("query_router")


class QueryType(Enum):
    """Types of queries the router can handle."""

    STRUCTURED_SCHEDULE = "structured_schedule"
    SEMANTIC_SEARCH = "semantic_search"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Results of query analysis."""

    query_type: QueryType
    confidence: float
    structured_patterns: List[str]
    schedule_entities: List[str]
    requires_real_time: bool
    explanation: str


class SmartQueryRouter:
    """Intelligently routes queries to appropriate processing systems."""

    def __init__(
        self, structured_storage, vector_db_service, intent_recognizer
    ):
        """
        Initialize the query router.

        Args:
            structured_storage: StructuredDataStorage instance
            vector_db_service: VectorDBService instance
            intent_recognizer: IntentRecognizer instance
        """
        self.structured_storage = structured_storage
        self.vector_db_service = vector_db_service
        self.intent_recognizer = intent_recognizer

        # Schedule-specific query patterns
        self.structured_patterns = {
            "time_query": [
                r"what time.*(?:class|lecture|lab)",
                r"when (?:is|does).*(?:start|begin|happen)",
                r"(?:start|begin|end) time.*(?:for|of)",
                r"schedule.*(?:for|of).*(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday)",
            ],
            "room_query": [
                r"(?:which|what) room.*(?:is|for)",
                r"where (?:is|does).*(?:class|lecture|lab)",
                r"location.*(?:of|for)",
                r"room.*(?:for|of)",
            ],
            "availability_query": [
                r"(?:which|what).*(?:room|class).*(?:free|available)",
                r"(?:free|available).*(?:room|at|during)",
                r"(?:empty|vacant).*room",
                r"room.*(?:available|free).*(?:at|during)",
            ],
            "instructor_query": [
                r"who (?:teaches|is teaching)",
                r"(?:teacher|instructor|lecturer).*(?:for|of)",
                r"taught by.*who",
                r"Dr\.?\s+\w+.*(?:teaches|class)",
            ],
            "class_schedule": [
                r"(?:schedule|timetable).*(?:for|of).*(?:BICS|class)",
                r"(?:BICS|class).*(?:\d[ABC]).*(?:schedule|timetable|class)",
                r"what (?:classes|subjects).*(?:does|has).*(?:BICS|\d[ABC])",
                r"(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday).*class",
            ],
            "subject_query": [
                r"(?:ICS|HED|MATH|STAT)\s*\d{4}",
                r"(?:object oriented|database|network|calculus|algebra).*(?:class|course)",
                r"when.*(?:object oriented|database|network|calculus|algebra)",
            ],
        }

        # Schedule entities that indicate structured queries
        self.schedule_entities = [
            # Class groups
            r"BICS\s+\d[ABC]",
            # Time references
            r"\d{1,2}[:\.]\d{2}(?:\s*[AP]M)?",
            r"(?:morning|afternoon|evening)",
            r"(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            # Rooms
            r"(?:LT|Lab|Room|MSB|SLS|STMB|Lecture Theatre)\s*\d*",
            # Course codes
            r"(?:ICS|HED|MATH|STAT)\s*\d{4}",
            # Instructors
            r"Dr\.?\s+[A-Z][a-z]+",
        ]

        # Keywords that indicate semantic search is better
        self.semantic_keywords = [
            "explain",
            "definition",
            "what is",
            "how to",
            "why",
            "policy",
            "requirement",
            "procedure",
            "process",
            "admission",
            "fee",
            "cost",
            "payment",
            "scholarship",
            "facilities",
            "library",
            "accommodation",
            "housing",
            "club",
            "society",
            "event",
            "ceremony",
            "graduation",
        ]

        # Compile patterns for efficiency
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_structured_patterns = {}
        for category, patterns in self.structured_patterns.items():
            self.compiled_structured_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        self.compiled_schedule_entities = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.schedule_entities
        ]

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to determine the best processing approach.

        Args:
            query: User query string

        Returns:
            QueryAnalysis with routing decision
        """
        query_lower = query.lower()

        # Check for structured patterns
        structured_matches = self._find_structured_patterns(query)
        structured_score = len(structured_matches)

        # Check for schedule entities
        entity_matches = self._find_schedule_entities(query)
        entity_score = len(entity_matches)

        # Check for semantic indicators
        semantic_score = sum(
            1 for keyword in self.semantic_keywords if keyword in query_lower
        )

        # Check if query requires real-time information
        requires_real_time = self._check_real_time_requirements(query)

        # Calculate confidence and determine query type
        total_structured_score = structured_score * 2 + entity_score

        if total_structured_score >= 3 and semantic_score <= 1:
            query_type = QueryType.STRUCTURED_SCHEDULE
            confidence = min(0.9, 0.5 + (total_structured_score * 0.1))
            explanation = f"Strong schedule-specific patterns detected: {structured_matches}"

        elif total_structured_score >= 1 and semantic_score >= 2:
            query_type = QueryType.HYBRID
            confidence = 0.6
            explanation = "Mixed indicators suggest hybrid approach needed"

        elif semantic_score > total_structured_score:
            query_type = QueryType.SEMANTIC_SEARCH
            confidence = min(0.9, 0.5 + (semantic_score * 0.1))
            explanation = f"Semantic keywords detected: {[kw for kw in self.semantic_keywords if kw in query_lower]}"

        elif total_structured_score > 0:
            query_type = QueryType.STRUCTURED_SCHEDULE
            confidence = 0.7
            explanation = f"Some schedule patterns found: {structured_matches}"

        else:
            query_type = QueryType.UNKNOWN
            confidence = 0.3
            explanation = (
                "No clear patterns detected, defaulting to semantic search"
            )

        return QueryAnalysis(
            query_type=query_type,
            confidence=confidence,
            structured_patterns=structured_matches,
            schedule_entities=entity_matches,
            requires_real_time=requires_real_time,
            explanation=explanation,
        )

    def _find_structured_patterns(self, query: str) -> List[str]:
        """Find structured query patterns in the query."""
        matches = []

        for category, patterns in self.compiled_structured_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    matches.append(category)
                    break  # Only count each category once

        return matches

    def _find_schedule_entities(self, query: str) -> List[str]:
        """Find schedule-related entities in the query."""
        matches = []

        for pattern in self.compiled_schedule_entities:
            found = pattern.findall(query)
            matches.extend(found)

        return matches

    def _check_real_time_requirements(self, query: str) -> bool:
        """Check if query requires real-time schedule information."""
        real_time_indicators = [
            "now",
            "currently",
            "at the moment",
            "right now",
            "today",
            "this morning",
            "this afternoon",
            "this evening",
        ]

        query_lower = query.lower()
        return any(
            indicator in query_lower for indicator in real_time_indicators
        )

    # Canned responses for intents that must not touch the vector DB
    _CANNED = {
        IntentType.GENERAL_CHAT: (
            "Hello! I'm the KnowStrath assistant for Strathmore University. "
            "Feel free to ask me anything about admissions, fees, courses, facilities, "
            "policies, schedules, or student life."
        ),
        IntentType.FEEDBACK: (
            "You're welcome! Let me know if there's anything else I can help you with."
        ),
        IntentType.OFF_TOPIC: (
            "I'm here to help with questions about Strathmore University — "
            "admissions, fees, courses, facilities, policies, schedules, and student life. "
            "I'm not able to assist with topics outside that scope. "
            "Is there something university-related I can help you with?"
        ),
    }

    def route_query(
        self, query: str, conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the appropriate processing system.

        Args:
            query: User query string
            conversation_context: Optional conversation context

        Returns:
            Processing results with routing information
        """
        logger.info("Analyzing query: %s", query)

        # Classify intent first — short-circuit before any DB access for
        # greetings, feedback, and out-of-scope queries.
        intent_info = self.intent_recognizer.recognize_intent(query, conversation_context)
        intent_type = intent_info["intent_type"]

        if intent_type in self._CANNED:
            logger.info(
                "Short-circuit: intent=%s confidence=%.2f",
                intent_type,
                intent_info["confidence"],
            )
            return {
                "answer": self._CANNED[intent_type],
                "intent_info": intent_info,
                "search_results": [],
                "result_count": 0,
                "approach": "canned",
                "routing_info": {
                    "primary_approach": "canned",
                    "intent_type": intent_type,
                    "reason": "Query does not require knowledge base lookup",
                },
            }

        # Analyse structural patterns for remaining intents
        analysis = self.analyze_query(query)

        logger.info(
            "Query type: %s (confidence: %.2f) — %s",
            analysis.query_type.value,
            analysis.confidence,
            analysis.explanation,
        )

        # Route based on analysis
        if analysis.query_type == QueryType.STRUCTURED_SCHEDULE:
            return self._handle_structured_query(
                query, analysis, conversation_context
            )

        elif analysis.query_type == QueryType.HYBRID:
            return self._handle_hybrid_query(
                query, analysis, conversation_context
            )

        elif analysis.query_type == QueryType.SEMANTIC_SEARCH:
            return self._handle_semantic_query(
                query, analysis, conversation_context
            )

        else:  # UNKNOWN
            # Default to semantic search with fallback to structured
            semantic_result = self._handle_semantic_query(
                query, analysis, conversation_context
            )

            # If semantic search returns few results, try structured as fallback
            if len(semantic_result.get("search_results", [])) < 3:
                logger.info("Low semantic results, trying structured fallback")
                structured_result = self._try_structured_fallback(query)
                if (
                    structured_result
                    and structured_result.get("result_count", 0) > 0
                ):
                    return {
                        **structured_result,
                        "routing_info": {
                            "primary_approach": "semantic",
                            "fallback_used": "structured",
                            "analysis": analysis.__dict__,
                        },
                    }

            return {
                **semantic_result,
                "routing_info": {
                    "primary_approach": "semantic",
                    "analysis": analysis.__dict__,
                },
            }

    def _handle_structured_query(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Handle queries with structured data approach."""
        logger.info("Using STRUCTURED query processing")

        try:
            result = self.structured_storage.query_with_natural_language(query)

            if result["success"] and result["result_count"] > 0:
                return {
                    "answer": result["formatted_response"],
                    "sql_query": result["sql_query"],
                    "search_results": result["results"],
                    "result_count": result["result_count"],
                    "approach": "structured",
                    "routing_info": {
                        "primary_approach": "structured",
                        "analysis": analysis.__dict__,
                    },
                }
            else:
                logger.info("Structured query returned no results, falling back to semantic search")
                return self._handle_semantic_query(query, analysis, context)

        except Exception as e:
            logger.error(f"Structured query failed: {e}")
            return self._handle_semantic_query(query, analysis, context)

    def _handle_hybrid_query(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Handle queries that benefit from both approaches."""
        logger.info("Using HYBRID query processing")

        try:
            # Try structured first
            structured_result = (
                self.structured_storage.query_with_natural_language(query)
            )

            # Get semantic results
            semantic_results = self._get_semantic_results(query, top_k=10)

            # Combine results
            combined_answer = self._combine_structured_and_semantic(
                query, structured_result, semantic_results
            )

            return {
                "answer": combined_answer,
                "structured_results": structured_result.get("results", []),
                "semantic_results": semantic_results,
                "sql_query": structured_result.get("sql_query"),
                "result_count": len(structured_result.get("results", []))
                + len(semantic_results),
                "approach": "hybrid",
                "routing_info": {
                    "primary_approach": "hybrid",
                    "analysis": analysis.__dict__,
                },
            }

        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")
            return self._handle_semantic_query(query, analysis, context)

    def _handle_semantic_query(
        self,
        query: str,
        analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Handle queries with semantic search approach."""
        logger.info("Using SEMANTIC search processing")

        try:
            # Get semantic search results
            search_results = self.vector_db_service.multi_query_search(
                query, top_k=15
            )

            return {
                "search_results": search_results,
                "result_count": len(search_results),
                "approach": "semantic",
                "routing_info": {
                    "primary_approach": "semantic",
                    "analysis": analysis.__dict__,
                },
            }

        except Exception as e:
            logger.error(f"Semantic query failed: {e}")
            return {
                "error": f"Query processing failed: {str(e)}",
                "search_results": [],
                "result_count": 0,
                "approach": "semantic",
                "routing_info": {
                    "primary_approach": "semantic",
                    "analysis": analysis.__dict__,
                    "error": str(e),
                },
            }

    def _try_structured_fallback(self, query: str) -> Optional[Dict[str, Any]]:
        """Try structured query as fallback for unknown queries."""
        try:
            result = self.structured_storage.query_with_natural_language(query)
            if result["success"] and result["result_count"] > 0:
                return result
        except Exception:
            pass
        return None

    def _get_semantic_results(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Get semantic search results."""
        try:
            return self.vector_db_service.multi_query_search(query, top_k=top_k)
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _combine_structured_and_semantic(
        self,
        query: str,
        structured_result: Dict[str, Any],
        semantic_results: List[Dict[str, Any]],
    ) -> str:
        """Combine structured and semantic results into a coherent answer."""

        answer_parts = []

        # Add structured answer if available
        if structured_result.get("success") and structured_result.get(
            "formatted_response"
        ):
            answer_parts.append("**Schedule Information:**")
            answer_parts.append(structured_result["formatted_response"])

        # Add relevant semantic context if available
        if semantic_results:
            # Filter for high-relevance semantic results
            relevant_semantic = [
                r for r in semantic_results[:3] if r.get("score", 0) > 0.7
            ]

            if relevant_semantic:
                answer_parts.append("\n**Additional Context:**")
                for result in relevant_semantic[:2]:  # Limit to top 2
                    text = result.get("text", "")[:200]  # Truncate for brevity
                    if text:
                        answer_parts.append(f"• {text}...")

        return (
            "\n".join(answer_parts)
            if answer_parts
            else "I couldn't find specific information to answer your question."
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get statistics about query routing patterns."""
        # This could be enhanced to track routing decisions over time
        # For now, return basic information about the router

        return {
            "structured_pattern_categories": len(self.structured_patterns),
            "total_structured_patterns": sum(
                len(patterns) for patterns in self.structured_patterns.values()
            ),
            "schedule_entity_patterns": len(self.schedule_entities),
            "semantic_keywords": len(self.semantic_keywords),
            "supports_hybrid_queries": True,
            "fallback_available": True,
        }
