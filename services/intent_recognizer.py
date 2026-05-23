import re
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intent_recognizer")


class IntentType(str, Enum):
    FACTUAL_QUERY = "factual_query"
    PROCEDURAL_QUERY = "procedural_query"
    EXPLANATION_QUERY = "explanation_query"
    COMPARISON_QUERY = "comparison_query"
    OFF_TOPIC = "off_topic"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    GENERAL_CHAT = "general_chat"
    TIME_SENSITIVE_QUERY = "time_sensitive_query"
    CONTEXTUAL_REFERENCE = "contextual_reference"


class TopicCategory(str, Enum):
    ACADEMICS = "academics"
    ADMISSIONS = "admissions"
    FEES = "fees"
    FACILITIES = "facilities"
    POLICIES = "policies"
    STUDENT_LIFE = "student_life"
    EVENTS = "events"
    GENERAL = "general"
    SCHEDULE = "schedule"
    OTHER = "other"


class IntentRecognizer:
    def __init__(self):
        self.intent_patterns = {
            IntentType.FACTUAL_QUERY: [
                r"what is",
                r"what are",
                r"who is",
                r"who are",
                r"when is",
                r"when are",
                r"where is",
                r"where are",
                r"how many",
                r"how much",
                r"is there",
                r"^can",
                r"^do",
                r"^does",
            ],
            IntentType.PROCEDURAL_QUERY: [
                r"how (do|can|to|should)",
                r"what (steps|process)",
                r"procedure for",
                r"steps to",
                r"guidelines for",
                r"process of",
                r"apply for",
                r"register for",
            ],
            IntentType.EXPLANATION_QUERY: [
                r"why",
                r"explain",
                r"reason",
                r"elaborate",
                r"clarify",
                r"what happens if",
                r"what does it mean",
            ],
            IntentType.COMPARISON_QUERY: [
                r"compare",
                r"difference between",
                r"versus",
                r"vs",
                r"similarities between",
                r"better",
                r"prefer",
                r"advantage",
                r"disadvantage",
            ],
            IntentType.FEEDBACK: [
                r"thank",
                r"helpful",
                r"appreciate",
                r"good answer",
                r"makes sense",
                r"understood",
                r"got it",
                r"thanks",
            ],
            IntentType.GENERAL_CHAT: [
                r"^hi\b",
                r"^hello\b",
                r"^hey\b",
                r"^greetings",
                r"^how are you",
                r"nice to meet",
                r"good (morning|afternoon|evening)",
            ],
            IntentType.TIME_SENSITIVE_QUERY: [
                r"class(es)? (today|remaining|left)",
                r"schedule (today|now|remaining)",
                r"what('s| is) next",
                r"upcoming (class|classes)",
                r"what do I have (today|left|now)",
                r"time(table)?",
                r"(today|tomorrow|current) class(es)?",
                r"class(es)? (for|on) today",
            ],
            IntentType.CONTEXTUAL_REFERENCE: [
                r"(his|her|their) (predecessor|successor)",
                r"what about (monday|tuesday|wednesday|thursday|friday)",
                r"what about (it|that|this|them|those)",
                r"(that|this) (course|class|program)",
                r"(same|similar) (thing|process|requirement)",
                r"for (it|that|this|them)",
                r"about (it|that|this|them)",
            ],
        }

        self.topic_keywords = {
            TopicCategory.ACADEMICS: [
                "course",
                "program",
                "degree",
                "class",
                "lecture",
                "semester",
                "faculty",
                "credit",
                "grade",
                "gpa",
                "academic",
                "professor",
                "exam",
                "test",
                "assignment",
                "study",
                "research",
                "thesis",
                "dissertation",
                "graduation",
            ],
            TopicCategory.ADMISSIONS: [
                "admission",
                "application",
                "apply",
                "enrollment",
                "entry",
                "requirements",
                "qualification",
                "eligibility",
                "transfer",
                "accept",
                "reject",
                "offer",
            ],
            TopicCategory.FEES: [
                "fee",
                "tuition",
                "payment",
                "cost",
                "expense",
                "financial",
                "scholarship",
                "grant",
                "loan",
                "aid",
                "funding",
                "bursary",
                "discount",
                "installment",
            ],
            TopicCategory.FACILITIES: [
                "library",
                "lab",
                "cafeteria",
                "hostel",
                "dorm",
                "accommodation",
                "residence",
                "housing",
                "wifi",
                "internet",
                "computer",
                "sports",
                "gym",
                "field",
                "court",
            ],
            TopicCategory.POLICIES: [
                "policy",
                "rule",
                "regulation",
                "code",
                "conduct",
                "discipline",
                "penalty",
                "attendance",
                "absence",
                "leave",
                "suspension",
                "expulsion",
                "plagiarism",
                "academic",
                "misconduct",
                "appeal",
                "complaint",
                "grievance",
                "rights",
                "obligations",
                "deadline",
                "extension",
                "postponement",
            ],
            TopicCategory.STUDENT_LIFE: [
                "club",
                "society",
                "association",
                "activity",
                "event",
                "party",
                "festival",
                "ceremony",
                "volunteer",
                "service",
                "community",
                "mentoring",
                "counseling",
                "welfare",
                "health",
                "medical",
                "career",
                "job",
                "internship",
                "placement",
            ],
            TopicCategory.EVENTS: [
                "orientation",
                "graduation",
                "convocation",
                "seminar",
                "workshop",
                "conference",
                "competition",
                "exhibition",
                "fair",
                "ceremony",
                "celebration",
                "meeting",
            ],
            TopicCategory.SCHEDULE: [
                "timetable",
                "schedule",
                "class",
                "lecture",
                "today",
                "tomorrow",
                "now",
                "next",
                "remaining",
                "left",
                "time",
                "session",
                "period",
                "upcoming",
                "week",
                "day",
                "morning",
                "afternoon",
                "evening",
            ],
        }

        # Compile patterns for faster matching
        self.compiled_intent_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_intent_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def recognize_intent(
        self, query: str, conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced intent recognition that considers conversation context.

        Args:
            query: User query
            conversation_context: Optional context from memory processor

        Returns:
            Intent information including context-aware adjustments
        """
        # Clean query
        query = query.strip().lower()

        # Check for clarification intent first
        clarification_indicators = [
            "you mentioned",
            "you said",
            "what about",
            "tell me more about",
            "elaborate on",
            "explain more",
            "what do you mean by",
            "can you clarify",
        ]

        for indicator in clarification_indicators:
            if indicator in query:
                return {
                    "intent_type": IntentType.CLARIFICATION,
                    "confidence": 0.8,
                    "topic": self._determine_topic(query),
                    "requires_context": True,
                }

        # Check if conversation context indicates this should be contextual
        if (
            False
            and conversation_context
            and conversation_context.get("has_context")
        ):
            # If memory system detected references, boost contextual intent
            if conversation_context.get("context_is_relevant"):
                contextual_score = 0.0
                for intent, patterns in self.compiled_intent_patterns.items():
                    if intent == IntentType.CONTEXTUAL_REFERENCE:
                        for pattern in patterns:
                            if pattern.search(query):
                                contextual_score = 0.9
                                break

                if contextual_score > 0:
                    return {
                        "intent_type": IntentType.CONTEXTUAL_REFERENCE,
                        "confidence": contextual_score,
                        "topic": self._determine_topic(query),
                        "requires_context": True,
                        "memory_confidence": conversation_context.get(
                            "relevance_score", 0
                        ),
                    }

        # Standard intent recognition
        intent_scores = {}
        for intent, patterns in self.compiled_intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(query):
                    score += 1

            if score > 0:
                intent_scores[intent] = score / len(patterns)

        # Determine most likely intent
        if not intent_scores:
            intent_type = IntentType.FACTUAL_QUERY
            confidence = 0.5
        else:
            intent_type = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[intent_type]

        # Check if query is off-topic
        is_off_topic, topic = self._check_if_off_topic(query)

        if is_off_topic:
            intent_type = IntentType.OFF_TOPIC
            confidence = 0.7

        # Determine if this intent typically requires context
        requires_context = intent_type in [
            IntentType.CLARIFICATION,
            IntentType.CONTEXTUAL_REFERENCE,
            IntentType.EXPLANATION_QUERY,
        ]

        result = {
            "intent_type": intent_type,
            "confidence": confidence,
            "topic": topic,
            "requires_context": requires_context,
        }

        # Add memory-related metadata if available
        if False and conversation_context:
            result["memory_available"] = conversation_context.get(
                "has_context", False
            )
            result["memory_relevance"] = conversation_context.get(
                "relevance_score", 0
            )
            if conversation_context.get("needs_clarification"):
                result["intent_type"] = IntentType.CLARIFICATION
                result["clarification_needed"] = True

        return result

    def _determine_topic(self, query: str) -> TopicCategory:
        """Determine the topic category of a query."""
        topic_counts = {}
        query_words = set(re.findall(r"\b\w+\b", query.lower()))

        for topic, keywords in self.topic_keywords.items():
            matches = sum(
                1
                for keyword in keywords
                if keyword in query_words or keyword in query.lower()
            )
            if matches > 0:
                topic_counts[topic] = matches

        if not topic_counts:
            return TopicCategory.GENERAL

        return max(topic_counts, key=topic_counts.get)

    def _check_if_off_topic(self, query: str) -> tuple[bool, TopicCategory]:
        """Check if a query is off-topic."""
        is_off_topic = False
        topic = self._determine_topic(query)

        off_topic_indicators = [
            "NASA",
            "SpaceX",
            "World Cup",
            "Olympics",
            "United Nations",
            "President of USA",
            "European Union",
            "Marvel",
            "Disney",
            "Hollywood",
            "Bitcoin",
            "NFT",
            "PlayStation",
            "Xbox",
            "Nintendo",
            "Apple",
            "Google",
            "Tesla",
            "Amazon",
            "Facebook",
        ]

        education_terms = [
            "student",
            "university",
            "college",
            "course",
            "professor",
            "lecturer",
            "class",
            "degree",
            "education",
            "academic",
            "school",
            "faculty",
            "study",
            "campus",
            "learning",
            "dean",
            "curriculum",
            "semester",
            "exam",
            "library",
            "assignment",
            "graduation",
            "admission",
            "department",
        ]

        query_lower = query.lower()

        has_off_topic_terms = any(
            term.lower() in query_lower for term in off_topic_indicators
        )
        has_education_terms = any(
            term in query_lower for term in education_terms
        )

        if has_off_topic_terms and not has_education_terms:
            is_off_topic = True

        return is_off_topic, topic

    def should_use_conversation_context(
        self, intent_info: Dict[str, Any]
    ) -> bool:
        """
        Determine if conversation context should be used based on intent.

        Args:
            intent_info: Result from recognize_intent()

        Returns:
            bool: Whether to use conversation context
        """
        # Always use context for these intents
        contextual_intents = [
            IntentType.CLARIFICATION,
            IntentType.CONTEXTUAL_REFERENCE,
        ]

        if intent_info["intent_type"] in contextual_intents:
            return True

        # Use context if memory system indicated it's available and relevant
        if (
            False
            and intent_info.get("memory_available")
            and intent_info.get("memory_relevance", 0) > 0.4
        ):
            return True

        # Use context if intent requires it
        if intent_info.get("requires_context", False):
            return True

        return False
