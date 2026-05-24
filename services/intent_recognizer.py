import os
import re
import json
import logging
from enum import Enum
from typing import Any, Dict, Optional

from openai import OpenAI

logger = logging.getLogger("intent_recognizer")


class IntentType(str, Enum):
    FACTUAL_QUERY = "factual_query"
    PROCEDURAL_QUERY = "procedural_query"
    EXPLANATION_QUERY = "explanation_query"
    COMPARISON_QUERY = "comparison_query"
    SCHEDULE_QUERY = "schedule_query"
    FEES_QUERY = "fees_query"
    ADMISSION_QUERY = "admission_query"
    NAVIGATION_QUERY = "navigation_query"
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
    NAVIGATION = "navigation"
    OTHER = "other"


# Fast regex pre-filters for trivially unambiguous cases (no API call needed).
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|greetings|howdy|good\s+(morning|afternoon|evening))\b",
    re.IGNORECASE,
)
_FEEDBACK_RE = re.compile(
    r"\b(thank(s|\s+you)|appreciate|helpful|makes\s+sense|understood|got\s+it|good\s+answer)\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """You are an intent classification engine for KnowStrath, a question-answering \
assistant for Strathmore University, Nairobi, Kenya.

Classify the user query and return a JSON object with exactly these keys:
{
  "intent_type": "<see options below>",
  "topic": "<see options below>",
  "confidence": <float between 0.0 and 1.0>,
  "is_off_topic": <true or false>,
  "reasoning": "<one sentence>"
}

INTENT TYPE OPTIONS:
- factual_query        : Requests a general fact (what, who, where, when, how many/much) not covered by a more specific type below
- procedural_query     : Asks how to do something — steps, process, procedure
- explanation_query    : Asks why, or requests an explanation/elaboration
- comparison_query     : Compares options, differences, advantages, preferences
- schedule_query       : Specifically about class timetables, lecture schedules, what classes happen on which day/time/room
- fees_query           : Specifically about tuition fees, costs, payments, financial aid, scholarships, bursaries
- admission_query      : Specifically about admission requirements, application process, entry qualifications, enrollment
- navigation_query     : Asks where a place is on campus — office, lab, building, facility location
- time_sensitive_query : Asks about current/upcoming events, announcements, deadlines, or real-time status (not schedules)
- contextual_reference : Refers to something mentioned earlier (it, that, the previous, same)
- clarification        : Asks to clarify or expand on a prior answer
- feedback             : Expresses thanks, satisfaction, or acknowledgment (no question)
- general_chat         : Purely conversational — greetings, small talk, no information request
- off_topic            : Completely unrelated to Strathmore University or university life

CLASSIFICATION PRIORITY: Prefer the most specific type. E.g., a question about tuition amounts is fees_query \
not factual_query; a question about class timetables is schedule_query not time_sensitive_query.

TOPIC OPTIONS:
academics | admissions | fees | facilities | policies | student_life | events | schedule | navigation | general | other

IN-SCOPE definition: Anything related to Strathmore University — its programs, operations, \
facilities, policies, student life, schedules, fees, admissions, and campus events — is IN SCOPE. \
A query that mentions an external brand or entity in an educational context \
(e.g., "Does Strathmore partner with Google?", "Can I use Google Scholar for research?") \
is still IN SCOPE.

OUT-OF-SCOPE: Queries with no connection to Strathmore University or university life \
(sports results, entertainment, general world news, politics unrelated to higher education).

Return only the JSON object. No markdown. No text outside the JSON."""


class IntentRecognizer:
    def __init__(self):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def recognize_intent(
        self, query: str, conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify the intent of a user query.

        Uses regex fast-paths for unambiguous greetings and one-liner feedback,
        then falls back to a gpt-4o-mini call with JSON-mode output for everything
        else. If the LLM call fails, a lightweight regex fallback is used.

        Args:
            query: Raw user query string.
            conversation_context: Reserved for future use (memory system disabled).

        Returns:
            Dict with keys: intent_type, topic, confidence, requires_context,
            and optionally reasoning.
        """
        query = query.strip()
        if not query:
            return self._make_result(IntentType.GENERAL_CHAT, TopicCategory.GENERAL, 1.0)

        # Fast path: pure greeting (single turn, short)
        if _GREETING_RE.match(query) and len(query.split()) <= 6:
            return self._make_result(IntentType.GENERAL_CHAT, TopicCategory.GENERAL, 0.97)

        # Fast path: pure feedback / acknowledgment (short, no question mark)
        if _FEEDBACK_RE.search(query) and "?" not in query and len(query.split()) <= 10:
            return self._make_result(IntentType.FEEDBACK, TopicCategory.GENERAL, 0.95)

        try:
            return self._classify_with_llm(query)
        except Exception as e:
            logger.warning("LLM intent classification failed, using regex fallback: %s", e)
            return self._regex_fallback(query)

    def _classify_with_llm(self, query: str) -> Dict[str, Any]:
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        parsed = json.loads(response.choices[0].message.content)

        # Coerce to known enums, defaulting gracefully on unknown values
        try:
            intent_type = IntentType(parsed.get("intent_type", IntentType.FACTUAL_QUERY))
        except ValueError:
            intent_type = IntentType.FACTUAL_QUERY

        try:
            topic = TopicCategory(parsed.get("topic", TopicCategory.GENERAL))
        except ValueError:
            topic = TopicCategory.GENERAL

        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.7))))

        # LLM is authoritative on off-topic; override intent if flagged
        if parsed.get("is_off_topic", False):
            intent_type = IntentType.OFF_TOPIC

        result = self._make_result(intent_type, topic, confidence)
        result["reasoning"] = parsed.get("reasoning", "")
        return result

    def _regex_fallback(self, query: str) -> Dict[str, Any]:
        """Minimal regex classifier used only when the LLM call fails."""
        q = query.lower()
        if any(w in q for w in ["timetable", "schedule", "class today", "lecture today", "upcoming class", "what time is", "when does class"]):
            intent, topic = IntentType.SCHEDULE_QUERY, TopicCategory.SCHEDULE
        elif any(w in q for w in ["fee", "tuition", "cost", "payment", "scholarship", "bursary", "how much"]):
            intent, topic = IntentType.FEES_QUERY, TopicCategory.FEES
        elif any(w in q for w in ["admission", "apply", "application", "entry requirement", "enroll"]):
            intent, topic = IntentType.ADMISSION_QUERY, TopicCategory.ADMISSIONS
        elif any(w in q for w in ["where is", "location of", "how do i get to", "which building", "directions to"]):
            intent, topic = IntentType.NAVIGATION_QUERY, TopicCategory.NAVIGATION
        elif any(w in q for w in ["how do", "how to", "steps to", "procedure for", "process of"]):
            intent, topic = IntentType.PROCEDURAL_QUERY, TopicCategory.GENERAL
        elif any(w in q for w in ["why", "explain", "reason", "elaborate", "what does it mean"]):
            intent, topic = IntentType.EXPLANATION_QUERY, TopicCategory.GENERAL
        elif any(w in q for w in ["compare", "difference between", "versus", " vs ", "better", "advantage"]):
            intent, topic = IntentType.COMPARISON_QUERY, TopicCategory.GENERAL
        else:
            intent, topic = IntentType.FACTUAL_QUERY, TopicCategory.GENERAL

        result = self._make_result(intent, topic, 0.5)
        result["reasoning"] = "Regex fallback — LLM unavailable"
        return result

    @staticmethod
    def _make_result(
        intent_type: IntentType,
        topic: TopicCategory,
        confidence: float,
    ) -> Dict[str, Any]:
        requires_context = intent_type in {
            IntentType.CLARIFICATION,
            IntentType.CONTEXTUAL_REFERENCE,
            IntentType.EXPLANATION_QUERY,
        }
        return {
            "intent_type": intent_type,
            "topic": topic,
            "confidence": confidence,
            "requires_context": requires_context,
        }

    def should_use_conversation_context(self, intent_info: Dict[str, Any]) -> bool:
        return intent_info.get("requires_context", False)
