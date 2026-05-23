import re
import logging
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comprehensive_memory_processor")


class ReferenceType(str, Enum):
    """Types of references that can be made in conversation."""

    PERSON = "person"
    PLACE = "place"
    CONCEPT = "concept"
    TIME = "time"
    ACADEMIC = "academic"
    PROCESS = "process"


@dataclass
class ContextualEntity:
    """Enhanced entity with contextual information."""

    entity: str
    entity_type: ReferenceType
    message_index: int
    context_snippet: str
    confidence: float
    related_entities: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type.value,
            "message_index": self.message_index,
            "context_snippet": self.context_snippet,
            "confidence": self.confidence,
            "related_entities": self.related_entities,
            "temporal_context": self.temporal_context,
        }


@dataclass
class SemanticContext:
    """Comprehensive semantic context for a conversation."""

    entities: Dict[ReferenceType, List[ContextualEntity]]
    entity_relationships: Dict[str, List[str]]  # entity -> related entities
    temporal_sequences: List[List[str]]
    topic_evolution: List[Tuple[int, str]]  # (message_index, topic)
    last_updated: int

    def __post_init__(self):
        if not self.entities:
            self.entities = defaultdict(list)
        if not self.entity_relationships:
            self.entity_relationships = defaultdict(list)


class EntityExtractor:
    """Advanced entity extraction with contextual awareness."""

    def __init__(self):
        self.entity_patterns = {
            ReferenceType.PERSON: [
                r"\b(vc|vice.?chancellor|president|dean|director|principal|registrar|librarian)\b",
                r"\b(professor|dr\.?|mr\.?|ms\.?|mrs\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                r"\b(coordinator|head|chairman|chairwoman)\b",
            ],
            ReferenceType.PLACE: [
                r"\b(hostel|dormitory|library|lab|cafeteria|office|building|room|campus|hall)\s*(\w+)?",
                r"\b(strathmore|university|college|school)\b",
                r"\b(venue|location|place)\b",
            ],
            ReferenceType.ACADEMIC: [
                r"\b(class|classes|course|program|degree|major|semester|term)\s*(\w+)?",
                r"\b([A-Z]{2,4}\s*\d+\.?\d*[A-Z]*)\b",  # Course codes like ICS 3.1E, MATH 101
                r"\b(schedule|timetable|curriculum)\b",
                r"\b(lecture|tutorial|lab|practical)\b",
                r"\b(year\s*\d+|semester\s*\d+)\b",
            ],
            ReferenceType.TIME: [
                r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                r"\b(today|tomorrow|yesterday|next\s+week|this\s+week)\b",
                r"\b(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm))\b",  # Times
                r"\b(morning|afternoon|evening|night)\b",
                r"\b(deadline|due\s+date)\b",
            ],
            ReferenceType.CONCEPT: [
                r"\b(fees?|tuition|payment|scholarship|grant|loan|discount)\b",
                r"\b(exam|test|assignment|grade|gpa|result)\b",
                r"\b(admission|registration|enrollment|application)\b",
                r"\b(policy|rule|regulation|requirement)\b",
            ],
            ReferenceType.PROCESS: [
                r"\b(application\s+process|registration\s+process|how\s+to)\b",
                r"\b(step|procedure|method|way)\b",
                r"\b(deadline|timeline|schedule)\b",
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # Temporal relationship patterns
        self.temporal_sequences = [
            [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ],
            ["semester 1", "semester 2", "semester 3"],
            ["year 1", "year 2", "year 3", "year 4"],
            ["morning", "afternoon", "evening"],
        ]

    def extract_entities(
        self, text: str, message_index: int
    ) -> List[ContextualEntity]:
        """Extract entities with enhanced contextual information."""
        entities = []
        text_lower = text.lower()

        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text_lower)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 1:  # Filter very short matches

                        # Extract surrounding context
                        start = max(0, match.start() - 30)
                        end = min(len(text), match.end() + 30)
                        context_snippet = text[start:end].strip()

                        # Calculate confidence based on pattern specificity
                        confidence = self._calculate_confidence(
                            entity_text, entity_type, text
                        )

                        # Extract related entities in the same message
                        related_entities = self._find_related_entities(
                            text_lower, entity_text, entity_type
                        )

                        # Detect temporal context for time-based entities
                        temporal_context = (
                            self._extract_temporal_context(
                                text_lower, entity_text
                            )
                            if entity_type == ReferenceType.TIME
                            else None
                        )

                        entity = ContextualEntity(
                            entity=entity_text,
                            entity_type=entity_type,
                            message_index=message_index,
                            context_snippet=context_snippet,
                            confidence=confidence,
                            related_entities=related_entities,
                            temporal_context=temporal_context,
                        )
                        entities.append(entity)

        return entities

    def _calculate_confidence(
        self, entity: str, entity_type: ReferenceType, full_text: str
    ) -> float:
        """Calculate confidence score for entity extraction."""
        base_confidence = 0.7

        # Boost confidence for specific patterns
        if entity_type == ReferenceType.ACADEMIC and re.match(
            r"[A-Z]{2,4}\s*\d+", entity
        ):
            base_confidence = 0.95  # High confidence for course codes

        if entity_type == ReferenceType.PERSON and any(
            title in entity for title in ["vc", "dean", "professor"]
        ):
            base_confidence = 0.9  # High confidence for clear titles

        if entity_type == ReferenceType.TIME and entity in [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
        ]:
            base_confidence = 0.85  # High confidence for weekdays

        # Reduce confidence if entity appears in a question vs statement
        if "?" in full_text:
            base_confidence *= 0.9

        return min(base_confidence, 1.0)

    def _find_related_entities(
        self, text: str, primary_entity: str, entity_type: ReferenceType
    ) -> List[str]:
        """Find entities that appear in context with the primary entity."""
        related = []

        # For academic entities, look for related academic terms
        if entity_type == ReferenceType.ACADEMIC:
            academic_terms = re.findall(
                r"\b(?:class|course|program|schedule|timetable|year\s*\d+)\b",
                text,
            )
            related.extend(
                [term for term in academic_terms if term != primary_entity]
            )

        # For time entities, look for other time references
        if entity_type == ReferenceType.TIME:
            time_terms = re.findall(
                r"\b(?:monday|tuesday|wednesday|thursday|friday|morning|afternoon|evening)\b",
                text,
            )
            related.extend(
                [term for term in time_terms if term != primary_entity]
            )

        return list(set(related))[:3]  # Limit to 3 most relevant

    def _extract_temporal_context(
        self, text: str, time_entity: str
    ) -> Optional[str]:
        """Extract temporal context for time-based entities."""
        # Look for sequences this entity might be part of
        for sequence in self.temporal_sequences:
            if time_entity in sequence:
                # Find other elements from this sequence in the text
                found_elements = [elem for elem in sequence if elem in text]
                if len(found_elements) > 1:
                    return f"sequence: {sequence}"

        return None


class MemoryProcessor:
    """
    Final comprehensive memory processor that handles all types of contextual references
    including people, academic concepts, schedules, and temporal relationships.
    """

    def __init__(
        self,
        max_context_tokens: int = 2000,
        memory_window: int = 50,
        ambiguity_threshold: float = 0.15,  # Confidence difference threshold for ambiguity
    ):
        self.max_context_tokens = max_context_tokens
        self.memory_window = memory_window
        self.ambiguity_threshold = ambiguity_threshold

        self.entity_extractor = EntityExtractor()
        self._compile_reference_patterns()

        logger.info("MemoryProcessor initialized")

    def _compile_reference_patterns(self):
        """Compile patterns for different types of references."""
        self.reference_patterns = {
            ReferenceType.PERSON: re.compile(
                r"\b(his|her|their|he|she|they|him|them|predecessor|successor)\b",
                re.IGNORECASE,
            ),
            ReferenceType.ACADEMIC: re.compile(
                r"\b(what about|for|on)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
                r"\b(that\s+class|this\s+class|the\s+course|same\s+program)\b",
                re.IGNORECASE,
            ),
            ReferenceType.TIME: re.compile(
                r"\b(what about|next|following|previous|last)\s+(day|week|month|time)\b|"
                r"\b(then|after|before|later)\b",
                re.IGNORECASE,
            ),
            ReferenceType.CONCEPT: re.compile(
                r"\b(it|this|that|these|those|same|similar)\b", re.IGNORECASE
            ),
            ReferenceType.PLACE: re.compile(
                r"\b(there|here|that\s+place|same\s+location)\b", re.IGNORECASE
            ),
        }

    def process_conversation_context(
        self,
        current_query: str,
        messages: List[Dict[str, Any]],
        max_history_messages: Optional[int] = 15,
    ) -> Dict[str, Any]:
        """
        Process conversation with comprehensive semantic understanding.
        """
        try:
            if not messages:
                return self._minimal_context()

            # Build comprehensive semantic context
            semantic_context = self._build_semantic_context(messages)

            # Analyze current query for references
            reference_analysis = self._analyze_query_references(
                current_query, semantic_context
            )

            # Handle different types of references
            if reference_analysis["has_references"]:
                return self._handle_contextual_query(
                    current_query,
                    messages,
                    semantic_context,
                    reference_analysis,
                )
            else:
                # Check for implicit context (like follow-up patterns)
                implicit_context = self._check_implicit_context(
                    current_query, messages[-3:] if messages else []
                )
                if implicit_context["needs_context"]:
                    return self._build_recent_context(
                        current_query, messages[-max_history_messages:]
                    )
                else:
                    return self._minimal_context()

        except Exception as e:
            logger.error(f"Error in comprehensive context processing: {str(e)}")
            return self._minimal_context(error=str(e))

    def _build_semantic_context(
        self, messages: List[Dict[str, Any]]
    ) -> SemanticContext:
        """Build comprehensive semantic context from conversation."""
        context = SemanticContext(
            entities=defaultdict(list),
            entity_relationships=defaultdict(list),
            temporal_sequences=[],
            topic_evolution=[],
            last_updated=0,
        )

        for i, msg_data in enumerate(messages):
            if not msg_data.get("content"):
                continue

            content = msg_data["content"]
            topic = msg_data.get("topic", "general")

            # Extract entities
            entities = self.entity_extractor.extract_entities(content, i)

            for entity in entities:
                context.entities[entity.entity_type].append(entity)

                # Build relationships
                for related in entity.related_entities:
                    context.entity_relationships[entity.entity].append(related)

            # Track topic evolution
            if i == 0 or (
                context.topic_evolution
                and context.topic_evolution[-1][1] != topic
            ):
                context.topic_evolution.append((i, topic))

        context.last_updated = len(messages) - 1
        return context

    def _analyze_query_references(
        self, query: str, semantic_context: SemanticContext
    ) -> Dict[str, Any]:
        """Analyze query for different types of references."""
        analysis = {
            "has_references": False,
            "reference_types": [],
            "potential_targets": defaultdict(list),
            "ambiguity_detected": False,
        }

        query_lower = query.lower()

        # Check each reference type
        for ref_type, pattern in self.reference_patterns.items():
            if pattern.search(query_lower):
                analysis["has_references"] = True
                analysis["reference_types"].append(ref_type)

                # Find potential targets for this reference type
                targets = self._find_reference_targets(
                    query, ref_type, semantic_context
                )
                analysis["potential_targets"][ref_type] = targets

        # Detect ambiguity
        for ref_type, targets in analysis["potential_targets"].items():
            if len(targets) > 1:
                # Check if confidences are too close (ambiguous)
                if len(targets) >= 2:
                    confidence_diff = abs(
                        targets[0].confidence - targets[1].confidence
                    )
                    if confidence_diff < self.ambiguity_threshold:
                        analysis["ambiguity_detected"] = True
                        analysis["ambiguous_type"] = ref_type
                        analysis["ambiguous_targets"] = targets[:2]

        return analysis

    def _find_reference_targets(
        self,
        query: str,
        ref_type: ReferenceType,
        semantic_context: SemanticContext,
    ) -> List[ContextualEntity]:
        """Find potential targets for a reference."""
        candidates = []
        query_lower = query.lower()

        # Get entities of the reference type
        entities = semantic_context.entities.get(ref_type, [])

        for entity in entities:
            confidence = entity.confidence

            # Boost confidence based on specific patterns
            if (
                ref_type == ReferenceType.PERSON
                and "predecessor" in query_lower
            ):
                if any(
                    role in entity.entity
                    for role in ["vc", "dean", "president", "director"]
                ):
                    confidence += 0.2

            elif ref_type == ReferenceType.ACADEMIC:
                # Handle "what about tuesday" type queries
                if any(
                    day in query_lower
                    for day in [
                        "monday",
                        "tuesday",
                        "wednesday",
                        "thursday",
                        "friday",
                    ]
                ):
                    if any(
                        academic in entity.entity
                        for academic in ["class", "course", "schedule", "ics"]
                    ):
                        confidence += 0.25
                    # Check if there's a course code in the entity's context
                    if re.search(r"[A-Z]{2,4}\s*\d+", entity.context_snippet):
                        confidence += 0.3

            elif ref_type == ReferenceType.TIME:
                # Handle temporal sequence references
                if (
                    entity.temporal_context
                    and "sequence" in entity.temporal_context
                ):
                    confidence += 0.2

            # Apply recency bias (more recent = higher confidence)
            recency_boost = 0.1 * (
                1 - (len(entities) - entities.index(entity)) / len(entities)
            )
            confidence += recency_boost

            # Update entity confidence
            entity.confidence = min(confidence, 1.0)
            candidates.append(entity)

        # Sort by confidence and return top candidates
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)[:3]

    def _handle_contextual_query(
        self,
        current_query: str,
        messages: List[Dict[str, Any]],
        semantic_context: SemanticContext,
        reference_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle queries that have contextual references."""

        # Check for ambiguity first
        if reference_analysis["ambiguity_detected"]:
            return self._handle_ambiguous_reference(
                current_query, reference_analysis
            )

        # Build context with resolved references
        resolved_targets = []
        for ref_type, targets in reference_analysis[
            "potential_targets"
        ].items():
            if targets:
                resolved_targets.append(targets[0])  # Take highest confidence

        if not resolved_targets:
            return self._minimal_context()

        # Get relevant message indices
        relevant_indices = set()
        for target in resolved_targets:
            # Include target message and surrounding context
            for i in range(
                max(0, target.message_index - 1),
                min(len(messages), target.message_index + 2),
            ):
                relevant_indices.add(i)

        # Add recent context
        recent_indices = set(range(max(0, len(messages) - 4), len(messages)))
        relevant_indices.update(recent_indices)

        # Build context messages
        context_messages = []
        for idx in sorted(relevant_indices):
            if idx < len(messages) and messages[idx].get("content"):
                context_messages.append(
                    {
                        "role": (
                            "user"
                            if messages[idx].get("isUserMessage", True)
                            else "assistant"
                        ),
                        "content": messages[idx]["content"],
                        "message_index": idx,
                        "is_reference_source": any(
                            t.message_index == idx for t in resolved_targets
                        ),
                    }
                )

        # Format context
        formatted_context = self._format_comprehensive_context(
            context_messages, resolved_targets
        )

        return {
            "has_context": True,
            "context_is_relevant": True,
            "relevance_score": max([t.confidence for t in resolved_targets]),
            "relevance_reasons": [
                f"Resolved {t.entity_type.value} reference: {t.entity}"
                for t in resolved_targets
            ],
            "context_messages": context_messages[-8:],
            "formatted_context": formatted_context,
            "resolved_references": [t.to_dict() for t in resolved_targets],
            "context_instructions": self._get_comprehensive_instructions(
                resolved_targets, reference_analysis
            ),
            "total_messages": len(messages),
            "context_message_count": len(context_messages),
            "comprehensive_context_used": True,
        }

    def _handle_ambiguous_reference(
        self, current_query: str, reference_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle cases where references are ambiguous."""

        ambiguous_targets = reference_analysis.get("ambiguous_targets", [])
        if len(ambiguous_targets) < 2:
            return self._minimal_context()

        # Generate clarification prompt based on reference type
        ref_type = reference_analysis.get("ambiguous_type")
        target1, target2 = ambiguous_targets[0], ambiguous_targets[1]

        if ref_type == ReferenceType.PERSON:
            clarification = f"Are you asking about the {target1.entity} or the {target2.entity}?"
        elif ref_type == ReferenceType.ACADEMIC:
            clarification = (
                f"Are you asking about {target1.entity} or {target2.entity}?"
            )
        else:
            clarification = f"Could you clarify whether you're referring to {target1.entity} or {target2.entity}?"

        return {
            "has_context": True,
            "context_is_relevant": True,
            "needs_clarification": True,
            "clarification_prompt": clarification,
            "ambiguous_references": [t.to_dict() for t in ambiguous_targets],
            "relevance_score": 0.8,
            "relevance_reasons": [
                "Ambiguous reference detected - clarification needed"
            ],
            "context_instructions": f"Ask for clarification: {clarification}",
            "comprehensive_context_used": True,
        }

    def _format_comprehensive_context(
        self,
        context_messages: List[Dict[str, Any]],
        resolved_targets: List[ContextualEntity],
    ) -> str:
        """Format comprehensive context with resolved references."""

        context_parts = []

        # Add reference resolution summary
        if resolved_targets:
            references = [
                f"{t.entity_type.value}: {t.entity}" for t in resolved_targets
            ]
            context_parts.append(
                f"Resolved references: {', '.join(references)}"
            )

        # Add relevant conversation history
        context_parts.append("Relevant conversation history:")

        for msg in context_messages[-6:]:
            role = "Student" if msg["role"] == "user" else "Assistant"
            content = (
                msg["content"][:150] + "..."
                if len(msg["content"]) > 150
                else msg["content"]
            )

            # Mark reference sources
            marker = (
                " [REFERENCE SOURCE]" if msg.get("is_reference_source") else ""
            )
            context_parts.append(f"{role}: {content}{marker}")

        return "\n".join(context_parts)

    def _get_comprehensive_instructions(
        self,
        resolved_targets: List[ContextualEntity],
        reference_analysis: Dict[str, Any],
    ) -> str:
        """Generate comprehensive instructions for context usage."""

        instructions = []

        for target in resolved_targets:
            if target.entity_type == ReferenceType.PERSON:
                if "predecessor" in reference_analysis.get("query_lower", ""):
                    instructions.append(
                        f"Query asks about predecessor of {target.entity}."
                    )
                else:
                    instructions.append(f"Pronoun refers to {target.entity}.")

            elif target.entity_type == ReferenceType.ACADEMIC:
                instructions.append(
                    f"Query relates to {target.entity} from previous context."
                )

            elif target.entity_type == ReferenceType.TIME:
                instructions.append(
                    f"Time reference: {target.entity} in context of previous discussion."
                )

        if not instructions:
            instructions.append("Use resolved context to answer the query.")

        return " ".join(instructions)

    def _check_implicit_context(
        self, query: str, recent_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check for implicit context needs (follow-up patterns)."""

        query_lower = query.lower()

        # Follow-up indicators
        follow_up_patterns = [
            "tell me more",
            "what about",
            "also",
            "additionally",
            "furthermore",
            "more details",
            "elaborate",
        ]

        needs_context = any(
            pattern in query_lower for pattern in follow_up_patterns
        )

        return {
            "needs_context": needs_context,
            "confidence": 0.6 if needs_context else 0.0,
            "reason": (
                "Follow-up pattern detected"
                if needs_context
                else "Independent query"
            ),
        }

    def _build_recent_context(
        self, current_query: str, recent_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context from recent messages for implicit follow-ups."""

        if not recent_messages:
            return self._minimal_context()

        context_messages = []
        for msg_data in recent_messages[-4:]:
            if msg_data.get("content"):
                context_messages.append(
                    {
                        "role": (
                            "user"
                            if msg_data.get("isUserMessage", True)
                            else "assistant"
                        ),
                        "content": (
                            msg_data["content"][:100] + "..."
                            if len(msg_data["content"]) > 100
                            else msg_data["content"]
                        ),
                    }
                )

        return {
            "has_context": True,
            "context_is_relevant": True,
            "relevance_score": 0.6,
            "relevance_reasons": ["Follow-up pattern detected"],
            "context_messages": context_messages,
            "formatted_context": self._format_recent_context(context_messages),
            "context_instructions": "Use recent conversation for follow-up context.",
            "total_messages": len(recent_messages),
            "context_message_count": len(context_messages),
            "comprehensive_context_used": False,
        }

    def _format_recent_context(self, messages: List[Dict[str, Any]]) -> str:
        """Format recent context for follow-up queries."""
        if not messages:
            return ""

        context_parts = ["Recent conversation:"]
        for msg in messages:
            role = "Student" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        return "\n".join(context_parts)

    def _minimal_context(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return minimal context structure for independent queries."""
        result = {
            "has_context": False,
            "context_is_relevant": False,
            "needs_clarification": False,
            "relevance_score": 0.0,
            "relevance_reasons": [],
            "context_messages": [],
            "formatted_context": "",
            "context_instructions": "Independent query - minimal context usage.",
            "total_messages": 0,
            "context_message_count": 0,
            "comprehensive_context_used": False,
        }
        if error:
            result["error"] = error
        return result


# Test scenarios function
def test_comprehensive_scenarios():
    """Test all the scenarios mentioned."""

    processor = MemoryProcessor()

    # Scenario 1: VC then other topics then predecessor
    print("=== Scenario 1: VC → Other topics → Predecessor ===")
    messages1 = [
        {
            "content": "Who is the VC?",
            "isUserMessage": True,
            "topic": "general",
        },
        {"content": "The VC is Dr. Ruth Kiraka", "isUserMessage": False},
        # ... 8 messages about fees ...
        {
            "content": "What are the hostel fees?",
            "isUserMessage": True,
            "topic": "fees",
        },
        {
            "content": "Hostel fees are 150,000 KSh per semester",
            "isUserMessage": False,
        },
    ]

    result1 = processor.process_conversation_context(
        "Who was his predecessor?", messages1
    )
    print(f"Needs clarification: {result1.get('needs_clarification', False)}")
    print(f"Instructions: {result1.get('context_instructions', '')}\n")

    # Scenario 2: Ambiguous reference (VC + Dean)
    print("=== Scenario 2: VC + Dean → Ambiguous Predecessor ===")
    messages2 = [
        {
            "content": "Who is the VC?",
            "isUserMessage": True,
            "topic": "general",
        },
        {"content": "The VC is Dr. Ruth Kiraka", "isUserMessage": False},
        {
            "content": "Who is the dean of students?",
            "isUserMessage": True,
            "topic": "general",
        },
        {
            "content": "The dean of students is Prof. John Mbugua",
            "isUserMessage": False,
        },
    ]

    result2 = processor.process_conversation_context(
        "Who was his predecessor?", messages2
    )
    print(f"Needs clarification: {result2.get('needs_clarification', False)}")
    print(f"Clarification prompt: {result2.get('clarification_prompt', '')}\n")

    # Scenario 3: Class schedule reference
    print("=== Scenario 3: ICS 3.1E Monday → What about Tuesday ===")
    messages3 = [
        {
            "content": "What classes do we have on Monday for ICS 3.1E?",
            "isUserMessage": True,
            "topic": "schedule",
        },
        {
            "content": "ICS 3.1E Monday classes: Data Structures at 9:00 AM, Algorithms at 2:00 PM",
            "isUserMessage": False,
        },
        # ... other topics ...
        {
            "content": "What about the library hours?",
            "isUserMessage": True,
            "topic": "facilities",
        },
    ]

    result3 = processor.process_conversation_context(
        "What about Tuesday?", messages3
    )
    print(f"Context relevant: {result3.get('context_is_relevant', False)}")
    print(
        f"Resolved references: {[r.get('entity', '') for r in result3.get('resolved_references', [])]}"
    )
    print(f"Instructions: {result3.get('context_instructions', '')}")
