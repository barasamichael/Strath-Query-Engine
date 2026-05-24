"""
Tests for the MemoryProcessor service.

Run with:  python -m pytest tests/test_memory_processor.py -v
or directly: python tests/test_memory_processor.py
"""
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.memory_processor import MemoryProcessor


def _make_msg(content: str, is_user: bool = True) -> dict:
    return {"content": content, "isUserMessage": is_user}


def run_scenarios():
    processor = MemoryProcessor()

    # ------------------------------------------------------------------
    # Scenario 1: Single person mentioned → pronoun resolves unambiguously
    # ------------------------------------------------------------------
    print("=== Scenario 1: Single VC → 'his predecessor' ===")
    messages = [
        _make_msg("Who is the VC?"),
        _make_msg("The Vice Chancellor is Dr. Ruth Kiraka.", is_user=False),
        _make_msg("What are the hostel fees?"),
        _make_msg("Hostel fees are 150,000 KSh per semester.", is_user=False),
    ]
    result = processor.process_conversation_context("Who was her predecessor?", messages)
    print(f"  has_context:        {result['has_context']}")
    print(f"  needs_clarification:{result['needs_clarification']}")
    print(f"  resolved_references:{result['resolved_references']}")
    print(f"  instructions:       {result['context_instructions']}")
    print()

    # ------------------------------------------------------------------
    # Scenario 2a: Gender disambiguates — should resolve directly, not ask
    # Dr. Ruth Kiraka (female) + Prof. John Mbugua (male) → "his" = John Mbugua
    # ------------------------------------------------------------------
    print("=== Scenario 2a: VC (female) + Dean (male) → 'his' resolves directly ===")
    messages2a = [
        _make_msg("Who is the VC?"),
        _make_msg("The VC is Dr. Ruth Kiraka.", is_user=False),
        _make_msg("Who is the dean of students?"),
        _make_msg("The dean of students is Prof. John Mbugua.", is_user=False),
    ]
    result2a = processor.process_conversation_context("Who was his predecessor?", messages2a)
    print(f"  needs_clarification:{result2a['needs_clarification']}  (expected: False — gender disambiguates)")
    print(f"  resolved_to:        {[r.get('entity') for r in result2a.get('resolved_references', [])]}")
    print()

    # ------------------------------------------------------------------
    # Scenario 2b: Two male people mentioned → genuinely ambiguous pronoun
    # ------------------------------------------------------------------
    print("=== Scenario 2b: Two male people → 'his predecessor' is ambiguous ===")
    messages2b = [
        _make_msg("Who is the dean of the School of Computing?"),
        _make_msg("The dean is Prof. James Kahiigi.", is_user=False),
        _make_msg("Who is the registrar?"),
        _make_msg("The registrar is Mr. Peter Odhiambo.", is_user=False),
    ]
    result2b = processor.process_conversation_context("Who was his predecessor?", messages2b)
    print(f"  needs_clarification:{result2b['needs_clarification']}  (expected: True — both male, equally recent)")
    print(f"  clarification:      {result2b.get('clarification_prompt', '')}")
    print()

    # ------------------------------------------------------------------
    # Scenario 3: Course schedule follow-up
    # ------------------------------------------------------------------
    print("=== Scenario 3: ICS 3.1E Monday → 'What about Tuesday?' ===")
    messages3 = [
        _make_msg("What classes does BICS 3A have on Monday?"),
        _make_msg(
            "BICS 3A Monday: Data Structures (ICS 3104) at 9:00 AM in LT1, "
            "Algorithms at 2:00 PM in Lab 4.",
            is_user=False,
        ),
    ]
    result3 = processor.process_conversation_context("What about Tuesday?", messages3)
    print(f"  context_relevant:   {result3['context_is_relevant']}")
    print(f"  resolved_references:{result3['resolved_references']}")
    print(f"  instructions:       {result3['context_instructions']}")
    print()

    # ------------------------------------------------------------------
    # Scenario 4: Independent query — no context needed
    # ------------------------------------------------------------------
    print("=== Scenario 4: Independent question — no references ===")
    messages4 = [
        _make_msg("What are the library hours?"),
        _make_msg("The library is open 7 AM – 11 PM on weekdays.", is_user=False),
    ]
    result4 = processor.process_conversation_context(
        "What are the admission requirements for BBIT?", messages4
    )
    print(f"  has_context:        {result4['has_context']}")
    print(f"  context_relevant:   {result4['context_is_relevant']}")
    print()

    # ------------------------------------------------------------------
    # Scenario 5: Empty history
    # ------------------------------------------------------------------
    print("=== Scenario 5: Empty conversation history ===")
    result5 = processor.process_conversation_context("What is the fee structure?", [])
    print(f"  has_context:        {result5['has_context']}")
    print(f"  memory_available:   {result5['memory_available']}")
    print()


if __name__ == "__main__":
    run_scenarios()
