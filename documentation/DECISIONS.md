# KnowStrath — Engineering Decisions & Constraints

## Scope

This document records deliberate architectural decisions and hard constraints agreed upon by the team. It exists so future contributors understand **why** things are the way they are and do not accidentally undo intentional choices.

---

## Hard Constraints

### 1. English Only
The system is English-only. No multilingual support (Swahili, Sheng, etc.) will be added until explicitly decided otherwise. Intent classifiers, prompts, keyword lists, and documentation should all assume English input.

### 2. OpenAI API Exclusively
All LLM and embedding calls must go through the OpenAI API (`openai` Python SDK). Claude / Anthropic API calls are not permitted. Current models in use:
- Generation: `gpt-4o-mini`
- Embeddings: `text-embedding-ada-002`
- Intent classification: `gpt-4o-mini` (added May 2026)

### 3. Memory System — Rebuilt, Re-enabling Pending
The conversation memory system (`MemoryProcessor`) was previously disabled after causing reliability issues during a live demo. The implementation has been completely rewritten (May 2026) and is now ready for integration testing before being re-enabled at the API layer.

**Do not re-enable at the API layer** (`api/main.py` — the `use_memory` guard) until the test suite in `tests/test_memory_processor.py` passes end-to-end with a real OpenAI key.

---

## Architecture Decisions

### Intent Recognition — LLM-Based (May 2026)
**Old approach:** Pure regex pattern matching with a 19-item hardcoded off-topic blocklist. Confidence was a meaningless ratio (matched patterns / total patterns).

**New approach:** `gpt-4o-mini` with JSON-mode structured output classifies intent, topic, and off-topic flag in a single call. A regex fast-path handles unambiguous greetings and one-liner feedback without an API call.

**Why:** Regex cannot handle paraphrasing, indirect phrasing, or brand mentions in an educational context (e.g., "Does Strathmore partner with Google?"). The LLM understands scope semantically.

**Cost:** ~$0.000015 per classification call at gpt-4o-mini pricing — negligible relative to generation cost.

### Memory Processor — spaCy NER + OpenAI Coreference (May 2026)
**Old approach:** Pure regex entity extraction running on lowercased text with patterns that required uppercase characters — silently broken for all real names and course codes. Coreference resolution ("his", "that class") used pattern matching with no semantic understanding. Test code lived in the production module.

**New approach:**
- **spaCy `en_core_web_sm`** handles NER (PERSON, ORG, GPE, DATE, TIME, FAC, etc.) correctly without requiring pre-lowercased text.
- A custom spaCy **Matcher** adds Strathmore-specific patterns: course codes (`ICS 3104`, `HED 2201`) and class groups (`BICS 3A`).
- **gpt-4o-mini JSON call** resolves pronouns and vague references ("his", "it", "that class") to named entities from history — semantically, not with regex.
- **tiktoken** counts tokens accurately to respect the `max_context_tokens` budget.
- Entity objects are never mutated in place; confidence scores come from the LLM, not from hardcoded magic numbers.
- Test scenarios moved to `tests/test_memory_processor.py`.

**Cost:** one gpt-4o-mini call (≤350 tokens) per query that contains a pronoun or vague reference. Queries with no references bypass the LLM call entirely.

### Query Routing — Intent-Gated (May 2026)
Greetings (`GENERAL_CHAT`), thank-you messages (`FEEDBACK`), and out-of-scope queries (`OFF_TOPIC`) now short-circuit the router before any vector DB or structured DB lookup. Previously all queries hit the vector DB regardless of intent.

### Off-Topic Detection — Scope-Described (May 2026)
**Old approach:** Blocklist of 19 brand/entity names. Missed everything not on the list; incorrectly rejected valid queries mentioning listed brands.

**New approach:** The LLM prompt defines the in-scope boundary by description ("anything related to Strathmore University — programs, operations, facilities, policies, student life, schedules, fees, admissions, events"). The model determines relevance contextually.

---

## 8.2 Commit Message Format

Every commit message must follow this structure exactly. A subject line, a blank line, and a body. The body is not optional. Every commit must have all three parts.

```
type(scope): short description in present tense

Body explaining what changed and why. Not what the code does line by line,
but why the change was made, what problem it solves, what was broken before,
or what decision was taken. Use as many lines as needed. Wrap at 72
characters per line.
```

**Types:** `feat`, `fix`, `test`, `refactor`, `docs`, `chore`

**Scope:** the module or area of the system being changed, lowercase with hyphens. Examples: `auth`, `file-upload`, `folder-nav`, `access-control`, `notice-board`, `task-management`, `audit-log`, `backup`, `dashboard`, `library`, `user-management`.

**Subject line rules:**
- Present tense, lowercase after the closing parenthesis
- No full stop at the end
- Maximum 72 characters
- Must describe what changes, not how

**Body rules:**
- Separated from the subject by exactly one blank line
- Explains why the change was made, what problem existed before, and what decision was taken
- Where a change closes or addresses a requirement, reference the requirement ID (example: `closes FRS-AUTH-004`)
- Maximum 72 characters per line
- Must be present on every commit without exception

**Example:**

```
feat(auth): add otp verification step to password reset flow

Password reset previously sent a link that expired after 30 minutes,
which caused friction when users did not check email immediately.
Changed to a six-digit OTP entered on the same screen as the reset
form so the flow completes without navigating away from the page.

OTP is generated using secrets.randbelow, stored as a bcrypt hash,
and expires after ten minutes. Marked as used immediately on success
to prevent reuse.

closes FRS-AUTH-004
```

**Rules for AI coding agents — non-negotiable:**

AI agents must not include any reference to themselves, their name, or their involvement in any commit message. This includes but is not limited to phrases such as "generated by", "created by Claude", "AI-assisted", "co-authored by", or any variation thereof. Commit messages must read as authored by the developer and contain no indication of AI involvement.

AI agents must use the author identity already configured in the local git configuration. They must not set or override the `--author` flag, must not add a `Co-authored-by` trailer, and must not modify `.git/config` or any git identity setting. The commit must be made with `git commit` only, using the message composed according to this standard, with no additional flags that affect authorship.

---

## Known Limitations (To Fix)

| # | Issue | Location | Priority |
|---|-------|----------|----------|
| 1 | Three parallel intent systems (`intent_recognizer.py`, `embeddings.py`, inline in `response_generator.py`) — not unified | All three files | Medium |
| 2 | `_quick_intent_recognition` in `response_generator.py` is still regex-only | `response_generator.py:236` | Medium |
| 3 | Topic detection in `embeddings.py` uses embedding similarity but is initialised lazily and silently skipped if embeddings fail | `embeddings.py:202` | Low |
| 4 | `print()` statements used for routing debug output instead of logger | `query_router.py:274` | Low |
