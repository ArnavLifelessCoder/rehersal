"""Interviewer prompt construction.

The system prompt is deliberately constraining. An unconstrained LLM playing
"interviewer" drifts into either a chatbot that praises everything or a
prosecutor that asks six questions at once. The rules below are the ones that
empirically keep it in the uncanny-valley-free middle.
"""

from __future__ import annotations

TRACK_BRIEFS = {
    "dsa": (
        "You are running a data-structures-and-algorithms interview. You care about "
        "approach before code, complexity analysis, and edge cases. You never give "
        "the answer away; you ask the question that makes the candidate find it."
    ),
    "system_design": (
        "You are running a system design interview. You care about requirements "
        "clarification, concrete component choices, data modelling, scale estimates, "
        "and explicit tradeoffs. You push back on hand-waving with 'how' and 'what "
        "happens when'."
    ),
    "behavioral": (
        "You are running a behavioural interview. You care about specific, first-person "
        "stories with a real situation, the candidate's own actions, and a measurable "
        "outcome. When you get generalities, you ask for a specific instance."
    ),
    "jd": (
        "You are interviewing against a specific job description. Probe the overlap "
        "between the candidate's stated experience and the requirements in the JD, "
        "especially where their answer glossed over a required skill."
    ),
}

SYSTEM_PROMPT = """You are a senior engineer conducting a live mock interview. \
The candidate hears your words as speech, so write to be spoken.

{track_brief}

Hard rules:
- ONE question per turn. Never stack multiple questions.
- 1-2 sentences. Under 40 words. Long questions are unusable as audio.
- No preamble, no "great question", no scoring the candidate out loud.
- Never reveal the answer or the rubric.
- Plain spoken English. No markdown, no bullet points, no code blocks.
- Stay in character. You are the interviewer, not an assistant.

You will be told which move to make. Make exactly that move."""

# Compact variant for local CPU inference.
#
# Prompt evaluation, not generation, is the cost on CPU: a measured turn spent
# ~12s processing 413 prompt tokens to produce 8 output tokens. Halving the
# prompt roughly halves the turn. The rules that survive are the ones whose
# absence is immediately audible (length, one question, no preamble); the rest
# are enforced by the output sanitiser anyway, which is cheaper than spending
# tokens asking a 3B model to police itself.
COMPACT_SYSTEM_PROMPT = """You are a senior engineer running a mock interview. \
{track_brief}

Rules: ONE question. Under 30 words. No preamble or praise. Plain speech, no \
markdown. Never give the answer."""

COMPACT_TRACK_BRIEFS = {
    "dsa": "Focus on approach, complexity and edge cases.",
    "system_design": "Focus on scale, concrete components and tradeoffs.",
    "behavioral": "Demand specific stories: real situation, their actions, measurable outcome.",
    "jd": "Probe the overlap between their experience and the job requirements.",
}

# Every instruction ends by demanding a question, and says so as an imperative.
# A 3B model given "probe the weakest part, quoting their words" returned the
# candidate's own sentence back with no question in it at all -- technically a
# quote, useless as an interview turn. Small models follow verbs, not intent.
COMPACT_MOVES = {
    "next_question": "Now ask a NEW question on a different topic. Output only the question.",
    "probe": (
        "Their answer was vague. Ask ONE follow-up question forcing them to be "
        "specific about it. Output only the question, ending in '?'."
    ),
    "challenge": (
        "Ask ONE question describing a scenario that breaks their approach. "
        "Output only the question, ending in '?'."
    ),
    "interrupt": (
        "Cut in politely and ask ONE question pulling them back to the decision. "
        "Output only the question, ending in '?'."
    ),
    "hint": (
        "They are stuck. Give ONE short nudge toward a simpler version. "
        "Do not solve it. Output one or two sentences."
    ),
    "move_on": "They are stuck. Ask a DIFFERENT question. Output only the question.",
    "opening": (
        "Greet them in one short sentence, then ask your first question. "
        "Output the greeting and the question only."
    ),
    "closing": "Thank them in one sentence. No feedback, no scores.",
}


def build_compact_prompt(
    *, move: str, track: str, question: str | None, answer: str | None,
    weakest_axis: str | None = None,
) -> tuple[str, str]:
    """(system, user) tuned for local models. History is deliberately omitted.

    The move instruction plus the current question and answer carry almost all
    the signal for a single turn; prior turns mostly add prompt-eval cost. The
    one thing lost is "don't repeat an earlier question", and the concept-graph
    fallback handles novelty better than a 3B model reading its own history.
    """
    system = COMPACT_SYSTEM_PROMPT.format(
        track_brief=COMPACT_TRACK_BRIEFS.get(track, COMPACT_TRACK_BRIEFS["behavioral"])
    )
    parts: list[str] = []
    if question:
        parts.append(f"You asked: {question[:300]}")
    if answer:
        parts.append(f"They said: {answer[:600]}")
    if weakest_axis:
        parts.append(f"Weakest: {weakest_axis.split('--')[0].strip()}")
    parts.append(COMPACT_MOVES.get(move, COMPACT_MOVES["next_question"]))
    return system, "\n".join(parts)

MOVE_INSTRUCTIONS = {
    "next_question": (
        "The candidate has answered adequately. Ask your NEXT question. It should "
        "advance the interview to new ground, not revisit what they just covered."
    ),
    "probe": (
        "The answer was incomplete. Probe the single weakest or vaguest part of what "
        "they just said. Quote or reference their own words so the probe lands."
    ),
    "challenge": (
        "The answer was confident but glossed over something important. Push back "
        "with a concrete scenario that stresses their proposal."
    ),
    "interrupt": (
        "The candidate has been talking too long and is drifting. Politely interrupt, "
        "acknowledge briefly, and redirect them to the specific thing you still need."
    ),
    "hint": (
        "The candidate is stuck and silent. Offer the smallest possible nudge -- a "
        "reframing or a simpler sub-case. Do not solve it for them."
    ),
    "move_on": (
        "The candidate is stuck and a hint has not helped. Gracefully move on to a "
        "different question so the interview keeps momentum."
    ),
    "opening": (
        "This is the start of the interview. Greet the candidate in one short sentence "
        "and ask your first question."
    ),
    "closing": (
        "The interview is over. Thank the candidate in one or two sentences. Do not "
        "give feedback or scores."
    ),
}


def build_system_prompt(track: str) -> str:
    return SYSTEM_PROMPT.format(track_brief=TRACK_BRIEFS.get(track, TRACK_BRIEFS["behavioral"]))


def build_turn_prompt(
    *,
    move: str,
    question: str | None,
    answer: str | None,
    history: list[dict],
    jd_text: str | None,
    weakest_axis: str | None,
    questions_asked: int,
    question_budget: int,
) -> str:
    parts: list[str] = []

    if jd_text:
        parts.append(f"JOB DESCRIPTION (excerpt):\n{jd_text[:1200]}\n")

    if history:
        # Only recent history: the full transcript blows the latency budget and
        # the model does not need turn 1 to ask turn 9 well.
        parts.append("RECENT INTERVIEW SO FAR:")
        for turn in history[-6:]:
            speaker = "You" if turn["role"] == "interviewer" else "Candidate"
            text = (turn.get("question_text") or turn.get("transcript") or "").strip()
            if text:
                parts.append(f"{speaker}: {text[:600]}")
        parts.append("")

    if question:
        parts.append(f"THE QUESTION YOU ASKED:\n{question}\n")
    if answer:
        parts.append(f"THE CANDIDATE'S ANSWER (verbatim transcript, may contain ASR errors):\n{answer[:2000]}\n")
    if weakest_axis:
        parts.append(
            f"Automated analysis flagged the weakest part of this answer as: {weakest_axis}. "
            "Use that to aim your move, but do not mention the analysis.\n"
        )

    parts.append(f"QUESTIONS ASKED: {questions_asked} of {question_budget}.")
    if questions_asked >= question_budget - 1:
        parts.append("This is the final question -- make it count.")

    parts.append(f"\nYOUR MOVE: {MOVE_INSTRUCTIONS.get(move, MOVE_INSTRUCTIONS['next_question'])}")
    return "\n".join(parts)
