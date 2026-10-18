"""The turn-taking state machine, as pure logic.

Deliberately free of I/O, asyncio, and model calls: it takes measured signals
and returns a decision. That makes the interview's *behaviour* fully unit
testable without audio, a network, or a model -- which is the only way the
700ms endpointing and the probe/advance policy stay correct as everything
around them changes.

    LISTENING --(endpoint)--> THINKING --(decide)--> SPEAKING --(done)--> LISTENING
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"      # interviewer is talking
    LISTENING = "listening"    # candidate's floor
    THINKING = "thinking"      # deciding the next move
    ENDED = "ended"


class Move(str, Enum):
    OPENING = "opening"
    NEXT_QUESTION = "next_question"
    PROBE = "probe"
    CHALLENGE = "challenge"
    INTERRUPT = "interrupt"
    HINT = "hint"
    MOVE_ON = "move_on"
    CLOSING = "closing"


@dataclass
class AnswerSignals:
    """Everything the controller is allowed to base a decision on."""

    content_overall: float
    relevance: float
    completeness: float
    structure: float
    correctness_signal: float
    word_count: int
    duration_s: float
    was_silent: bool = False
    was_rambling: bool = False
    confidence: float = 1.0

    def weakest_axis(self) -> str | None:
        axes = {
            "relevance -- they answered a different question": self.relevance,
            "completeness -- they left out key parts": self.completeness,
            "structure -- the answer was hard to follow": self.structure,
            "specificity -- they stayed abstract": self.correctness_signal,
        }
        name, value = min(axes.items(), key=lambda kv: kv[1])
        return name if value < 0.6 else None


@dataclass
class TurnPolicy:
    """Thresholds governing probe-vs-advance. Tuned to be *slightly* generous:
    an interviewer that probes every answer is exhausting and stops feeling
    like a signal. Two consecutive probes on one question is the hard cap."""

    answered_threshold: float = 0.62
    probe_threshold: float = 0.30
    challenge_confidence: float = 0.72   # confident + shallow => challenge
    challenge_completeness: float = 0.55
    max_probes_per_question: int = 2
    min_words_for_answer: int = 12


@dataclass
class SessionState:
    track: str
    question_budget: int = 12
    policy: TurnPolicy = field(default_factory=TurnPolicy)

    state: State = State.IDLE
    questions_asked: int = 0
    probes_on_current: int = 0
    hints_on_current: int = 0
    current_question: str | None = None
    turn_index: int = 0

    def decide(self, signals: AnswerSignals) -> Move:
        """Pick the next interviewer move from measured answer signals."""
        p = self.policy

        # Out of budget -- wrap up regardless of how the answer went.
        if self.questions_asked >= self.question_budget:
            return Move.CLOSING

        # Silence handling comes first: hint once, then move on. Repeatedly
        # hinting at someone who is stuck is the worst interviewer behaviour.
        if signals.was_silent or signals.word_count < 3:
            if self.hints_on_current == 0:
                return Move.HINT
            return Move.MOVE_ON

        if signals.was_rambling:
            return Move.INTERRUPT

        # Too little substance to evaluate -- probe rather than score it.
        if signals.word_count < p.min_words_for_answer:
            if self.probes_on_current < p.max_probes_per_question:
                return Move.PROBE
            return Move.NEXT_QUESTION

        # Confident, fluent, but thin. This is the case a naive threshold gets
        # wrong: high relevance + low completeness looks "good enough" on an
        # average, so we branch on it explicitly before the average is taken.
        if (
            signals.correctness_signal >= p.challenge_confidence
            and signals.completeness < p.challenge_completeness
            and self.probes_on_current < p.max_probes_per_question
        ):
            return Move.CHALLENGE

        if signals.content_overall >= p.answered_threshold:
            return Move.NEXT_QUESTION

        if (
            signals.content_overall >= p.probe_threshold
            and self.probes_on_current < p.max_probes_per_question
        ):
            return Move.PROBE

        # Weak answer, already probed twice -- diminishing returns, move on.
        return Move.NEXT_QUESTION

    def apply(self, move: Move) -> None:
        """Commit the bookkeeping for a chosen move."""
        if move in (Move.NEXT_QUESTION, Move.MOVE_ON, Move.OPENING):
            self.questions_asked += 1
            self.probes_on_current = 0
            self.hints_on_current = 0
        elif move in (Move.PROBE, Move.CHALLENGE):
            self.probes_on_current += 1
        elif move == Move.HINT:
            self.hints_on_current += 1
        self.state = State.SPEAKING if move != Move.CLOSING else State.ENDED

    def is_new_question(self, move: Move) -> bool:
        return move in (Move.NEXT_QUESTION, Move.MOVE_ON, Move.OPENING)
