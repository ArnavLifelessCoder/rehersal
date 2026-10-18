"""The offline interviewer: composes questions instead of selecting them.

Given an answer, it:
  1. detects which concepts the candidate actually raised,
  2. lifts a salient phrase from their own wording,
  3. finds the gap — a companion concept a good answer would have paired with
     what they said, but which they never mentioned,
  4. composes a question aimed at that gap, quoting them.

That's what makes it feel responsive rather than canned. "You reached for a
hash map. What happens when two different inputs land in the same bucket?" is
not a question anyone wrote down in advance for this session — it exists
because the candidate said "hash map" and never said "collision".

It is still weaker than a good LLM: it can't understand a genuinely novel
answer, and its phrasing space is bounded. When an API key is present the
hosted brain takes over. This is the floor, not the ceiling — but it's a floor
that reasons about content.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncIterator

from .concepts import TRACK_CONCEPTS, ConceptIndex

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Words that are never the interesting part of what someone said.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in",
    "on", "for", "with", "as", "is", "are", "was", "were", "be", "been", "it",
    "this", "that", "these", "those", "i", "you", "we", "they", "my", "your",
    "our", "would", "could", "should", "can", "will", "do", "does", "did",
    "have", "has", "had", "at", "by", "from", "not", "no", "what", "how", "why",
    "when", "where", "which", "who", "about", "there", "here", "just", "like",
    "really", "actually", "basically", "um", "uh", "kind", "sort", "thing",
    "things", "stuff", "going", "want", "need", "get", "got", "make", "made",
    "use", "used", "using", "one", "two", "also", "very", "much", "more", "some",
    "think", "know", "say", "said", "mean", "maybe", "probably", "guess", "well",
}

# Phrases that make a probe sound like it was listening.
_ACKS = (
    "Okay.", "Right.", "Got it.", "Mm-hm.", "Sure.",
)

_PROBE_FRAMES = (
    "{ack} {question}",
    "{question}",
    "Let me push on that. {question}",
    "Hold on there. {question}",
)


class SalientPhrase:
    """Pulls the phrase worth quoting back out of an answer.

    Prefers multi-word technical phrases over single words: quoting "consistent
    hashing" lands, quoting "hashing" is generic, and quoting "thing" is worse
    than quoting nothing. When nothing clears the bar we fall back to templates
    with no {term} slot rather than inserting a filler word, because a probe
    that quotes the wrong phrase is more damaging than one that quotes none.
    """

    _TECH_HINT = re.compile(
        r"\b(cache|caching|shard|shardin|queue|index|replica|hash|pointer|graph|"
        r"recursion|complexity|latency|throughput|schema|partition|token|thread|"
        r"lock|retry|timeout|migration|pipeline|cluster|endpoint|payload|buffer)\w*",
        re.I,
    )

    # Compound terms where the technical word comes FIRST. The modifier-noun
    # rule below would clip these to their head word, and "you reached for
    # hash" instead of "you reached for a hash map" sounds like a transcription
    # error rather than a question.
    _COMPOUNDS = (
        "hash map", "hash table", "linked list", "binary search", "binary tree",
        "load balancer", "message queue", "connection pool", "cache layer",
        "consistent hashing", "priority queue", "sliding window", "two pointer",
        "circuit breaker", "rate limit", "write ahead log", "bloom filter",
        "read replica", "dead letter queue", "service mesh", "event loop",
    )

    @classmethod
    def extract(cls, text: str) -> str | None:
        if not text:
            return None
        lowered = text.lower()

        # 0. A known compound beats everything -- it's the most specific thing
        #    the candidate said, and quoting it exactly is what lands.
        for compound in cls._COMPOUNDS:
            if compound in lowered:
                return compound

        # 1. A technical bigram is the best thing to quote -- but only a real
        #    one. Two constraints, both learned from bad output:
        #      * modifier-noun order ("consistent hashing", "redis cache"): the
        #        technical word must be SECOND, or we produce verb phrases like
        #        "migration took" that read as though we misheard.
        #      * within a single clause: bigrams formed across a comma give
        #        "weeks migration", which is not a phrase anybody said.
        for clause in re.split(r"[,.;:!?]|\band\b|\bbut\b", lowered):
            words = re.findall(r"[a-z][a-z\-']+", clause)
            for i in range(len(words) - 1):
                a, b = words[i], words[i + 1]
                if a in _STOP or b in _STOP:
                    continue
                if cls._TECH_HINT.match(b):
                    return f"{a} {b}"

        # 2. A single technical term.
        if (m := cls._TECH_HINT.search(lowered)) is not None:
            return m.group(0)

        # 3. The most repeated content word -- what they kept coming back to.
        #    Counted over the WHOLE answer: reusing the clause-loop variable
        #    here would silently score only the final clause.
        counts: dict[str, int] = {}
        for w in re.findall(r"[a-z][a-z\-']+", lowered):
            if w in _STOP or len(w) < 5:
                continue
            counts[w] = counts.get(w, 0) + 1
        if counts:
            best, n = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))
            if n >= 2 or len(best) >= 7:
                return best
        return None


class SemanticBrain:
    """Composes interviewer turns from a concept graph. No network, no model."""

    name = "semantic"

    def __init__(self, track: str = "behavioral", seed: int | None = None,
                 jd_text: str | None = None) -> None:
        self.track = track
        self._rng = random.Random(seed)
        self._index = ConceptIndex(list(TRACK_CONCEPTS.get(track, TRACK_CONCEPTS["behavioral"])))
        self._covered: set[str] = set()
        self._asked_openers: set[str] = set()
        self._used_templates: set[str] = set()
        self._jd_terms = self._parse_jd(jd_text) if jd_text else []
        self._current_concept: str | None = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _parse_jd(jd_text: str) -> list[str]:
        """Requirement-ish phrases from a job description, for the `jd` track."""
        terms: list[str] = []
        for line in jd_text.splitlines():
            line = line.strip(" -•*\t")
            if 12 < len(line) < 160 and not line.endswith(":"):
                terms.append(line)
        return terms[:20]

    def _fresh(self, options: tuple[str, ...] | list[str]) -> str | None:
        """Pick an unused template, so the interviewer never repeats itself."""
        unused = [o for o in options if o not in self._used_templates]
        if not unused:
            return None
        choice = self._rng.choice(unused)
        self._used_templates.add(choice)
        return choice

    def _fill(self, template: str, term: str | None, concept_label: str | None) -> str | None:
        if "{term}" in template:
            if not term:
                return None
            template = template.replace("{term}", term)
        if "{concept}" in template:
            if not concept_label:
                return None
            template = template.replace("{concept}", concept_label)
        return template

    def _compose(
        self, candidates: list[tuple[str, str | None, str | None]]
    ) -> str | None:
        """Try (template, term, label) triples until one fills cleanly."""
        for template, term, label in candidates:
            if template in self._used_templates:
                continue
            filled = self._fill(template, term, label)
            if filled:
                self._used_templates.add(template)
                return filled
        return None

    # ------------------------------------------------------------------ moves

    def _opening(self) -> str:
        starters = [c for c in self._index.openers() if c.starter]
        concept = self._rng.choice(starters or self._index.openers())
        self._asked_openers.add(concept.key)
        self._current_concept = concept.key
        return f"Thanks for making the time. {concept.opener}"

    def _next_question(self, answer: str | None) -> str:
        # Prefer a concept adjacent to what they've already shown they know --
        # an interview that jumps randomly between topics feels like a quiz,
        # not a conversation.
        adjacent: list[str] = []
        for key in self._covered:
            concept = self._index.by_key.get(key)
            if concept:
                adjacent.extend(
                    k for k in concept.companions
                    if k not in self._covered and k not in self._asked_openers
                )

        pool = [
            self._index.by_key[k] for k in adjacent
            if k in self._index.by_key and self._index.by_key[k].opener
        ]
        if not pool:
            pool = [c for c in self._index.openers() if c.key not in self._asked_openers]

        if pool:
            concept = self._rng.choice(pool)
            self._asked_openers.add(concept.key)
            self._current_concept = concept.key
            return concept.opener

        # Every opener is spent. Deepen or stress something already covered
        # rather than asking a question the candidate has already answered --
        # a repeat is the most obvious possible tell that nobody is listening.
        return self._exhausted_question()

    def _exhausted_question(self) -> str:
        covered = [self._index.by_key[k] for k in self._covered if k in self._index.by_key]
        self._rng.shuffle(covered)
        for concept in covered:
            if (q := self._compose(
                [(t, None, concept.label) for t in concept.stress + concept.deepen]
            )):
                self._current_concept = concept.key
                return q
        closers = (
            "Let's zoom out. Of everything we've covered, which part would you most want to revisit?",
            "Last one: what's the biggest risk in what you've described to me today?",
            "If you had another week on this, what would you change first?",
        )
        return self._fresh(closers) or "What would you change first, given more time?"

    def _probe(self, answer: str | None, weakest_axis: str | None) -> str:
        term = SalientPhrase.extract(answer or "")
        mentioned = self._index.detect(answer or "")

        # 1. The best probe: they raised X, a good answer pairs X with Y, and
        #    they never said Y. Ask about Y, in the words of X.
        for key in mentioned:
            concept = self._index.by_key.get(key)
            if not concept:
                continue
            for companion_key in concept.companions:
                if companion_key in mentioned or companion_key in self._covered:
                    continue
                companion = self._index.by_key.get(companion_key)
                if not companion or not companion.gap_probe:
                    continue
                if (q := self._compose(
                    [(t, term, companion.label) for t in companion.gap_probe]
                )):
                    self._current_concept = companion_key
                    return q

        # 2. Otherwise go deeper on what they did raise.
        for key in mentioned:
            concept = self._index.by_key.get(key)
            if not concept or not concept.deepen:
                continue
            if (q := self._compose([(t, term, concept.label) for t in concept.deepen])):
                self._current_concept = key
                return q

        # 3. They said nothing the graph recognises. Aim at the weak axis.
        return self._axis_probe(weakest_axis, term)

    def _axis_probe(self, weakest_axis: str | None, term: str | None) -> str:
        axis = (weakest_axis or "").split("--")[0].strip()
        banks = {
            "relevance": (
                "I want to bring you back to what I asked. Can you answer that part directly?",
                "That's adjacent to the question. Take another run at what I actually asked.",
            ),
            "completeness": (
                "That's a start. What's the part you haven't covered yet?",
                "You've got the outline. Fill in the piece you skipped over.",
            ),
            "structure": (
                "Let me slow you down. Give me that again in order -- start to finish.",
                "Take it one step at a time. What happens first?",
            ),
            "specificity": (
                "That's still fairly abstract. Give me a concrete example.",
                "Make that specific for me. What would that actually look like in practice?",
            ),
        }
        options = banks.get(axis) or (
            "Say more about that.",
            "Unpack that a little for me.",
            "Can you go one level deeper there?",
        )
        if term:
            options = options + (f"Tell me more about the {term} part specifically.",)
        if (picked := self._fresh(options)) is not None:
            return picked
        return f"Say more about the {term} part." if term else "Can you go a level deeper there?"

    def _challenge(self, answer: str | None) -> str:
        term = SalientPhrase.extract(answer or "")
        for key in self._index.detect(answer or ""):
            concept = self._index.by_key.get(key)
            if not concept or not concept.stress:
                continue
            if (q := self._compose([(t, term, concept.label) for t in concept.stress])):
                return q
        generic = (
            "Let's stress that. What's the first thing that breaks under ten times the load?",
            "A reviewer tells you that's over-engineered for the actual problem. How do you respond?",
            "What would you give up if you had half the time to build it?",
            "Where's the weakest assumption in what you just described?",
        )
        return self._fresh(generic) or "What's the weakest part of that approach?"

    def _interrupt(self, answer: str | None) -> str:
        term = SalientPhrase.extract(answer or "")
        if term:
            options = (
                f"Let me jump in -- I want to make sure we get to the decision. On {term}, what did you actually choose?",
                f"I'll pause you there, that's useful context. Land it for me: what did you do about {term}?",
            )
        else:
            options = (
                "Let me jump in there -- can you land on the specific decision you made?",
                "I'll pause you, that's helpful background. What's the short version of what you did?",
            )
        return self._fresh(options) or "Let me pause you there -- what's the decision you landed on?"

    def _hint(self) -> str:
        concept = self._index.by_key.get(self._current_concept or "")
        if concept and concept.key in {"complexity", "hashing", "recursion", "dynamic_programming"}:
            options = (
                "Take your time. Try the smallest version first -- what if the input had three elements?",
                "Let's simplify: forget performance entirely. What's the most obvious thing that works at all?",
            )
        elif self.track == "behavioral":
            options = (
                "No rush. Think of any specific instance, even a small one, and start from there.",
                "It doesn't have to be dramatic. What's the most recent time something like that came up?",
            )
        else:
            options = (
                "Take a second. What would you build if you only had one server and no constraints?",
                "Start from the user. What does someone actually do with this system?",
            )
        return self._fresh(options) or "Take your time -- start with the simplest version you can think of."

    def _move_on(self, answer: str | None) -> str:
        prefix = self._rng.choice((
            "No problem, let's park that.",
            "That's fine, it's a tricky one.",
            "Let's come back to that if there's time.",
        ))
        return f"{prefix} {self._next_question(answer)}"

    def _closing(self) -> str:
        return (
            "That's all the time we have. Thanks for talking it through with me -- "
            "your report is generating now."
        )

    # ------------------------------------------------------------------ entry

    async def ask(
        self,
        *,
        move: str,
        track: str = "",
        question: str | None = None,
        answer: str | None = None,
        history: list[dict] | None = None,
        jd_text: str | None = None,
        weakest_axis: str | None = None,
        questions_asked: int = 0,
        question_budget: int = 12,
    ) -> AsyncIterator[str]:
        if answer:
            self._covered.update(self._index.detect(answer))

        if move == "opening":
            text = self._opening()
        elif move == "probe":
            text = self._probe(answer, weakest_axis)
        elif move == "challenge":
            text = self._challenge(answer)
        elif move == "interrupt":
            text = self._interrupt(answer)
        elif move == "hint":
            text = self._hint()
        elif move == "move_on":
            text = self._move_on(answer)
        elif move == "closing":
            text = self._closing()
        else:
            text = self._next_question(answer)

        # Occasional acknowledgement, so it doesn't read as a machine firing
        # questions. Only on probes -- an "okay" before a brand-new topic sounds
        # like it's responding to something it isn't.
        if move in ("probe", "challenge") and self._rng.random() < 0.35:
            text = f"{self._rng.choice(_ACKS)} {text}"

        for sentence in _SENTENCE_END.split(text):
            if sentence.strip():
                await asyncio.sleep(0.01)  # honour the streaming contract
                yield sentence.strip()
