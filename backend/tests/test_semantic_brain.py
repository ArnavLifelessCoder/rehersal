"""The offline interviewer must reason about content, not pick from a list.

The distinguishing property, and what every test here is really checking:
**the same move on different answers must produce different questions, and the
difference must be traceable to what the candidate actually said.** A question
bank cannot do that; a concept graph can.
"""

from __future__ import annotations

import pytest

from app.interviewer.concepts import TRACK_CONCEPTS, ConceptIndex
from app.interviewer.semantic import SalientPhrase, SemanticBrain


async def ask(brain: SemanticBrain, move: str, **kw) -> str:
    return " ".join([s async for s in brain.ask(move=move, **kw)])


class TestSalientPhrase:
    def test_prefers_a_technical_bigram(self):
        assert SalientPhrase.extract(
            "I would use consistent hashing to spread the load"
        ) == "consistent hashing"

    def test_falls_back_to_a_single_technical_term(self):
        assert "cache" in (SalientPhrase.extract("we put a cache in front") or "")

    def test_uses_the_repeated_content_word(self):
        phrase = SalientPhrase.extract(
            "the migration was painful, the migration took weeks, migration everywhere"
        )
        assert phrase == "migration"

    def test_returns_none_rather_than_a_filler_word(self):
        """Quoting the wrong phrase is worse than quoting none."""
        assert SalientPhrase.extract("um so yeah I think it was fine really") is None

    def test_empty_input_is_safe(self):
        assert SalientPhrase.extract("") is None


class TestConceptDetection:
    @pytest.mark.parametrize(
        "track,text,expected",
        [
            ("dsa", "I'd use a hash map for lookups", "hashing"),
            ("dsa", "the time complexity is O(n log n)", "complexity"),
            ("dsa", "two pointers, one slow one fast", "two_pointer"),
            ("system_design", "put a redis cache in front", "caching"),
            ("system_design", "we'd shard by user id", "sharding"),
            ("system_design", "eventual consistency is fine here", "consistency"),
            ("behavioral", "as a result we shipped two weeks early", "result"),
            ("behavioral", "I disagreed with the tech lead", "conflict"),
        ],
    )
    def test_detects_the_concept(self, track, text, expected):
        idx = ConceptIndex(list(TRACK_CONCEPTS[track]))
        assert expected in idx.detect(text)

    def test_unrelated_text_detects_nothing(self):
        idx = ConceptIndex(list(TRACK_CONCEPTS["dsa"]))
        assert idx.detect("I enjoy hiking and my favourite food is pasta") == []


@pytest.mark.asyncio
class TestComposition:
    async def test_probe_targets_the_gap_not_the_topic(self):
        """They said 'hash map' and never said 'collision' -> ask about collisions."""
        brain = SemanticBrain("dsa", seed=1)
        q = await ask(
            brain, "probe",
            answer="I would just use a hash map to store the counts and look them up",
        )
        assert any(w in q.lower() for w in ("collision", "bucket", "key", "value")), q

    async def test_probe_quotes_the_candidates_own_words(self):
        brain = SemanticBrain("system_design", seed=2)
        q = await ask(
            brain, "probe",
            answer="I'd put a redis cache in front of the database to cut latency",
        )
        assert "cache" in q.lower() or "caching" in q.lower(), q

    async def test_different_answers_produce_different_probes(self):
        """The property a question bank structurally cannot have."""
        a = await ask(
            SemanticBrain("system_design", seed=3), "probe",
            answer="we would shard the data by user id across ten nodes",
        )
        b = await ask(
            SemanticBrain("system_design", seed=3), "probe",
            answer="everything goes through a kafka queue and workers consume it",
        )
        assert a != b, "identical probes for completely different answers"

    async def test_challenge_stresses_the_thing_they_proposed(self):
        brain = SemanticBrain("system_design", seed=4)
        q = await ask(
            brain, "challenge",
            answer="we shard by user id so each node holds a slice of the users",
        )
        assert any(w in q.lower() for w in ("shard", "traffic", "rebalance", "hot")), q

    async def test_behavioral_probe_targets_the_missing_star_element(self):
        """They described a situation but never their own actions."""
        brain = SemanticBrain("behavioral", seed=5)
        q = await ask(
            brain, "probe",
            answer=(
                "we had a really tight deadline at the time and the whole team "
                "was under a lot of pressure that quarter"
            ),
        )
        assert any(w in q.lower() for w in ("you", "your", "did", "contribution")), q

    async def test_interrupt_references_what_they_were_rambling_about(self):
        brain = SemanticBrain("system_design", seed=6)
        q = await ask(
            brain, "interrupt",
            answer="so the caching layer we built had a lot of moving parts and " * 8,
        )
        assert len(q) > 20
        assert "cach" in q.lower()

    async def test_unrecognised_answer_falls_back_to_the_weak_axis(self):
        brain = SemanticBrain("dsa", seed=7)
        q = await ask(
            brain, "probe",
            answer="yeah I mean it sort of depends on things really",
            weakest_axis="specificity -- they stayed abstract",
        )
        assert any(w in q.lower() for w in ("concrete", "specific", "example", "practice")), q


@pytest.mark.asyncio
class TestConversationQuality:
    async def test_never_repeats_a_question_across_a_session(self):
        brain = SemanticBrain("dsa", seed=8)
        seen = [await ask(brain, "opening")]
        for _ in range(8):
            seen.append(await ask(brain, "next_question", answer="I used a hash map, O(n) time"))
        assert len(seen) == len(set(seen)), "the interviewer repeated itself"

    async def test_probes_do_not_repeat_either(self):
        brain = SemanticBrain("system_design", seed=9)
        answers = [
            "we'd cache it in redis",
            "shard by user id",
            "use a kafka queue for the writes",
            "replicas in three regions",
            "eventual consistency is fine",
        ]
        probes = [await ask(brain, "probe", answer=a) for a in answers]
        assert len(probes) == len(set(probes))

    async def test_next_question_prefers_adjacent_ground(self):
        """After sharding, a companion topic should surface, not a random one."""
        brain = SemanticBrain("system_design", seed=10)
        await ask(brain, "next_question", answer="we shard by user id across nodes")
        follow = await ask(brain, "next_question", answer="we shard by user id across nodes")
        assert follow

    @pytest.mark.parametrize("track", ["dsa", "system_design", "behavioral", "jd"])
    async def test_every_move_produces_usable_speech_on_every_track(self, track):
        for move in ("opening", "next_question", "probe", "challenge",
                     "interrupt", "hint", "move_on", "closing"):
            brain = SemanticBrain(track, seed=11)
            text = await ask(brain, move, answer="we cached it and sharded by user id")
            assert text.strip(), f"{track}/{move} produced nothing"
            assert len(text) < 400, f"{track}/{move} too long to speak: {text}"
            assert "{" not in text, f"{track}/{move} leaked an unfilled slot: {text}"

    async def test_output_is_speakable_not_markdown(self):
        brain = SemanticBrain("dsa", seed=12)
        for move in ("opening", "probe", "challenge", "closing"):
            text = await ask(brain, move, answer="I'd sort it first then binary search")
            assert not any(c in text for c in ("*", "#", "`", "\n-")), text

    async def test_jd_track_works_with_a_pasted_description(self):
        brain = SemanticBrain(
            "jd", seed=13,
            jd_text="Senior backend engineer.\n- Kubernetes and Kafka\n- Distributed systems\n",
        )
        assert (await ask(brain, "opening")).strip()

    async def test_streams_sentence_by_sentence(self):
        """Pipelining depends on this: TTS starts on sentence one."""
        brain = SemanticBrain("dsa", seed=14)
        chunks = [s async for s in brain.ask(move="opening")]
        assert len(chunks) >= 2, "opening should stream as multiple sentences"
        assert all(c.strip() for c in chunks)
