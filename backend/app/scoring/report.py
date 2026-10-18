"""Session report: the end screen.

Design constraint from the spec's §11 -- this thing judges people, so it has to
be honest about what it knows. Three rules enforced here:

  1. Every headline score ships with the confidence it was computed at, and
     low-confidence sessions say so in plain English at the top.
  2. Every "worst moment" cites the specific measured evidence, with a timestamp
     the user can jump to and verify. No unfalsifiable vibes.
  3. Strengths are surfaced alongside weaknesses. A report that only lists
     failures is not more rigorous, it's just worse feedback -- users
     over-correct on whatever was listed first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..session.manager import InterviewSession


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
    """Confidence-weighted average: a score we're unsure about pulls less."""
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return _mean([v for v, _ in pairs])
    return round(sum(v * w for v, w in pairs) / total_w, 3)


def _find_worst_moments(session: "InterviewSession", limit: int = 5) -> list[dict]:
    """Rank concrete, citable failures across all candidate turns."""
    moments: list[dict] = []

    for turn in session.turns:
        if turn.role != "candidate" or not turn.features:
            continue
        f = turn.features
        fillers = turn.fillers or {}
        base = {"turn_idx": turn.idx, "question": turn.question_text}

        fpm = f.get("filler_per_min", 0.0)
        if fpm >= 8:
            top = sorted((fillers.get("breakdown") or {}).items(), key=lambda kv: -kv[1])[:3]
            detail = ", ".join(f'"{w}" x{c}' for w, c in top)
            moments.append({
                **base,
                "severity": min(fpm / 25.0, 1.0),
                "reason": f"{fpm:.0f} fillers per minute ({detail})",
                "metric": "filler_per_min",
                "value": fpm,
                "ts_ms": _first_filler_ts(turn),
            })

        wpm = f.get("words_per_min", 140)
        if wpm > 180:
            moments.append({
                **base, "severity": min((wpm - 180) / 60.0, 1.0),
                "reason": f"raced at {wpm:.0f} words/min -- likely nerves",
                "metric": "words_per_min", "value": wpm, "ts_ms": 0,
            })
        elif wpm < 90 and f.get("word_count", 0) > 20:
            moments.append({
                **base, "severity": min((90 - wpm) / 40.0, 1.0),
                "reason": f"dragged at {wpm:.0f} words/min",
                "metric": "words_per_min", "value": wpm, "ts_ms": 0,
            })

        eye = f.get("eye_contact_pct")
        if eye is not None and eye < 0.4 and (turn.delivery or {}).get("confidence", 0) > 0.5:
            moments.append({
                **base, "severity": min((0.4 - eye) / 0.4, 1.0),
                "reason": f"eye contact dropped to {eye * 100:.0f}% on this answer",
                "metric": "eye_contact_pct", "value": eye, "ts_ms": 0,
            })

        freeze = f.get("longest_freeze_ms", 0.0)
        if freeze > 2500:
            moments.append({
                **base, "severity": min(freeze / 10000.0, 1.0),
                "reason": f"went completely still for {freeze / 1000:.1f}s",
                "metric": "longest_freeze_ms", "value": freeze, "ts_ms": int(freeze),
            })

        pause = f.get("longest_pause_ms", 0.0)
        if pause > 3000:
            moments.append({
                **base, "severity": min(pause / 8000.0, 1.0),
                "reason": f"{pause / 1000:.1f}s of dead air mid-answer",
                "metric": "longest_pause_ms", "value": pause, "ts_ms": int(pause),
            })

        content = turn.content or {}
        if content.get("relevance", 1.0) < 0.35 and content.get("confidence", 0) > 0.4:
            moments.append({
                **base, "severity": 1.0 - content["relevance"],
                "reason": "this answer drifted away from what was asked",
                "metric": "relevance", "value": content["relevance"], "ts_ms": 0,
            })

    moments.sort(key=lambda m: -m["severity"])
    return moments[:limit]


def _first_filler_ts(turn) -> int:
    """Approximate ms offset of the first filler, from token position."""
    spans = (turn.fillers or {}).get("spans") or []
    if not spans or not turn.transcript:
        return 0
    total_tokens = max(len(turn.transcript.split()), 1)
    frac = spans[0]["token_index"] / total_tokens
    return int(frac * turn.duration_s * 1000)


def _find_strengths(session: "InterviewSession", limit: int = 3) -> list[dict]:
    strengths: list[dict] = []
    for turn in session.turns:
        if turn.role != "candidate" or not turn.features:
            continue
        f, d, c = turn.features, turn.delivery or {}, turn.content or {}
        subs = d.get("subscores", {})
        if f.get("filler_per_min", 99) < 3 and f.get("word_count", 0) > 40:
            strengths.append({
                "turn_idx": turn.idx,
                "reason": f"clean, filler-free delivery ({f['filler_per_min']:.1f}/min)",
                "metric": "filler_per_min",
            })
        if c.get("structure", 0) > 0.75:
            elements = ", ".join((c.get("notes") or {}).get("structure_elements", []))
            strengths.append({
                "turn_idx": turn.idx,
                "reason": f"well-structured answer ({elements})",
                "metric": "structure",
            })
        if subs.get("confidence", 0) > 0.78:
            strengths.append({
                "turn_idx": turn.idx,
                "reason": "steady, well-projected voice",
                "metric": "confidence",
            })
    seen: set[str] = set()
    unique = []
    for s in strengths:
        if s["metric"] not in seen:
            seen.add(s["metric"])
            unique.append(s)
    return unique[:limit]


def _summary_text(
    overall_d: float, overall_c: float, worst: list[dict], conf: float,
    *, had_vision: bool, speak_s: float, answers: int,
) -> str:
    def band(x: float) -> str:
        return "strong" if x >= 0.75 else "solid" if x >= 0.6 else "mixed" if x >= 0.45 else "rough"

    lines = [
        f"Delivery came out {band(overall_d)} ({overall_d * 100:.0f}/100) and content "
        f"{band(overall_c)} ({overall_c * 100:.0f}/100)."
    ]
    if worst:
        lines.append(f"The single biggest thing to fix: {worst[0]['reason']}.")

    # Name the actual reason confidence is low. "short or ran without the
    # camera" makes the user guess which, and a caveat you can't act on is
    # just noise.
    if conf < 0.5:
        reasons = []
        if not had_vision:
            reasons.append("the camera was off, so nothing about your presence was measured")
        if speak_s < 60:
            reasons.append(f"you only spoke for {speak_s:.0f} seconds in total")
        if answers < 3:
            reasons.append(f"there {'was' if answers == 1 else 'were'} only {answers} answer"
                           f"{'' if answers == 1 else 's'} to average over")
        if reasons:
            lines.append(
                "Treat these numbers as directional: " + "; ".join(reasons) + "."
            )
        else:
            lines.append("Treat these numbers as directional -- there wasn't much signal.")
    return " ".join(lines)


def build_report(session: "InterviewSession") -> dict:
    candidate_turns = [t for t in session.turns if t.role == "candidate"]

    delivery_pairs = [
        (t.delivery["overall"], t.delivery.get("confidence", 0.5))
        for t in candidate_turns if t.delivery
    ]
    content_pairs = [
        (t.content["overall"], t.content.get("confidence", 0.5))
        for t in candidate_turns if t.content
    ]
    overall_delivery = _weighted_mean(delivery_pairs)
    overall_content = _weighted_mean(content_pairs)

    subscore_names = ["confidence", "clarity", "pace", "presence"]
    delivery_breakdown = {
        name: _mean([t.delivery["subscores"][name] for t in candidate_turns if t.delivery])
        for name in subscore_names
    }

    total_fillers = sum((t.fillers or {}).get("filler_count", 0) for t in candidate_turns)
    speak_s = sum(t.duration_s for t in candidate_turns)
    all_breakdown: dict[str, int] = {}
    for t in candidate_turns:
        for word, n in ((t.fillers or {}).get("breakdown") or {}).items():
            all_breakdown[word] = all_breakdown.get(word, 0) + n

    mean_conf = _mean([c for _, c in delivery_pairs + content_pairs])
    worst = _find_worst_moments(session)
    had_vision = session._session_cv.has_data

    return {
        "session_id": session.id,
        "track": session.track,
        "duration_s": round((session.ended_at or 0) - session.started_at, 1),
        "questions_asked": session.state.questions_asked,
        "answers_given": len(candidate_turns),
        "overall_delivery": overall_delivery,
        "overall_content": overall_content,
        "confidence": round(mean_conf, 3),
        "delivery_breakdown": delivery_breakdown,
        "content_breakdown": {
            axis: _mean([t.content[axis] for t in candidate_turns if t.content])
            for axis in ("relevance", "structure", "completeness", "correctness_signal")
        },
        "speaking_time_s": round(speak_s, 1),
        "filler_total": total_fillers,
        "filler_per_min": round(total_fillers / max(speak_s / 60.0, 1e-6), 2),
        "filler_breakdown": dict(sorted(all_breakdown.items(), key=lambda kv: -kv[1])[:10]),
        "worst_moments": worst,
        "strengths": _find_strengths(session),
        "delivery_timeline": [
            {
                "turn_idx": t.idx,
                "delivery": t.delivery["overall"] if t.delivery else None,
                "content": t.content["overall"] if t.content else None,
                "words_per_min": (t.features or {}).get("words_per_min"),
                "filler_per_min": (t.features or {}).get("filler_per_min"),
                "eye_contact_pct": (t.features or {}).get("eye_contact_pct"),
            }
            for t in candidate_turns
        ],
        "cv_timeline": session._session_cv.timeline(),
        "latency": session.latency.summary(),
        "backends": {
            "asr": session._asr.backend,
            "brain": session.brain.name,
            "delivery_scorer": session.deps.delivery_scorer.name,
            "content_scorer": getattr(session.deps.content_scorer._embedder, "name", "lexical"),
        },
        "had_vision": had_vision,
        # The trace tail rides along in the report. When someone says "it cut
        # me off mid-answer yesterday", this is what makes that diagnosable
        # today instead of unreproducible.
        "debug_events": session.trace.tail_for_report(),
        "event_counts": session.trace.counts(),
        "summary_text": _summary_text(
            overall_delivery, overall_content, worst, mean_conf,
            had_vision=had_vision, speak_s=speak_s, answers=len(candidate_turns),
        ),
        "transcript": [t.to_dict() for t in session.turns],
    }
