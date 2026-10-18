"""Concept graphs -- the knowledge the offline interviewer reasons over.

This is what replaces a question list. Each concept knows what a good answer
pairs it with (`companions`), how to go deeper on it (`deepen`), and how to
stress it (`stress`). Questions are then *composed* from the gap between what
the candidate said and what the graph expects, quoting their own words back.

The difference is observable: a list can only ever ask what it was written to
ask, in an order chosen before the candidate spoke. A graph asks "you brought
up sharding but never mentioned the hot-key problem" -- which requires knowing
both what they said and what it implies.

Slots available to templates:
    {term}     a salient phrase lifted from the candidate's own answer
    {concept}  the human-readable name of a concept they raised
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    aliases: tuple[str, ...]
    companions: tuple[str, ...] = ()
    opener: str = ""
    deepen: tuple[str, ...] = ()
    stress: tuple[str, ...] = ()
    # Asked when the candidate raised a companion but never this concept.
    gap_probe: tuple[str, ...] = ()
    # Suitable as the FIRST question of an interview. Most openers assume some
    # shared context ("two users edit the same document...") and land badly as
    # a greeting; starters are the ones that stand alone cold.
    starter: bool = False


def _c(key, label, aliases, **kw) -> Concept:
    return Concept(key=key, label=label, aliases=aliases, **kw)


# ---------------------------------------------------------------------- DSA

DSA = [
    _c("complexity", "time and space complexity",
       ("complexity", "big o", "o(n", "o(1", "o(log", "linear time", "quadratic",
        "time complexity", "space complexity", "runtime"),
       companions=("tradeoff", "edge_cases"),
       opener="Let's start here: given an unsorted array of integers, how would you find the two that sum to a target?", starter=True,
       deepen=(
           "You said {term}. Walk me through how you arrived at that complexity -- what's the dominant operation?",
           "Where exactly does the {term} cost come from? Point me at the line that dominates.",
       ),
       stress=(
           "Suppose the input no longer fits in memory. Does your complexity argument still hold?",
           "If I told you the constant factor made that slower than a naive loop for n under ten thousand, what would you change?",
       ),
       gap_probe=(
           "You've described the approach but not its cost. What's the time and space complexity of what you just outlined?",
       )),

    _c("hashing", "hashing and hash maps",
       ("hash", "hashmap", "hash map", "dictionary", "dict", "set", "lookup table"),
       companions=("collision", "space_tradeoff"),
       opener="How would you find the first non-repeating character in a long string?", starter=True,
       deepen=(
           "You reached for {term}. What's the key, and what's the value?",
           "What happens to {term} when two different inputs land in the same bucket?",
       ),
       stress=(
           "Your hash map is now bigger than RAM. What's your next move?",
           "What if the keys are adversarially chosen to collide?",
       ),
       gap_probe=(
           "You're using a hash structure -- what's your collision behaviour, and when does it start to matter?",
       )),

    _c("two_pointer", "two-pointer and sliding window",
       ("two pointer", "two pointers", "sliding window", "left and right",
        "fast and slow", "slow pointer", "fast pointer"),
       companions=("invariant", "edge_cases"),
       opener="Given a sorted array, how would you find a pair summing to a target using constant extra space?",
       deepen=(
           "With {term}, what's the invariant that guarantees you never miss a valid pair?",
           "How do you decide which pointer to move at each step?",
       ),
       stress=("Does {term} still work if the array isn't sorted? What changes?",)),

    _c("recursion", "recursion and divide-and-conquer",
       ("recursion", "recursive", "recurse", "divide and conquer", "backtrack",
        "backtracking", "subproblem"),
       companions=("base_case", "complexity"),
       opener="How would you generate all valid combinations of n pairs of parentheses?",
       deepen=(
           "In your recursive formulation, what's the base case, and why is it sufficient?",
           "How deep does {term} go in the worst case, and does that worry you?",
       ),
       stress=("What's your stack depth at n equals a million? How would you avoid blowing it?",)),

    _c("dynamic_programming", "dynamic programming",
       ("dynamic programming", "dp", "memo", "memoize", "memoization",
        "tabulation", "overlapping subproblem", "optimal substructure"),
       companions=("complexity", "state_definition"),
       opener="How would you find the length of the longest increasing subsequence in an array?",
       deepen=(
           "Define your DP state precisely. What does each entry actually mean?",
           "What's the recurrence relation behind {term}?",
       ),
       stress=("Can you get the space down from two dimensions to one? What do you lose?",)),

    _c("sorting", "sorting",
       ("sort", "sorted", "sorting", "quicksort", "mergesort", "heap sort",
        "order the", "ordering"),
       companions=("complexity", "stability"),
       opener="You need to sort a billion 32-bit integers on a machine with 1GB of RAM. How?",
       deepen=(
           "You're sorting first -- what does sorting buy you that you couldn't get otherwise?",
           "Is the sort you'd use stable, and does that matter here?",
       ),
       stress=("Sorting is n log n. Is there a linear-time approach for this input?",)),

    _c("graph", "graphs and traversal",
       ("graph", "bfs", "dfs", "breadth first", "depth first", "adjacency",
        "node", "vertex", "edge", "topological", "shortest path", "dijkstra"),
       companions=("cycle_detection", "complexity"),
       opener="Given a list of course prerequisites, how would you determine a valid order to take them?",
       deepen=(
           "You'd use {term} -- why that traversal rather than the other one?",
           "How do you represent the graph, and what does that cost you?",
       ),
       stress=("What if the graph has a cycle? How would you even detect it?",)),

    _c("edge_cases", "edge cases",
       ("edge case", "corner case", "empty", "null", "negative", "duplicate",
        "overflow", "boundary", "off by one"),
       companions=(),
       deepen=("Which edge case worries you most in what you've described?",),
       gap_probe=(
           "Let's stress it: what does your approach do on an empty input, and on one with all duplicates?",
       )),

    _c("collision", "hash collisions", ("collision", "chaining", "open addressing", "probing"),
       deepen=("How does your collision strategy affect worst-case lookup?",)),
    _c("invariant", "loop invariants", ("invariant", "always true", "maintains"),
       deepen=("State the invariant precisely -- what's true at the top of every iteration?",)),
    _c("base_case", "base cases", ("base case", "terminates", "stopping condition"),
       deepen=("Is your base case reachable from every path?",)),
    _c("tradeoff", "tradeoffs", ("tradeoff", "trade off", "trade-off", "in exchange", "at the cost of"),
       deepen=("What did you give up to get {term}?",)),
    _c("space_tradeoff", "space-time tradeoff", ("extra space", "space for time", "in place", "auxiliary")),
    _c("state_definition", "DP state", ("state", "subproblem", "dp[", "table")),
    _c("stability", "sort stability", ("stable", "stability", "preserve order")),
    _c("cycle_detection", "cycle detection", ("cycle", "loop in the graph", "visited set")),
]

# -------------------------------------------------------------- SYSTEM DESIGN

SYSTEM_DESIGN = [
    _c("requirements", "requirements and scale",
       ("requirement", "how many users", "qps", "traffic", "scale", "throughput",
        "daily active", "reads per", "writes per", "latency budget"),
       companions=("capacity_estimate",),
       opener="Let's design a URL shortener that serves a hundred million redirects a day. Where do you start?", starter=True,
       deepen=(
           "You mentioned {term}. Put a number on it -- what read and write volume are we actually designing for?",
           "Which of those requirements is the one that actually constrains the design?",
       ),
       gap_probe=(
           "Before we go further -- what are the actual numbers here? Reads per second, writes per second, data size?",
       )),

    _c("caching", "caching",
       ("cache", "caching", "redis", "memcached", "cdn", "in memory", "ttl", "hot data"),
       companions=("invalidation", "consistency"),
       opener="How would you make a read-heavy feed load in under a hundred milliseconds?",
       deepen=(
           "You'd put {term} in front. What's your eviction policy, and what's the hit rate you're betting on?",
           "What's the TTL, and what breaks if it's wrong in either direction?",
       ),
       stress=(
           "Your cache goes cold after a deploy and every request hits the database at once. What happens?",
           "One key becomes ten percent of all traffic. Does {term} still hold up?",
       ),
       gap_probe=(
           "You're caching -- how does a cached entry ever become wrong, and how do you find out?",
       )),

    _c("database", "data storage",
       ("database", "db", "postgres", "mysql", "sql", "nosql", "dynamo", "cassandra",
        "mongo", "store it in", "persist"),
       companions=("schema", "indexing", "replication"),
       opener="What's the data model for a system that has to serve both a user's timeline and a global search?",
       deepen=(
           "You picked {term}. What specifically about this workload made that the right call?",
           "Sketch the schema. What are the primary access patterns it has to serve?",
       ),
       stress=("Your primary goes down mid-write. What does the client see, and what have you lost?",)),

    _c("sharding", "partitioning and sharding",
       ("shard", "sharding", "partition", "split the data", "consistent hashing",
        "horizontal scal"),
       companions=("hot_key", "rebalancing"),
       opener="Your database is at capacity on a single box. Walk me through splitting it.",
       deepen=(
           "What's your partition key for {term}, and why that one?",
           "Which queries become expensive once you've partitioned this way?",
       ),
       stress=(
           "One shard is taking sixty percent of the traffic. How did that happen and what do you do?",
           "You need to add three more shards. How do you rebalance without downtime?",
       ),
       gap_probe=(
           "You're partitioning -- what's the key, and what happens when one partition gets disproportionately hot?",
       )),

    _c("replication", "replication and availability",
       ("replica", "replication", "failover", "leader", "follower", "primary",
        "secondary", "standby", "multi region"),
       companions=("consistency", "failure"),
       opener="How do you keep serving reads when an entire availability zone disappears?",
       deepen=(
           "With {term}, is replication synchronous or asynchronous, and what did that choice cost you?",
           "How does a client discover the new leader after a failover?",
       ),
       stress=("A network partition splits your replicas in half. Which side accepts writes?",)),

    _c("consistency", "consistency model",
       ("consistency", "consistent", "eventual", "strong consistency", "acid",
        "linearizable", "stale read", "cap theorem"),
       companions=("replication", "tradeoff"),
       opener="Two users edit the same document at the same moment. What does each of them see?",
       deepen=(
           "You said {term}. What's the worst-case staleness a user could actually observe?",
           "Which operations in this system genuinely need strong consistency, and which don't?",
       ),
       stress=("A user writes, then immediately reads, and doesn't see their own write. Acceptable?",)),

    _c("queue", "queues and async processing",
       ("queue", "kafka", "sqs", "rabbit", "async", "background job", "worker",
        "event", "pub sub", "publish"),
       companions=("idempotency", "backpressure"),
       opener="How would you handle a spike ten times larger than normal traffic without dropping requests?",
       deepen=(
           "With {term}, what's your delivery guarantee -- at least once, at most once, exactly once?",
           "What happens to a message that fails processing five times?",
       ),
       stress=("Your consumers fall behind and the queue grows for six hours. Then what?",),
       gap_probe=(
           "You've made that path asynchronous -- how do you handle a duplicate delivery?",
       )),

    _c("api", "API design",
       ("api", "endpoint", "rest", "grpc", "graphql", "request", "response",
        "http", "interface"),
       companions=("pagination", "versioning"),
       opener="Design the API surface a mobile client would use for this. What are the endpoints?",
       deepen=("What does the response to {term} actually look like? Give me the fields.",),
       stress=("How do you evolve that endpoint a year later without breaking old clients?",)),

    _c("failure", "failure modes",
       ("fail", "failure", "outage", "down", "retry", "timeout", "circuit breaker",
        "fallback", "degrade"),
       companions=("monitoring",),
       deepen=("What's the blast radius when {term} happens? Who notices first?",),
       gap_probe=(
           "Let's talk about failure. What's the first component to break under real load, and how do you know?",
       )),

    _c("hot_key", "hot partitions", ("hot key", "hot partition", "skew", "celebrity problem"),
       deepen=("How would you even detect a hot key in production?",)),
    _c("invalidation", "cache invalidation", ("invalidate", "invalidation", "stale", "evict", "purge")),
    _c("indexing", "indexing", ("index", "indexed", "secondary index", "b-tree", "full text")),
    _c("schema", "schema design", ("schema", "table", "column", "normalis", "normaliz", "denormalis", "denormaliz")),
    _c("idempotency", "idempotency", ("idempoten", "duplicate", "exactly once", "dedup")),
    _c("backpressure", "backpressure", ("backpressure", "rate limit", "throttle", "shed load")),
    _c("capacity_estimate", "capacity estimates", ("gigabyte", "terabyte", "per second", "per day", "estimate")),
    _c("monitoring", "observability", ("monitor", "metric", "alert", "logging", "trace", "dashboard")),
    _c("pagination", "pagination", ("paginat", "cursor", "offset", "page size", "limit")),
    _c("versioning", "versioning", ("version", "backward compatib", "deprecat")),
    _c("tradeoff", "tradeoffs", ("tradeoff", "trade off", "trade-off", "downside", "in exchange")),
    _c("rebalancing", "rebalancing", ("rebalance", "resharding", "migrate the data")),
]

# ----------------------------------------------------------------- BEHAVIOURAL

BEHAVIORAL = [
    _c("situation", "the situation",
       ("we had", "there was", "at the time", "the context", "i was working",
        "my team was", "the company", "back when"),
       companions=("action", "result"),
       opener="Tell me about a time you disagreed with a technical decision on your team.", starter=True,
       deepen=(
           "Set the scene a bit more -- when was {term}, and who else was involved?",
           "What was actually at stake if it went badly?",
       ),
       gap_probe=("Ground this for me: what was the specific situation, and when?",)),

    _c("action", "your specific actions",
       ("i did", "i built", "i decided", "i wrote", "i reached out", "i proposed",
        "i implemented", "so i", "i took", "i led", "i changed"),
       companions=("result", "reasoning"),
       deepen=(
           "You said {term}. What did you personally do, as opposed to the team?",
           "Walk me through the first concrete thing you did after that.",
       ),
       gap_probe=(
           "You've told me what happened, but not what YOU did. What was your specific contribution?",
       )),

    _c("result", "the outcome",
       ("as a result", "the outcome", "we shipped", "it reduced", "we improved",
        "ended up", "in the end", "eventually we", "the impact"),
       companions=("learning",),
       deepen=(
           "How did you measure {term}? Give me the number if you have it.",
           "How did you know it had actually worked?",
       ),
       stress=("What would have happened if you'd done nothing?",),
       gap_probe=("How did it actually turn out? What changed as a result?",)),

    _c("conflict", "disagreement and conflict",
       ("disagree", "conflict", "pushback", "tension", "argued", "friction",
        "didn't see eye", "opposed"),
       companions=("resolution", "empathy"),
       opener="Tell me about a time you had to give difficult feedback to a peer.",
       deepen=(
           "What was their actual argument in {term}? Steelman it for me.",
           "How did the other person feel about how that went?",
       ),
       stress=("If they'd refused to move at all, what would you have done?",)),

    _c("failure", "failure and recovery",
       ("failed", "failure", "mistake", "went wrong", "broke", "outage",
        "regret", "shouldn't have", "in hindsight"),
       companions=("learning", "action"),
       opener="Tell me about something you shipped that broke in production.", starter=True,
       deepen=(
           "What was the first sign that {term} was going wrong?",
           "How much of that was in your control?",
       ),
       stress=("Has that same class of mistake come up again since? What did you do differently?",)),

    _c("learning", "what you took from it",
       ("learned", "learning", "takeaway", "next time", "since then", "taught me",
        "changed how i"),
       deepen=("You said you learned {term}. Where have you actually applied that since?",),
       gap_probe=("What would you do differently if you hit this again tomorrow?",)),

    _c("stakeholder", "working across people",
       ("stakeholder", "product manager", "pm", "designer", "customer", "client",
        "leadership", "cross functional", "other team"),
       companions=("communication",),
       opener="Tell me about a time you had to push back on a product requirement.",
       deepen=("How did you bring {term} around to your view?",),
       stress=("What if they'd escalated over your head?",)),

    _c("ambiguity", "deciding without enough information",
       ("unclear", "ambiguous", "not sure what", "no requirements", "figure out",
        "didn't know", "uncertainty"),
       opener="Describe a time you had to make a call without enough information.",
       deepen=("What was the one piece of information you most wished you'd had?",),
       stress=("How would you have known sooner that you'd picked wrong?",)),

    _c("deadline", "pressure and prioritisation",
       ("deadline", "crunch", "time pressure", "ship date", "launch", "behind schedule"),
       companions=("tradeoff",),
       opener="Tell me about a project where you had to learn something new under a deadline.",
       deepen=("What did you cut to make {term}, and who decided?",)),

    _c("reasoning", "why you chose that", ("because", "the reason", "i thought", "my thinking was", "rationale")),
    _c("resolution", "how it resolved", ("we agreed", "resolved", "compromise", "settled on", "aligned")),
    _c("empathy", "the other perspective", ("their point", "they felt", "from their side", "understood why")),
    _c("communication", "how you communicated", ("explained", "wrote up", "presented", "doc", "meeting", "one on one")),
    _c("tradeoff", "the tradeoff", ("tradeoff", "trade off", "instead of", "at the cost of", "chose to")),
]

TRACK_CONCEPTS: dict[str, list[Concept]] = {
    "dsa": DSA,
    "system_design": SYSTEM_DESIGN,
    "behavioral": BEHAVIORAL,
    "jd": BEHAVIORAL + SYSTEM_DESIGN,
}


# Disfluencies that collide with real aliases. "sort of" is the dangerous one:
# an unanchored substring search reads it as the *sorting* concept, so a
# content-free answer ("it sort of depends") gets probed about complexity. These
# are stripped before detection -- they're already counted as fillers, and they
# carry no content by definition.
_FILLER_PHRASES = (
    "sort of", "kind of", "sorts of", "kinds of", "you know", "i mean",
    "type of", "types of",
)


@dataclass
class ConceptIndex:
    """Lookup helper built once per track."""

    concepts: list[Concept]
    by_key: dict[str, Concept] = field(default_factory=dict)
    _patterns: dict[str, re.Pattern] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.by_key = {c.key: c for c in self.concepts}
        # Anchor each alias at a word start. Without \b, "index" matches
        # "reindex" and "hash" matches "rehash". No trailing \b, because
        # several aliases are deliberate stems ("shardin", "normalis",
        # "idempoten") that must still match their inflections.
        for c in self.concepts:
            joined = "|".join(re.escape(a) for a in sorted(c.aliases, key=len, reverse=True))
            self._patterns[c.key] = re.compile(rf"\b(?:{joined})", re.I)

    @staticmethod
    def _clean(text: str) -> str:
        lowered = text.lower()
        for phrase in _FILLER_PHRASES:
            lowered = lowered.replace(phrase, " ")
        return lowered

    def detect(self, text: str) -> list[str]:
        """Concept keys mentioned in the text, ordered by first appearance."""
        if not text:
            return []
        cleaned = self._clean(text)
        found: list[tuple[int, str]] = []
        for c in self.concepts:
            if (m := self._patterns[c.key].search(cleaned)) is not None:
                found.append((m.start(), c.key))
        return [k for _, k in sorted(found)]

    def openers(self) -> list[Concept]:
        return [c for c in self.concepts if c.opener]
