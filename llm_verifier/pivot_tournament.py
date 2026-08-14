"""
Probabilistic Pivot Tournament (PPT): O(Nk) best-of-N selection.

Instead of a full O(N^2) round-robin, PPT selects the best candidate in
three steps:

  1) Ring pass: score the N adjacent directed pairs of a random Hamiltonian
     cycle. Every candidate appears once in each prompt slot, so the
     verifier's slot bias cancels around the ring.
  2) Pivot selection: the top-k candidates by ring-pass mean preference
     w_i / c_i become the pivot set P.
  3) Pivot rounds: score every non-pivot-vs-pivot and pivot-vs-pivot pair,
     aggregate all comparisons into w_i, c_i, and return argmax_i w_i / c_i.

Total comparisons: N + k(N - k) + C(k, 2) — linear in N for fixed k.

Each comparison's rewards (R_a, R_b) become a soft win via Bradley-Terry,
p(a beats b) = sigmoid(R_a - R_b). The caller supplies a directed
`score(a, b) -> (R_a, R_b)` with `a` in slot A and `b` in slot B.
"""

import math
from itertools import combinations

DEFAULT_PIVOTS = 2


def ring_cycle(n, rng):
    """The N directed adjacent pairs of a random Hamiltonian cycle over `n`
    candidates."""
    if n <= 1:
        return []
    perm = list(range(n))
    rng.shuffle(perm)
    return [(perm[t], perm[(t + 1) % n]) for t in range(n)]


def bradley_terry(ra, rb):
    """p(a beats b) under the Bradley-Terry model on rewards in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-(ra - rb)))


def accumulate(pairs, score, w, c):
    """Score each directed pair and aggregate soft wins into w, c in place."""
    for a, b in pairs:
        ra, rb = score(a, b)
        p = bradley_terry(ra, rb)
        w[a] += p
        c[a] += 1
        w[b] += 1.0 - p
        c[b] += 1


def select_pivots(w, c, k):
    """Top-k candidates by mean preference w_i / c_i (ties broken by index)."""
    n = len(w)
    k = min(k, n)
    order = sorted(
        range(n),
        key=lambda i: (-(w[i] / c[i] if c[i] else 0.0), i))
    return order[:k]


def pivot_round_pairs(n, pivots):
    """Directed pairs for step 3: every non-pivot vs pivot, plus pivot vs
    pivot. Non-pivots take slot A; within P the lower index takes slot A."""
    pivot_set = set(pivots)
    non_pivots = [i for i in range(n) if i not in pivot_set]
    pairs = [(i, p) for i in non_pivots for p in pivots]
    pairs += list(combinations(sorted(pivots), 2))
    return pairs


def select_best(n, ring, k, score):
    """Run the full PPT given a pre-sampled `ring` (from `ring_cycle`) and a
    directed `score(a, b) -> (R_a, R_b)`. Returns (best_index,
    n_comparisons)."""
    w = [0.0] * n
    c = [0] * n

    # Step 1: ring pass.
    accumulate(ring, score, w, c)

    # Step 2: pivots = empirical leaders from the ring pass.
    pivots = select_pivots(w, c, k)

    # Step 3: pivot rounds, aggregated into the same w, c.
    pr_pairs = pivot_round_pairs(n, pivots)
    accumulate(pr_pairs, score, w, c)

    best = max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))
    return best, len(ring) + len(pr_pairs)
