"""
CS344L - Artificial Intelligence Lab | Lab 6 - Markov Models
Task 3 - A hidden Markov model  (roll 242816)

A factory machine is either Working or Broken, but we cannot see inside it -
we only observe whether each item it produces is Good or Faulty. The hidden
states follow a Markov chain (TRANSITION) and each state emits observations
with its own probabilities (EMISSION). We run the forward algorithm (sum over
all paths) and Viterbi (best single path) on four days of evidence, then
check both against a brute-force search over all 2^4 = 16 hidden paths.
"""

from itertools import product

STATES = ["Working", "Broken"]
OBSERVATIONS = ["Good", "Faulty"]

START = {"Working": 0.80, "Broken": 0.20}

TRANSITION = {
    "Working": {"Working": 0.85, "Broken": 0.15},
    "Broken":  {"Working": 0.30, "Broken": 0.70},
}

EMISSION = {
    "Working": {"Good": 0.90, "Faulty": 0.10},
    "Broken":  {"Good": 0.25, "Faulty": 0.75},
}

EVIDENCE = ["Good", "Faulty", "Faulty", "Good"]


def check_rows():
    # Every transition row and every emission row must sum to 1.0.
    valid = True
    for s in STATES:
        t = sum(TRANSITION[s][nxt] for nxt in STATES)
        e = sum(EMISSION[s][o] for o in OBSERVATIONS)
        good = abs(t - 1.0) < 1e-9 and abs(e - 1.0) < 1e-9
        valid = valid and good
        print(f"   {s:<8} transition sums to {t:.2f}, "
              f"emission sums to {e:.2f}   {'OK' if good else 'BAD'}")
    return valid


def forward(evidence):
    # alpha[day][state] = P(hidden states so far, evidence so far).
    # Day 0: start times emission. Every later day: add up the probability
    # of arriving from each previous state, then multiply by the emission
    # of what was actually seen. Returns (alpha table, P(evidence)).
    alpha = {s: START[s] * EMISSION[s][evidence[0]] for s in STATES}
    table = [dict(alpha)]
    for day in range(1, len(evidence)):
        new = {}
        for current in STATES:
            arriving = sum(alpha[prev] * TRANSITION[prev][current]
                           for prev in STATES)          # sum over paths
            new[current] = arriving * EMISSION[current][evidence[day]]
        alpha = new
        table.append(dict(alpha))
    return table, sum(alpha.values())


def filter_beliefs(table):
    # Turn each alpha row into a belief: divide by the row's own total.
    beliefs = []
    for row in table:
        total = sum(row.values())
        beliefs.append({s: row[s] / total for s in STATES})
    return beliefs


def viterbi(evidence):
    # Like forward, but max instead of sum: keep only the best way of
    # reaching each state, remember where it came from (back-pointers),
    # then walk backwards from the best final state. Returns (path, prob).
    score = {s: START[s] * EMISSION[s][evidence[0]] for s in STATES}
    came_from = [{s: None for s in STATES}]
    for day in range(1, len(evidence)):
        new_score = {}
        new_from = {}
        for current in STATES:
            best_prev = max(STATES,
                            key=lambda p: score[p] * TRANSITION[p][current])
            new_score[current] = (score[best_prev]
                                  * TRANSITION[best_prev][current]
                                  * EMISSION[current][evidence[day]])
            new_from[current] = best_prev
        score = new_score
        came_from.append(new_from)
    last = max(STATES, key=lambda s: score[s])
    prob = score[last]
    path = [last]
    for day in range(len(evidence) - 1, 0, -1):
        last = came_from[day][last]
        path.append(last)
    path.reverse()
    return path, prob


def brute_force(evidence):
    # Try every one of the 2^4 = 16 hidden paths directly.
    # Returns (total probability, best path, best path probability).
    total = 0.0
    best_path = None
    best_prob = -1.0
    for path in product(STATES, repeat=len(evidence)):
        p = START[path[0]] * EMISSION[path[0]][evidence[0]]
        for day in range(1, len(evidence)):
            p *= (TRANSITION[path[day - 1]][path[day]]
                  * EMISSION[path[day]][evidence[day]])
        total += p
        if p > best_prob:
            best_prob = p
            best_path = list(path)
    return total, best_path, best_prob


def main():
    # 1. Validate both matrices.
    print("Checking the model:")
    check_rows()
    print()

    # 2. Forward algorithm.
    table, p_evidence = forward(EVIDENCE)
    print("Forward algorithm")
    print("    day      saw   alpha Working    alpha Broken")
    for day, row in enumerate(table):
        print(f"{day:>7}  {EVIDENCE[day]:>7}"
              f"   {row['Working']:.4f}         {row['Broken']:.4f}")
    print(f"   P(evidence) = {p_evidence:.6f}")
    print()

    # 3. Filtering: belief about each day, with the most likely state.
    print("Filtering (belief about each day)")
    print("    day      saw    Working     Broken   most likely")
    for day, belief in enumerate(filter_beliefs(table)):
        w, b = belief["Working"], belief["Broken"]
        likely = "Working" if w >= b else "Broken"
        print(f"{day:>7}  {EVIDENCE[day]:>7}"
              f"     {w:.4f}     {b:.4f}   {likely}")
    print()

    # 4. Viterbi: the single best hidden path.
    path, prob = viterbi(EVIDENCE)
    print("Viterbi")
    print(f"   best path        : {path}")
    print(f"   its probability  : {prob:.6f}")
    print()

    # 5. Brute-force check over all 16 paths.
    total, best_path, best_prob = brute_force(EVIDENCE)
    same = (abs(total - p_evidence) < 1e-9 and best_path == path
            and abs(best_prob - prob) < 1e-9)
    print("Check against all 16 paths")
    print(f"   same answer : {same}")


if __name__ == "__main__":
    main()
