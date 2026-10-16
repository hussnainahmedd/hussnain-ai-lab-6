"""
CS344L - Artificial Intelligence Lab | Lab 6 - Markov Models
Task 2 - Simulate the chain and verify pi  (roll 242816)

Instead of calculating probabilities, we generate a real word sequence with
the same Markov chain from Task 1. If the maths is right, the fraction of
time the simulation spends on each word must approach Task 1's stationary
distribution as the run gets longer.
"""

import random

WORDS = ["the", "lab", "is", "open"]

TRANSITION = {
    "the":  {"the": 0.10, "lab": 0.60, "is": 0.20, "open": 0.10},
    "lab":  {"the": 0.10, "lab": 0.10, "is": 0.70, "open": 0.10},
    "is":   {"the": 0.30, "lab": 0.10, "is": 0.10, "open": 0.50},
    "open": {"the": 0.50, "lab": 0.20, "is": 0.20, "open": 0.10},
}

START = {"the": 1.00, "lab": 0.00, "is": 0.00, "open": 0.00}


def stationary(tolerance=1e-10):
    # Power iteration on the Task 1 chain: step a uniform belief until it
    # stops moving. Gives the pi we compare the simulation against.
    def next_step(distribution):
        new = {}
        for nxt in WORDS:
            total = 0.0
            for cur in WORDS:
                total += distribution[cur] * TRANSITION[cur][nxt]
            new[nxt] = total
        return new

    current = {w: 1.0 / len(WORDS) for w in WORDS}
    while True:
        nxt = next_step(current)
        if max(abs(nxt[w] - current[w]) for w in WORDS) < tolerance:
            return nxt
        current = nxt


def pick(distribution, rng):
    # Choose one word at random following the given probabilities: lay the
    # probabilities end to end on a 0..1 ruler, throw a dart with rng.random(),
    # and return whichever word's segment the dart lands in.
    dart = rng.random()
    running = 0.0
    for word in WORDS:
        running += distribution[word]
        if dart < running:
            return word
    return WORDS[-1]  # safety net for rounding


def simulate(steps, rng):
    # Generate a real sequence: the first word comes from START, every next
    # word is picked using the current word's transition row.
    sequence = [pick(START, rng)]
    for _ in range(steps - 1):
        sequence.append(pick(TRANSITION[sequence[-1]], rng))
    return sequence


def count_words(sequence):
    # The fraction of the sequence spent on each word.
    n = len(sequence)
    return {w: sequence.count(w) / n for w in WORDS}


def main():
    # 1. The first 12 words of one reproducible run.
    pi = stationary()
    print("First 12 words generated:")
    print("   " + " ".join(simulate(12, random.Random(42))))
    print()

    # 2. Counts for longer and longer runs, each with a fresh Random(42).
    print("Counts for longer and longer runs:")
    print("      steps" + "".join(f"{w:>9}" for w in WORDS) + "   max error")
    for steps in (100, 1000, 10000, 50000):
        seq = simulate(steps, random.Random(42))  # fresh rng for each row
        frac = count_words(seq)
        err = max(abs(frac[w] - pi[w]) for w in WORDS)
        print(f"{steps:>11}" + "".join(f"{frac[w]:>9.4f}" for w in WORDS)
              + f"{err:>11.4f}")
    print(f"{'pi':>11}" + "".join(f"{pi[w]:>9.4f}" for w in WORDS)
          + "           -")


if __name__ == "__main__":
    main()
