"""
CS344L - Artificial Intelligence Lab | Lab 6 - Markov Models
Task 1 - A Markov chain over four words  (roll 242816)

Model (in my own words):
  States           : the four words "the", "lab", "is", "open".
  Transition matrix: TRANSITION[cur][nxt] = P(next word is nxt | current word is cur).
  Initial belief   : START - we begin 100% sure the first word is "the".
"""

WORDS = ["the", "lab", "is", "open"]

TRANSITION = {
    "the":  {"the": 0.10, "lab": 0.60, "is": 0.20, "open": 0.10},
    "lab":  {"the": 0.10, "lab": 0.10, "is": 0.70, "open": 0.10},
    "is":   {"the": 0.30, "lab": 0.10, "is": 0.10, "open": 0.50},
    "open": {"the": 0.50, "lab": 0.20, "is": 0.20, "open": 0.10},
}

START = {"the": 1.00, "lab": 0.00, "is": 0.00, "open": 0.00}


def check_rows():
    # Print the total of every transition row and say whether the matrix
    # is valid. A valid chain needs every row to sum to 1.0.
    valid = True
    for word in WORDS:
        total = sum(TRANSITION[word][nxt] for nxt in WORDS)
        good = abs(total - 1.0) < 1e-9
        valid = valid and good
        print(f"   row {word}   sums to {total:.2f}   {'OK' if good else 'BAD'}")
    return valid


def next_step(distribution):
    # One step of the chain: for each possible next word, add up the
    # probability of reaching it from every current word. Two loops, no library.
    new = {}
    for nxt in WORDS:
        total = 0.0
        for cur in WORDS:
            total += distribution[cur] * TRANSITION[cur][nxt]
        new[nxt] = total
    return new


def forecast(start, steps):
    # Apply next_step repeatedly; return the belief at steps 0..steps.
    dist = dict(start)
    history = [dict(dist)]
    for _ in range(steps):
        dist = next_step(dist)
        history.append(dict(dist))
    return history


def find_stationary(tolerance=1e-10):
    # Power iteration: start from a uniform belief and keep stepping until
    # the largest change in any word drops below the tolerance.
    # Returns (stationary distribution, steps taken).
    current = {w: 1.0 / len(WORDS) for w in WORDS}
    step = 0
    while True:
        step += 1
        nxt = next_step(current)
        change = max(abs(nxt[w] - current[w]) for w in WORDS)
        current = nxt
        if change < tolerance:
            return current, step


def main():
    # 1. Validate the matrix.
    print("Checking the matrix:")
    check_rows()
    print()

    # 2. Forecast table, steps 0 to 5, starting from 'the'.
    print("Five steps starting from 'the':")
    print("    step" + "".join(f"{w:>9}" for w in WORDS))
    for i, dist in enumerate(forecast(START, 5)):
        print(f"{i:>8}" + "".join(f"{dist[w]:>9.4f}" for w in WORDS))
    print()

    # 3. Stationary distribution by power iteration.
    pi, n = find_stationary()
    print(f"Stationary distribution (after {n} steps):")
    print(f"   {'the':<6} {pi['the']:.4f}      {'lab':<6} {pi['lab']:.4f}")
    print(f"   {'is':<6} {pi['is']:.4f}      {'open':<6} {pi['open']:.4f}")
    print()

    # 4. Verification: push pi through the matrix once more.
    print("Verify pi * P = pi:")
    back = next_step(pi)
    for w in WORDS:
        diff = abs(back[w] - pi[w])
        print(f"   {w:<6} pi={pi[w]:.4f}  (pi*P)={back[w]:.4f}  diff={diff:.4f}")


if __name__ == "__main__":
    main()
