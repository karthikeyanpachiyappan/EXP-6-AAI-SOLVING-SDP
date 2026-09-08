# Exp - 6(AAI) - Solving a Stochastic Grid-World Markov Decision Process Using Value Iteration and Policy Iteration

A compact Python implementation of two dynamic-programming methods for solving a stochastic grid-world Markov decision process (MDP):

- **Value Iteration**
- **Policy Iteration**

The program calculates the utility of each state and prints the resulting greedy policy.

## Why sequential decision problems matter

A **sequential decision problem** is one in which a decision made now affects what choices, rewards, and risks are available later. The objective is not simply to choose the action with the best immediate result; it is to choose actions that lead to the best *long-term* outcome despite uncertainty.

This grid world is a small example: moving toward the goal may be worthwhile, but a movement can drift unexpectedly toward the trap. The agent must therefore balance the immediate step cost, the chance of reaching the reward, and the risk of a future penalty.

Solving problems like this is important because many real systems operate through a sequence of connected decisions:

- **Robotics and navigation:** choosing safe routes while accounting for imperfect movement and obstacles.
- **Operations and logistics:** planning inventory, deliveries, and resources when demand and travel conditions are uncertain.
- **Finance:** weighing current returns against longer-term risk and future market outcomes.
- **Healthcare:** selecting treatments over time as a patient's condition and response evolve.
- **Reinforcement learning:** training an agent to act effectively from feedback rather than fixed instructions.

Value iteration and policy iteration provide principled ways to solve a Markov decision process (MDP). They evaluate both immediate rewards and expected future utility, producing a policy that tells an agent what action to take in each state. This makes them foundational techniques for planning under uncertainty.

## Mathematical formulation


| Symbol | Meaning |
| --- | --- |
| s| Current state |
| a | Action selected in state \(s\) |
| s'| Possible successor (next) state |
| R(s) | Immediate reward in state \(s\) |
| P(s' \| s, a) | Probability of reaching \(s'\) after taking \(a\) in \(s\). The vertical bar \| means “given.” |
| γ | Discount factor, which controls how much future rewards matter relative to immediate rewards. In this program, γ = 1.0. |

### Bellman optimality equation

The optimal utility of a state is its immediate reward plus the discounted expected utility of the best available action:

$$
V^{\star}(s) = R(s) + \gamma \max_{a \in A(s)} \sum_{s'} P(s' \mid s, a)V^{*}(s')
$$

### Value iteration update

Value iteration repeatedly applies the Bellman optimality update until successive utility values change by less than the chosen tolerance:

$$
V_{k+1}(s) = R(s) + \gamma \max_{a \in A(s)} \sum_{s'} P(s' \mid s, a)V_k(s')
$$

For this program, an intended action succeeds with probability \(0.8\), while the agent drifts left or right with probability \(0.1\) each. Therefore, the expected successor utility is:

$$
\sum_{s'} P(s' \mid s, a)V(s') =
0.8V(s_{\text{intended}}) +
0.1V(s_{\text{left}}) +
0.1V(s_{\text{right}})
$$

### Arbitrary initial policy

Policy iteration begins with any valid action at each non-terminal state:

$$
\pi_0(s) \in A(s), \quad \forall s \notin \text{Terminal States}
$$

This implementation initializes the policy with `UP` for every non-terminal state.

### Policy evaluation

For a fixed policy \(\pi\), the utility of each state is calculated as:

$$
V^{\pi}(s) = R(s) + \gamma \sum_{s'} P(s' \mid s, \pi(s))V^{\pi}(s')
$$

### Policy improvement

The policy is improved by selecting the action with the largest expected successor utility:

$$
\pi_{\text{new}}(s) =
\arg\max_{a \in A(s)}
\sum_{s'} P(s' \mid s, a)V^{\pi}(s')
$$

Policy evaluation and improvement repeat until the policy does not change:

$$
\pi_{\text{new}} = \pi
$$

## Grid-world configuration

The environment is a 3 × 4 grid:

```text
+-------+-------+-------+--------+
| (2,0) | (2,1) | (2,2) | Goal   |
+-------+-------+-------+--------+
| (1,0) | Block | (1,2) | Trap   |
+-------+-------+-------+--------+
| (0,0) | (0,1) | (0,2) | (0,3)  |
+-------+-------+-------+--------+
```

| Element | Details |
| --- | --- |
| Normal-state reward | `-0.04` |
| Goal at `(2, 3)` | terminal state with reward `+1.0` |
| Trap at `(1, 3)` | terminal state with reward `-1.0` |
| Blocked cell | `(1, 1)` |
| Discount factor | `γ = 1.0` |
| Convergence threshold | `ε = 1e-4` |

Coordinates use `(row, column)` with `(0, 0)` at the bottom-left. The displayed arrays are flipped vertically so the top row appears first.

## Movement model

The available actions are `UP`, `DOWN`, `LEFT`, and `RIGHT`.

An intended action succeeds with probability **0.8**. With probability **0.1** each, the agent drifts to the action on its left or right. If a move would leave the grid or enter the blocked cell, the agent stays in its current state.

## Requirements

- Python 3.8 or newer
- NumPy

Install the dependency:

```bash
python3 -m pip install numpy
```

## Run

```bash
python3 valuePolicyIter.py
```

By default, the script runs value iteration and prints a utility table followed by the extracted policy.

Example output:

```text
--- Final Utility Table ---
[[ 0.812  0.868  0.918  1.   ]
 [ 0.762  0.796  0.66  -1.   ]
 [ 0.705  0.655  0.611  0.388]]

--- Extracted Policy Layout ---
[['RIGHT' 'RIGHT' 'RIGHT' 'GOAL']
 ['UP' 'UP' 'UP' 'TRAP']
 ['UP' 'LEFT' 'LEFT' 'LEFT']]
```
### Program:

```python
import numpy as np

# ============================================================
# EXPERIMENT 6 - SOLVING STOCHASTIC GRID-WORLD MDP
# USING VALUE ITERATION AND POLICY ITERATION
# ============================================================

# ------------------------------------------------------------
# 1. Grid Configuration
# ------------------------------------------------------------

rows, cols = 3, 4

# Rewards
R = np.full((rows, cols), -0.04)

# Goal and Trap
R[2, 3] = 1.0
R[1, 3] = -1.0

# Terminal states
terminals = [(2, 3), (1, 3)]

# Available actions
actions = ["UP", "DOWN", "LEFT", "RIGHT"]

# Discount factor and convergence threshold
gamma = 1.0
epsilon = 1e-4


# ------------------------------------------------------------
# 2. Movement Rules
# ------------------------------------------------------------

def get_next_state(r, c, action):

    if action == "UP":
        next_r, next_c = r + 1, c

    elif action == "DOWN":
        next_r, next_c = r - 1, c

    elif action == "LEFT":
        next_r, next_c = r, c - 1

    elif action == "RIGHT":
        next_r, next_c = r, c + 1

    # Boundary check
    if not (0 <= next_r < rows and 0 <= next_c < cols):
        return r, c

    # Blocked cell
    if next_r == 1 and next_c == 1:
        return r, c

    return next_r, next_c


# ------------------------------------------------------------
# 3. Stochastic Movement Model
# ------------------------------------------------------------

def get_action_distribution(action):

    if action == "UP":
        return "UP", "LEFT", "RIGHT"

    elif action == "DOWN":
        return "DOWN", "RIGHT", "LEFT"

    elif action == "LEFT":
        return "LEFT", "DOWN", "UP"

    elif action == "RIGHT":
        return "RIGHT", "UP", "DOWN"


# ------------------------------------------------------------
# 4. Expected Utility
# ------------------------------------------------------------

def expected_utility(U, r, c, action):

    intended, left, right = get_action_distribution(action)

    # Intended movement
    ri, ci = get_next_state(r, c, intended)

    # Left drift
    rl, cl = get_next_state(r, c, left)

    # Right drift
    rr, cr = get_next_state(r, c, right)

    # Probability:
    # 0.8 -> intended
    # 0.1 -> left drift
    # 0.1 -> right drift

    return (
        0.8 * U[ri, ci]
        + 0.1 * U[rl, cl]
        + 0.1 * U[rr, cr]
    )


# ============================================================
# 5. VALUE ITERATION
# ============================================================

def value_iteration(gamma=1.0, epsilon=1e-4):

    # Initial utilities
    U = np.zeros((rows, cols))

    # Keep terminal utilities fixed
    for r, c in terminals:
        U[r, c] = R[r, c]

    iterations = 0

    while True:

        U_next = U.copy()
        delta = 0.0

        for r in range(rows):
            for c in range(cols):

                # Do not update terminal states
                if (r, c) in terminals:
                    continue

                # Find maximum expected utility
                best_action_value = max(
                    expected_utility(U, r, c, action)
                    for action in actions
                )

                # Bellman update
                U_next[r, c] = (
                    R[r, c]
                    + gamma * best_action_value
                )

                delta = max(
                    delta,
                    abs(U_next[r, c] - U[r, c])
                )

        U = U_next
        iterations += 1

        # Stop when converged
        if delta < epsilon:
            break

    return U, extract_policy(U), iterations


# ============================================================
# 6. EXTRACT GREEDY POLICY
# ============================================================

def extract_policy(U):

    policy = {}

    for r in range(rows):
        for c in range(cols):

            if (r, c) in terminals:

                if (r, c) == (2, 3):
                    policy[(r, c)] = "GOAL"
                else:
                    policy[(r, c)] = "TRAP"

                continue

            # Choose action having maximum expected utility
            policy[(r, c)] = max(
                actions,
                key=lambda action:
                expected_utility(U, r, c, action)
            )

    return policy


# ============================================================
# 7. POLICY ITERATION
# ============================================================

def policy_iteration(gamma=1.0, epsilon=1e-4):

    # Initial utilities
    U = np.zeros((rows, cols))

    # Terminal utilities
    for r, c in terminals:
        U[r, c] = R[r, c]

    # Initial arbitrary policy
    policy = {}

    for r in range(rows):
        for c in range(cols):

            if (r, c) == (2, 3):
                policy[(r, c)] = "GOAL"

            elif (r, c) == (1, 3):
                policy[(r, c)] = "TRAP"

            else:
                policy[(r, c)] = "UP"

    iterations = 0

    while True:

        # ----------------------------------------------------
        # Policy Evaluation
        # ----------------------------------------------------

        while True:

            U_next = U.copy()
            delta = 0.0

            for r in range(rows):
                for c in range(cols):

                    if (r, c) in terminals:
                        continue

                    action = policy[(r, c)]

                    U_next[r, c] = (
                        R[r, c]
                        + gamma *
                        expected_utility(
                            U, r, c, action
                        )
                    )

                    delta = max(
                        delta,
                        abs(U_next[r, c] - U[r, c])
                    )

            U = U_next

            if delta < epsilon:
                break

        # ----------------------------------------------------
        # Policy Improvement
        # ----------------------------------------------------

        policy_stable = True

        for r in range(rows):
            for c in range(cols):

                if (r, c) in terminals:
                    continue

                old_action = policy[(r, c)]

                best_action = max(
                    actions,
                    key=lambda action:
                    expected_utility(
                        U, r, c, action
                    )
                )

                policy[(r, c)] = best_action

                if best_action != old_action:
                    policy_stable = False

        iterations += 1

        # Stop if policy no longer changes
        if policy_stable:
            break

    return U, policy, iterations


# ============================================================
# 8. DISPLAY FUNCTION
# ============================================================

def display_results(U, policy, title, iterations):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print("\nFinal Utility Table:")

    utility_grid = np.flipud(U)

    print(np.round(utility_grid, 3))

    print("\nExtracted Policy:")

    policy_grid = np.empty(
        (rows, cols),
        dtype=object
    )

    for r in range(rows):
        for c in range(cols):
            policy_grid[r, c] = policy[(r, c)]

    print(np.flipud(policy_grid))

    print("\nNumber of iterations:", iterations)


# ============================================================
# 9. RUN VALUE ITERATION
# ============================================================

U_value, policy_value, value_iterations = \
    value_iteration(gamma, epsilon)

display_results(
    U_value,
    policy_value,
    "VALUE ITERATION",
    value_iterations
)


# ============================================================
# 10. RUN POLICY ITERATION
# ============================================================

U_policy, policy_policy, policy_iterations = \
    policy_iteration(gamma, epsilon)

display_results(
    U_policy,
    policy_policy,
    "POLICY ITERATION",
    policy_iterations
)
```

### Output

<img width="655" height="622" alt="image" src="https://github.com/user-attachments/assets/5cb33bb5-9b8d-4052-b0a8-0320d7c34ade" />


### Value iteration (default)

```python
U, policy = value_iteration(gamma, epsilon)
```

Value iteration repeatedly applies the Bellman optimality update until the largest utility change is smaller than `epsilon`.

### Policy iteration

Comment out the value-iteration line and enable this one:

```python
U, policy = policy_iteration(gamma, epsilon)
```

Policy iteration alternates between evaluating the current policy and improving it until no action changes.

## Customize the environment

Edit these values near the top of the script to experiment with other MDPs:

- `rows`, `cols` — grid size
- `R` — reward table
- `terminals` — terminal-state coordinates
- `gamma` — discount factor
- `epsilon` — stopping tolerance
- `get_next_state()` — boundaries and blocked-cell behavior
- `get_action_distribution()` / `expected_utility()` — transition dynamics

## Notes

- Terminal-state utilities stay fixed at their assigned rewards.
- The policy grid includes a label for every non-terminal coordinate, including the blocked coordinate. Since that cell cannot be entered, its displayed action is not part of the reachable environment.
