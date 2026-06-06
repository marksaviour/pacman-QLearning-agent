# Q-Learning Pac-Man Agent

A reinforcement-learning Pac-Man agent that learns to play and win by
**Q-learning** — building up action-values purely from experience, with no model
of the environment. Over many training games the agent updates its Q-values from
the rewards it observes, balances **exploration against exploitation**, and
gradually converges on a policy strong enough to win consistently on the
`smallGrid` layout.

This was built as a group coursework project for a **Machine Learning** module,
in collaboration with [danielbate](https://github.com/danielbate).

---

## How it works

### Learning from experience
Pac-Man only learns by actually playing. Each step the agent observes a state,
takes an action, lands in a new state, and receives a reward; from that
transition it updates a Q-value estimating the long-term value of taking that
action in that state. There is no access to a model of the world and no way to
query future states — all knowledge is acquired online, over roughly 2000 games,
using the standard Q-learning update with learning rate **alpha** and discount
factor **gamma**.

### State representation
Rather than feeding the raw `GameState` to the learner, a lightweight
`GameStateFeatures` wrapper extracts only the features that matter for
decision-making (so distinct-but-equivalent game situations map to the same
learned state). It implements `__eq__` and `__hash__` so states can be used as
dictionary keys for Q-values and visit counts.

### Exploration
Acting greedily on early, unreliable Q-values would trap the agent in poor
behaviour, so action selection mixes exploitation with exploration: an
**epsilon-greedy** policy combined with **count-based exploration**, where an
exploration function uses how often a state–action pair has been visited to
favour under-explored options. Once training is complete, `epsilon` and `alpha`
are set to zero so the agent plays purely greedily on its learned values.

### Rewards and terminal states
Reward shaping drives the learning. Crucially, the win reward and lose penalty
are only available in `final()` (Pac-Man's move function is not called once an
episode ends), so terminal outcomes are learned there — without them the agent
would rationally rush into a ghost to cut its losses short.

---

## Running

The agent is invoked through the Berkeley Pac-Man framework. Drop
`mlLearningAgents.py` into a clean copy of the framework from
`pacman_base.zip`, then:

```bash
# Train for 2000 episodes, then play 10 scored games (the grading command)
python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid

# Pass custom hyperparameters (note: one -a each, no spaces)
python3 pacman.py -p QLearnAgent -l smallGrid -a numTraining=2 -a alpha=0.2
```

`-x` sets the number of (interface-free) training episodes, `-n` the total number
of games, and `-l` the layout. Training episodes are not scored; the printed
average reflects only the games played after training.

---

## Requirements

The coursework brief, FAQ, and marksheet imposed the following requirements,
all of which this agent satisfies. The coursework was worth 10% of the module.

### Core task
- Implement the **Q-learning** algorithm to control Pac-Man, with learning done
  **online** purely from playing games — no access to a model of the environment
  and no querying future states (`generatePacmanSuccessor` is disallowed).
- The agent needs two distinct parts: something that **learns** (adjusts Q-values
  from how well it plays) and something that **acts** (chooses moves from the
  Q-values while still exploring enough).
- Use the learning parameters provided: **alpha** (learning rate), **gamma**
  (discount), **epsilon** (exploration rate), and **numTraining** (number of
  training episodes).
- Win/loss rewards are only available in `final()` (since `getAction()` is not
  called after an episode ends), so terminal rewards must be learned there.

### Methods assessed (in `mlLearningAgents.py`)
- `computeReward`, `getQValue`, `updateCount`, `getCount`, `maxQValue`,
  `explorationFn`, `learn`, and `getAction` are each marked individually.
- `explorationFn` must **use counts** to return its value (count-based
  exploration). Epsilon-greedy alone forfeits the count-based-exploration marks;
  count-based, or epsilon-greedy combined with count-based, is acceptable.
- `GameStateFeatures` is a wrapper for extracting only the features useful to the
  learner (rather than using the full `GameState`), and must implement `__eq__`
  and `__hash__`.
- The random action selection in `getAction()` must be replaced with the real
  policy. `__init__`, `final`, `GameStateFeatures`, and additional helper methods
  may all be extended.

### Functionality targets (graded on games won within a time limit)
- Evaluated with: `python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid`
  — i.e. **2000 training episodes** then **10 scored games**.
- Must win **8 of 10** scored games on `smallGrid` for full marks; fewer wins
  scores proportionally. A run below threshold is re-run, with the best result of
  two runs taken.
- The scored run is **terminated after 5 minutes**; whatever has been won by then
  is what counts.
- The same `QLearnAgent` instance runs across all `-n` games, so per-game state
  must be reset in `final()`.
- Must also **run on a second, secret grid** (similar dimensions to the other
  built-in layouts). Only that it runs is tested there — not its win rate — to
  confirm behaviour is learned rather than hard-coded.

### Environment and code constraints
- The base agent class must be named **`QLearnAgent`**, with all submitted code in
  the single file **`mlLearningAgents.py`** (Python 3).
- Code must run in the standard lab environment (Anaconda Python 3). The standard
  Python libraries plus basic libraries such as **NumPy** and **pandas** are
  permitted for this coursework.
- **No modification** of any file in `pacman_base.zip` other than
  `mlLearningAgents.py`.
- Any externally sourced code (including module-distributed code) must be
  **credited** in comments, or it is treated as plagiarism.
- Submit a single ZIP containing only `mlLearningAgents.py` (never the compiled
  `.pyc`); not following submission instructions costs 10% of earned marks.

### Code quality (graded separately)
- Consistent style with good separation of tasks across methods and classes, and
  readable naming/whitespace.
- Clear comments explaining how each part works, including function parameters,
  with high-level references to the underlying theory.

---

## Attribution

The Pac-Man framework this agent runs in was originally developed at **UC
Berkeley** for their CS188 Intro to AI course. Only **`mlLearningAgents.py`** was edited by us — every other file in `pacman_base.zip` is
unmodified Berkeley-provided scaffolding and is not included here.

This was a group project completed with
[danielbate](https://github.com/danielbate).