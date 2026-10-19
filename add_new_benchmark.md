Hi! Here are the trajectories for a new task. The candidate trajectories, each
with its ground-truth reward (success / failure), live in
`data/task_name_trajs/`. Please stand up a verifier for this new benchmark by
doing the following.

## 1. Get oriented

Read `README.md` to understand the framework, then let's kick off a new verifier
for this benchmark. Inspect `data/task_name_trajs/` first to learn the
layout — the task/problem prompt, each candidate trajectory, and the
success/failure label.

## 2. Generate criteria

Write `criteria/task_name.md` using the exact format the loader
expects:

- `## Ground Truth Note`
- `## Criteria`
- one `### <id> — <Name>` block per criterion

Aim for **2–4 criteria** that are *decidable from the trajectory alone* and that
target this task's common failure modes. The criteria should avoid label leakage 
or reward hacking: they should evaluate observable behavior in the trajectory.
If several criteria decompose one broader scoring dimension, pass a complete
`criterion_weights={"criterion_a": 1.0, "criterion_b": 0.5}` mapping to
`llm_verifier.select` so that dimension keeps its intended aggregate weight.

## 3. Write a runner

Create `adapt_run.py` that:

- loads the candidate trajectories from `data/task_name_trajs/` into a list of strings,
- sets `problem` to the task prompt,
- calls:

  ```python
  llm_verifier.select(
      problem, trajectories,
      criteria="task_name",
      n_evaluations=8,   # K
      pivots=2,            # k
      seed=0,
      max_workers=50,      
  )
  ```

- prints `result.index`, `result.best`, `result.scores`, and `result.n_comparisons`.

Since ground-truth rewards are available, also report whether the selected
trajectory was actually a success (look up the label for `result.index`).

Have the runner save the run summary (chosen index, per-trajectory scores,
whether the pick was correct, and the config used) to
`results/task_name.txt`, matching the
existing `results/` convention.

## 4. Check credentials

Confirm `.env` has a `VERTEX_API_KEY` (Vertex AI only — logprob extraction
needs the Vertex API). If
not, tell me to add one — scoring is live API traffic.

## 5. Run it

Run `python adapt_run.py`, then report the chosen trajectory, the per-trajectory
scores, and whether the pick was correct.

## Constraints

Keep it minimal: one criteria file, one runner. Do not modify the core framework
(`fine_grained_reward.py`, `pivot_tournament.py`).
