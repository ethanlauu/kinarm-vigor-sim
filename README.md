# KINARM vigor simulation

Research prototype: does reward change reaching speed differently when faster movement creates more opportunities? A block is **one condition for 60 seconds**: A-low, A-high, B-low, or B-high. Targets do not mix conditions within a block.

- A: next target immediately on acquisition (10 s safety timeout).
- B: targets at 0, 5, ..., 55 s. Old target expires; no backlog. Exactly 12 opportunities.
- Low/high: 1/10 points. Enter target and remain for 150 ms.
- Continuous 2D mass/damping model; force commands in [-1,1]^2. No human-calibrated noise, latency, or metabolism.
- External objective: points minus `c * squared_command * dt`.
- Completion objective adds `(beta0 + beta1*q)*exp(-kappa*target_age)` on acquisition. This is a proposed subjective preference, not an established neuroscientific mechanism.
- PPO: one policy learns all four conditions, separate training per objective and random seed. Gamma=1; potential-based shaping only for training. Shaped training return is NOT money earned.

## Install (Git Bash on Windows)

Use Python 3.11 or 3.12 if available (3.13 can work when compatible wheels exist). Run commands from this folder. No GitHub repository is needed to run locally.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
# CPU wheel avoids unnecessary CUDA downloads; sufficient for these small nets.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip freeze > requirements-lock.txt
python -m pytest -q
```

For Linux/macOS replace activation with `source .venv/bin/activate`. On macOS use `python -m pip install torch` instead of the CPU index. If `python` is unavailable on Windows, use `py -3.12 -m venv .venv` to create the environment, then activate it.

## Quick first run

```bash
python baseline.py --check --episodes 2 --gains 25 100 --out runs
python compare.py --root runs
python demo.py --schedule B --q 10
```

Close the demo window when finished. Try A with `--schedule A --q 10`. The default demo uses a scripted controller, not PPO. `runs/comparison/behavior.png` and CSVs show baseline behavior. Commands may take time: 1 block = 6000 simulation steps.

## Verify the entire PPO pipeline (not a research result)

```bash
python train.py --model external --seeds 0 --steps 12288 --smoke --out smoke_runs
python train.py --model completion --seeds 0 --steps 12288 --smoke --out smoke_runs
python evaluate.py --root smoke_runs --episodes 2
python compare.py --root smoke_runs
```

Smoke training changes rollout/batch/epoch settings for a fast execution check. Expect poor policies, including no successful acquisitions. Empty behavior plots/CSV rows mean insufficient successful trials, not proof of absent reward effects. Training refuses to overwrite a run: use a fresh output directory for reruns.

## Full initial experiment

```bash
python train.py --model external --seeds 0 1 2 3 4 --steps 2000000 --out runs
python train.py --model completion --seeds 0 1 2 3 4 --steps 2000000 --out runs
python evaluate.py --root runs --episodes 100
python compare.py --root runs
python demo.py --model runs/completion/seed_0/model.zip --schedule B --q 10
```

These are ten separate training runs, roughly 20 million total transitions, not a quick check. PPO rounds to full rollouts. Eight environments use DummyVecEnv (batched serial stepping, not eight CPU processes). Exact hyperparameters are in `train.py`; settings and actual transition counts are saved per run. The initial budget does not guarantee convergence. Inspect success rates and learning curves; if extending training, give every candidate the same budget and rerun with a fresh output directory. Full PD gain sweep is optional: `python baseline.py --episodes 100 --out baselines`.

## Outputs and interpretation

`runs/<objective>/seed_<n>/` contains model.zip, config.json, installed package versions, progress.csv, per-environment monitor logs, and intermediate checkpoints. `evaluation/` contains:

- `blocks.csv`: raw utility, points, success rate (all presented targets), waiting movement.
- `trials.csv`: reaction time, movement time excluding final dwell, peak speed, actual distance, and outcome. Unfinished final targets are `block_end`, separate from `expired`; both earned zero.
- Optional per-block traces: rerun evaluation with `--save-steps`. Files in `evaluation/steps/` contain `columns` and a Unicode `data` matrix (`np.load(..., allow_pickle=False)`), plus schedule/q. Convert numeric columns to float; event is text. Old position/target/active state refer to the action's target; new position/velocity and `dwell_after` refer to the end of the step. Separate files bound memory and avoid one huge NPZ.

Evaluation deterministically selects actions. All models use new seeds 100000--100099 for paired target geometry; training seeds are separate. Evaluate all four conditions and inspect failures, not just successful movements. Human reaction-time variation is out of scope until delay/noise is calibrated.

`comparison/` provides raw distance-bin means, distance-matched means, high-minus-low log-MT effects and the A-minus-B interaction, points/success summaries, behavior plot, and PPO learning plots. Matched means use only distance bins present in every condition of a run; inspect `matched_bins`. Runs with no successful matched cells cannot provide a speed contrast. Outputs are descriptive, not significance tests. Five training seeds are algorithm replicates, not human subjects.

Evaluation overwrites the same run's `evaluation/` tables; use a copy of the run directory to retain multiple evaluations. If changing the episode count, old optional step files may remain; the CSVs define the current evaluated blocks.

## Human-data fitting (later)

Do not upload private participant data to GitHub. `human_data/` is ignored. Input CSV needs `split,schedule,q,distance,mt,peak_speed,success`. Assign **whole blocks** to `train`/`test` before fitting; for generalization to new people, split by participant. Use metres/seconds and the same event definitions. Each split needs valid successful trials.

Generate candidate configs varying c (external) and c,beta0,kappa (completion), initially values 0.3,1,3, keeping beta1=0. Each candidate gets its own `--config candidate.json --out candidates/name` training directory. Train/evaluate every candidate with the same budget/seeds before running:

```bash
python fit_human.py --human human_data/trials.csv --candidates candidates --out human_fit
```

The script ignores smoke models, averages simulator summaries over seeds, standardizes differences with training-human SDs, selects the best candidate **per objective on training data only**, then reports held-out loss. Candidates missing human training bins receive infinite loss. It does not automatically launch the large grid, fabricate human data, or perform inferential statistics. The user must maintain identical budgets/seeds across candidates. Predictive success does not demonstrate that brains implement PPO.

## Publish to GitHub (after local checks)

1. Visit https://github.com/new while signed in as **ethanlauu**.
2. Repository name: **kinarm-vigor-sim**; choose private initially if desired. Leave README, .gitignore, and license initialization unchecked (files already exist locally). Click Create repository.
3. From this project folder in Git Bash:

```bash
git init -b main
git add .
git status
git commit -m "Implement reaching-task simulator and PPO experiment pipeline"
git remote add origin https://github.com/ethanlauu/kinarm-vigor-sim.git
git push -u origin main
```

Authenticate through the browser/Git Credential Manager if prompted. Do not use your GitHub account password as a Git HTTPS password. If Git requests identity, set your own name and verified or GitHub noreply email with `git config user.name` and `git config user.email`, then retry the commit. No GitHub CLI is required. Runs, virtual environments, and human data are ignored. This download does not create or publish a remote repository.

## Source map

`env.py`: physics, schedule, reward, metrics; `rollout.py`: exports; `baseline.py`: PD sweep; `train.py`: PPO; `evaluate.py`: frozen policies; `compare.py`: summaries; `demo.py`: animation; `fit_human.py`: candidate selection; `tests/`: task invariants.

Reference: Chew et al. (2021), https://www.jneurosci.org/content/41/43/8963 (intrinsic performance value; not this completion-time formula). PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html . The included specification records the proposed experiment; software changes/clarifications above take precedence for output organization and implementation details.
