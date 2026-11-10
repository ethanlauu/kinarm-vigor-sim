# Validation performed

On Linux / Python 3.12:
- 9 automated tests passed (Gymnasium/SB3 environment contract, fixed conditions and paired geometry, deadline expiry, dwell and immediate replacement, exactly 12 B payouts, terminal priority, potential-shaping cancellation, constant completion bonus, metric timestamps).
- Both PPO objectives trained for 12,288 smoke-test transitions; checkpoint save/load succeeded.
- Normal production PPO configuration completed one 48,000-transition rollout and update.
- Both saved smoke policies evaluated across all four conditions, including per-step NPZ exports.
- PD baselines at gains 25 and 100 evaluated across all conditions under both objectives.
- Summary CSVs, reward contrasts, learning curves, and behavior PNG generated successfully.
- Human-fit selection ran on a synthetic software-test fixture (NOT human observations).

Full 2-million-transition training across five seeds has NOT been run. No human data were used; no neuroscientific or convergence result is claimed. Windows installation and interactive window rendering require local verification.

Tested package versions (your run writes its own exact package lock):
- numpy: 2.3.5
- gymnasium: 1.4.0
- torch: 2.15.0+cpu
- stable-baselines3: 2.9.2
- pandas: 2.2.3
- matplotlib: 3.10.8
- pytest: 9.2.0
