# Collaboration

Start from `main`, create a focused branch, and submit a pull request. Preserve
raw files and the original analysis plan. Plan amendments after inspecting
outcomes must be labeled exploratory and must not overwrite primary findings.
Reviewers should check the inferential unit, direction of d, duplicate policy,
training-only preprocessing, assumptions, and interpretation before accuracy.

```powershell
git switch main
git pull --ff-only
git switch -c analysis/describe-your-change
.\.venv\Scripts\python.exe scripts/run_experiment.py --dataset both
.\.venv\Scripts\python.exe -m pytest -q
git add <specific-files>
git commit -m "Describe the concrete change"
git push -u origin analysis/describe-your-change
```

Include source/config changes and regenerated affected outputs in the same PR.
Use a new `--output-dir` for exploratory designs so the original plan/results
remain inspectable. The runner rejects changed prepared inputs and splits.
Keep `.venv`, caches and credentials out of Git. Never pool repeated predictions
as independent pairs. CI reruns both cohorts and checks statistical invariants;
review figures and prose as well as the test result.
