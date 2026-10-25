# 🛡️ IDS CLI models

**No model is shipped with the project.** The CLI uses the models you export from the web trainer:

1. Train in the web trainer (for example on `data/raw/CIC-2017.csv`) and click **Export**. The bundle is saved in `instance/bundles/<run id>.bundle`, with its training record in `instance/runs/<run id>.json`.
2. Start the CLI: it lists your exports (newest first) and you pick one. `--list` shows them, and `--csv` / `--json` / `--info` use the newest one unless you pass `--bundle`.

The model card the CLI prints (and `--info`) comes from the bundle's own training record: dataset and its SHA-256, split, cross-validation and test metrics.

A bundle is opened only if it was exported on this install (signed with `instance/signing.key`). `instance/` never goes to Git, so model size is not limited, but large models (for example Random Forest on the full CIC file) take minutes to load and need several times their size in memory.

## Previous v4 model (withdrawn)

The v4 model (pickles, deleted in v5.0 step 0.5) used **`Attempted Category`**, a label-derived feature: its value comes from the attack label, so the model read the answer from its input. Its published accuracy (~97-98 %) is **withdrawn**. See [docs/baseline/FINDINGS.md](baseline/FINDINGS.md), section 7, and [docs/baseline/cli_v4.json](baseline/cli_v4.json).

