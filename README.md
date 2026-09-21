# Operational Twin Gap (clean reproduction)

Study type **EMP**. Pipeline without freeze/controller ceremony. Protocol lives in [`config/protocol.py`](config/protocol.py). The orchestrator is the only writer of [`artifacts/sot.json`](artifacts/sot.json).

## Install

```bash
pip install -r requirements.txt
bash pull_git_util.sh
```

Shared tools live in `util/` (snapshot of the util repo; URL in `.env` as `UTIL_REPO_URL`). Do not commit `.env`.

Place official BR-PVGen files in `data/raw/br_pvgen/` (DOI `10.5281/zenodo.21511487`):

```bash
python scripts/fetch_br_pvgen.py
python scripts/check_ready.py
```

Required files:

- `BR-PVGen_metadata.csv`
- `BR-PVGen_inverter.zip`
- `BR-PVGen_meteorological.zip`
- `Readme.txt`

## Run

```bash
python orchestrator.py list
python orchestrator.py run --from-stage 01 --to-stage 10
```

Stages:

| id | module | science |
|----|--------|---------|
| 01 | load | file presence and checksums |
| 02 | twins | exact nominal groups |
| 03 | panel | plant-level panel + primary scope |
| 04 | models | SplineRidge, HGB, Gself |
| 05 | support | CS4 / CS3 |
| 06 | transfer | zero-shot MAE |
| 07 | otg | RQ1 / RQ2 |
| 08 | rq3 | same-state matching, pairwise CS4 intersection, control−twin contrast |
| 09 | robustness | OFAT protocol (R1–R6 listed; R7 omitted) |
| 10 | figures | primary summary table |

## Compare with the previous SoT

From this directory, after a run:

```bash
python scripts/compare_sot.py --old ../artifacts/sot.json --new artifacts/sot.json
```

Success is matching **scientific numbers** (OTG summaries, RQ3 D_fleet, robustness table), not freeze/status/hash ceremony.

## Executor loop

This tree is the local **LLM executora**. The phrase **execute o loop** means: start `scripts/executor_loop.sh` in the current Cursor session with stdout watched for `AGENT_LOOP_TICK_executor`, then run whatever `.md` sits in `prompts/prompts_running/`.

```bash
# after git init + remotes in .env
bash scripts/executor_loop.sh
```

Queue Protocol v2: drop `Q<UTC>Z__<logical>.md` as a direct child of `prompts/`. The poller claims the FIFO head, prints the tick, and this chat executes it. Contract: [`docs/executor_communication_protocol.md`](docs/executor_communication_protocol.md), [`AGENTS.md`](AGENTS.md) item 12.

This tree is not a git working tree until you initialize it. Without `.git` and `.env`, the poller reports `GIT_ERROR` and will not claim jobs.

## Manuscript

[`paper/main.tex`](paper/main.tex) is the IEEE draft without freeze/pre-execution-rule language. Numbers currently match the completed study; regenerate tables from the ledger after a successful run.
