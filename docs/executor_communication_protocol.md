# Executor Communication Protocol

Local execution LLM for this study (`Operational_Twin_Gap` in `new_repo`).

The poller is `scripts/executor_loop.sh` (10s). This document is **not** a queued job.

When the user says **execute o loop**, start the poller in **this** Cursor session with stdout monitored for `AGENT_LOOP_TICK_executor`, then consume any running job.

## Role

You execute queued Markdown jobs. When the loop prints `AGENT_LOOP_TICK_executor`, or when `prompts/prompts_running/` contains exactly one `.md`, consume that job in this session. Do not wait for a human to paste the prompt.

This process is not the scientific controller. A `GO` does not authorize the next stage.

## Canonical paths

- Communication queue: `prompts/` (direct children only; Queue Protocol v2)
- Running: `prompts/prompts_running/` (at most one logical job)
- Consumed prompts: `prompts/prompts_executed/`
- Answers: `prompts/prompts_answers/`
- Transfer: `prompts/prompts_transfer/`
- Superseded (never executable): `prompts/prompts_superseded/`
- Source of Truth: `artifacts/sot.json` (written only by `orchestrator.py`)

## Wake

`scripts/executor_loop.sh` must run in this Cursor session with monitored stdout matching `AGENT_LOOP_TICK_executor`. On that line: read `prompts/prompts_running/`, execute that prompt, commit, push. Do not leave `RECOVERY_REQUIRED` unhandled.

Keep this chat open. The tick only wakes the session that started the monitored loop.

## Official execution

`orchestrator.py` is the only official pipeline executor. Do not run stage files directly unless the active prompt says so.

## Queue Protocol v2

Executable queued jobs are **direct** `.md` children of `prompts/` whose names match:

```text
^Q\d{8}T\d{6}Z__.+\.md$
```

The poller claims the FIFO head (`git mv` into `prompts_running/`, commit, optional push), then wakes this LLM with `RECOVERY_REQUIRED`. Execute that running file.

After GO / STOP / FAILED_* / NEEDS_CONTROLLER_DECISION / REDESIGN_REQUIRED:

- `prompts/prompts_running/<logical name>.md` → `prompts/prompts_executed/<logical name>.md`
- answer: `prompts/prompts_answers/<logical name without .md> - ANSWER.md`

## Poller statuses

| Status | Meaning | LLM action |
| --- | --- | --- |
| `IDLE` | no queued file, nothing running | wait |
| `JOB` | FIFO head in `prompts/` | poller claims it; next tick is `RECOVERY_REQUIRED` |
| `RECOVERY_REQUIRED` | exactly one `.md` in `prompts_running/` | execute that file now |
| `QUEUE_INVALID` | a direct child of `prompts/` is not `Q…Z__….md` | do not execute; report |
| `QUEUE_CONFLICT` | two running jobs, or duplicate queue ids | do not execute; report |
| `GIT_ERROR` | pull/claim/push failed | do not stash or reset; report |

## Preconditions

This tree must be a git working tree with remotes in `.env` (`GIT_PRV_REMOTE_URL`, `GIT_PRV_BRANCH`). Copy `.env` from `.env.example`. Optional Discord wake: `DISCORD_WEBHOOK_URL` (used by `util/discord_msg` via `config/discord_executor.yaml`). Refresh `util/` with `bash pull_git_util.sh`.
