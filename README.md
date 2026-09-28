# hermes-tam-memory

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) memory provider backed by
[total-agent-memory (TAM)](https://github.com/vbcherepanov/total-agent-memory).

With it, Hermes:

- saves each completed conversation turn to TAM, skipping noise such as greetings, slash
  commands and one-line exchanges; TAM's quality gate, if enabled, gets the final say;
- recalls relevant TAM memories before every turn and injects them within a token budget;
- gives the model two tools, `tam_recall` (targeted search) and `tam_save` (store a durable fact,
  decision, solution, lesson or convention);
- mirrors facts that Hermes' built-in memory tool adds;
- shares one memory store with every other agent connected to the same TAM, such as Claude Code,
  Codex CLI or Cursor.

Local-first by default: Hermes starts the `tam` binary installed on your machine and talks to it
over MCP stdio, so conversation text stays in TAM's local database. A remote mode connects to a
TAM HTTP server or a TAM team server for shared company memory.

## Requirements

- Hermes Agent 0.21.4 or newer.
- TAM 14.x installed and on `PATH` (local mode), for example:

  ```bash
  pipx install total-agent-memory      # or: uv tool install total-agent-memory
  tam --help                           # confirms the binary is reachable
  ```

  For remote mode, you need the URL of a running TAM HTTP endpoint. For a team server you also need
  an access token.

The plugin itself has **no Python dependencies**: it speaks MCP with the standard library and never
imports TAM into the Hermes environment.

## Install

From the Hermes plugin catalog (once the catalog entry is merged):

```bash
hermes plugins install tam
```

Directly from GitHub before that. The plugin lives in the `hermes_tam_memory/` subdirectory:

```bash
hermes plugins install vbcherepanov/hermes-tam-memory#hermes_tam_memory
```

As a pip package, for installations whose owner manages the Python environment (for example Nix).
The package registers the `hermes_agent.memory_providers` entry point `tam`:

```bash
pip install git+https://github.com/vbcherepanov/hermes-tam-memory
```

## Set up

```bash
hermes memory setup tam
```

The wizard asks plain questions, so you can also pipe the answers in:

```
Connection mode (local/remote) [local]:
TAM command or absolute path [tam]:
TAM data directory (blank keeps the current value, '-' = TAM default):
TAM project name for Hermes memories [hermes]:
```

In remote mode it asks for the URL (for example `http://127.0.0.1:3737/mcp/`) and an optional team
token instead. After saving, it connects once and prints the result, e.g.
`Connected: total-agent-memory 14.6.0 (local server, MCP 2025-06-18).` Start a new Hermes session to
activate the provider. `hermes memory status` shows the effective settings.

To run a TAM team server for a company:

```bash
tam-team --root /srv/tam user-add alice "Alice"
tam-team --root /srv/tam token-create --client hermes --out alice.token alice
tam-team --root /srv/tam serve --host 0.0.0.0 --port 3737
```

Then choose `remote`, enter `http://<host>:3737/mcp/`, and paste the token. The token is stored as
`TAM_API_TOKEN` in the profile's `.env`, never in `tam.json`. Recalled team memories show their
author (`by Alice`).

## Configuration

Settings live in `$HERMES_HOME/tam.json` (per Hermes profile, file mode 0600). The wizard writes the
first group; edit the file for the rest.

| Key | Default | Meaning |
|---|---|---|
| `mode` | `local` | `local` starts `command` over stdio; `remote` uses `url` over HTTP |
| `command` | `tam` | TAM executable name or absolute path (local) |
| `args` | `[]` | Extra arguments for `command` |
| `memory_dir` | `""` | Sets `TAM_MEMORY_DIR` for the spawned TAM; empty = TAM's default store |
| `env` | `{}` | Extra environment variables for the spawned TAM |
| `url` | `""` | TAM MCP endpoint (remote) |
| `project` | `hermes` | TAM project that saved turns belong to |
| `auto_capture` | `true` | Save completed turns |
| `auto_recall` | `true` | Inject recalled memories before each turn |
| `recall_scope` | `all` | `all` searches every TAM project; `project` only `project` |
| `recall_limit` | `6` | Maximum memories injected per turn (1–20) |
| `recall_budget_tokens` | `800` | Budget for the injected block (estimated at 4 characters per token) |
| `min_turn_chars` | `80` | Turns shorter than this (user and assistant text combined) are not saved |
| `max_turn_chars` | `6000` | Longer turns are truncated before saving |
| `recall_timeout` | `5.0` | Seconds for the per-turn recall; Hermes itself stops waiting at 8 s |
| `request_timeout` | `15.0` | Seconds for saves and tool calls |
| `startup_timeout` | `90.0` | Seconds allowed for TAM to start and finish the MCP handshake |
| `shutdown_timeout` | `5.0` | Seconds to flush queued saves at session end and on exit |
| `pending_limit` | `200` | Queued saves kept while TAM is unreachable; the oldest are dropped first |

Secret: `TAM_API_TOKEN` (remote mode, team server).

Profiles: each Hermes profile has its own `tam.json`. By default, all profiles share TAM's single
store and are told apart by the `hermes-profile:<name>` tag. To separate the data completely, set a
different `memory_dir` per profile.

## What is sent where

- **Saved**: the user message and the final assistant reply of each turn, after removing injected
  `<memory-context>` blocks and inline base64 data. Tool calls, tool results and system prompts are
  **not** saved. Built-in memory `add` writes are mirrored. Subagent, cron and flush runs do not
  save turns automatically; only an explicit `tam_save` call writes from them.
- **Local mode**: data goes to the `tam` process on your machine and into TAM's local store.
  The spawned process gets a reduced environment (`PATH`, `HOME`, locale, temp dirs, and `TAM_*`,
  `MEMORY_*`, `HF_*`, `FASTEMBED_*`, `OLLAMA_*` variables). Hermes' provider API keys are not passed
  to it.
- **Remote mode**: the same text goes to the configured URL. Use `https://` for anything that is
  not on localhost.
- TAM's own optional features (LLM quality gate, enrichment) follow TAM's configuration. See the TAM
  documentation.

## Behaviour when TAM is unavailable

Hermes never fails because of this provider. If TAM is missing, slow or crashes:

- the per-turn recall returns nothing for that turn;
- saves wait in a bounded queue and are retried after reconnecting, with exponential backoff from
  2 s to 120 s;
- `tam_recall` and `tam_save` return a JSON error that the model can read;
- session end and exit wait at most `shutdown_timeout` for queued saves, then log how many were
  abandoned.

Log lines use stable event names such as `tam.connected`, `tam.connect_failed`,
`tam.write_rejected` and `tam.shutdown`. They go to Hermes' logs under `$HERMES_HOME/logs/`. The
`tam.shutdown` line includes per-outcome counters and per-tool call counts with latency buckets.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install -e /path/to/hermes-agent          # tests import Hermes' MemoryProvider and MemoryManager
pytest                                        # unit tests against a scripted fake TAM
ruff check . && ruff format --check .
```

To run the integration tests against a real TAM, use throwaway data directories:

```bash
pipx install total-agent-memory   # or any venv with it installed
TAM_INTEGRATION_COMMAND=$(which tam) TAM_INTEGRATION_TEAM_COMMAND=$(which tam-team) pytest -m integration
```

## License

MIT © 2026 Vitalii Cherepanov
