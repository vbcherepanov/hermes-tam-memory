# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-10-03

### Security
- The spawned `tam` no longer inherits credentials. `HF_*` and `OLLAMA_*` were allowed by prefix,
  which passed Hermes' `HF_TOKEN` and `OLLAMA_API_KEY` to the child. Only the non-secret settings TAM
  reads are passed now (`OLLAMA_URL`, `OLLAMA_EMBED_MODEL`, `FASTEMBED_MODEL`, `FASTEMBED_CACHE_PATH`,
  `HF_HOME`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`), and any variable ending in `_TOKEN`,
  `_API_KEY`, `_SECRET` or `_PASSWORD` is dropped even under an allowed prefix. Keys TAM itself needs
  go through the `env` setting in `tam.json`. Reported in review of NousResearch/hermes-agent#126193.

## [0.1.0] - 2026-09-28

### Added
- `tam` memory provider for Hermes Agent (`memory.provider: tam`).
- Local mode: starts the installed `tam` command and talks MCP over stdio.
- Remote mode: MCP streamable HTTP to a TAM HTTP server or a TAM team server, with an optional
  Bearer token (`TAM_API_TOKEN`).
- Per-turn capture on a background writer with a bounded retry queue; client-side noise filter
  (trivial prompts, very short turns, injected recall blocks); TAM's quality gate verdicts are
  respected and counted.
- Recall before each turn within a token budget, excluding the current session's own turns.
- Agent tools `tam_recall` and `tam_save`.
- Mirroring of built-in memory `add` writes.
- `hermes memory setup tam` flow with a live connection check; `hermes memory status` details.
- Tested against TAM 14.6.0 (local stdio, HTTP transport and team server).
- Graceful degradation: TAM being absent, slow or crashing never raises into Hermes; reconnects
  with exponential backoff; bounded flush on session end and shutdown.
