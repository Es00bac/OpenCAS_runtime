# Changelog

All notable release-doc changes for the current OpenCAS repo are recorded here.

The goal of this changelog is accuracy, not marketing inflation.

## [0.1.0] - 2026-04-10

Initial release documentation bundle for the current OpenCAS repo state.

### Included

- Release README aligned to the current CLI and dashboard
- Installation guide aligned to the current repo layout and `open_llm_auth` dependency model
- Usage guide aligned to the actual operator paths: dashboard, HTTP API, and Telegram
- Features guide aligned to current memory, autonomy, daydream, usage, and Telegram capabilities
- API reference aligned to the currently mounted FastAPI routes
- Architecture guide aligned to the current subsystem layout
- Standalone release website with working documentation pages

### Corrected

- Removed stale references to nonexistent `python -m opencas chat` usage
- Corrected default dashboard address from `localhost:8000` to `127.0.0.1:8080`
- Removed broken website doc links by creating the referenced pages
- Removed inaccurate claims that OpenCAS is cloud-free by default
- Removed external implementation-name references that were not appropriate for release docs

### Current Product Truths

- State and control surfaces are local.
- Chat and embedding lanes normally use configured providers through `open_llm_auth`.
- The current repo is best described as a local-state autonomous agent with provider-flexible model execution.

## Unreleased

- Future release-note entries go here as the release bundle evolves.
