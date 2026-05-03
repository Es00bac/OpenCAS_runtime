# Changelog

All notable release-doc changes for the current OpenCAS repo are recorded here.

The goal of this changelog is accuracy, not marketing inflation.

## [0.1.1] - 2026-04-19

### Added

- Release docs synchronized with the current runtime surface: voice chat, Twilio-backed phone bridge, schedule API/dashboard, platform/extensions, telemetry logs, and the current dashboard tab set.
- Release website refreshed to feel more like an OpenCAS control room and less like a generic docs template.

### Updated

- Release README aligned to the current CLI, dashboard, and control-plane layout
- Installation guide aligned to the current repo layout and `open_llm_auth` dependency model
- Usage guide aligned to the actual operator paths: dashboard, HTTP API, voice, schedule, phone, platform, logs, and Telegram
- Features guide aligned to current memory, autonomy, execution, channel, platform, and telemetry capabilities
- API reference aligned to the currently mounted FastAPI routes
- Architecture guide aligned to the current subsystem layout

### Current Product Truths

- State and control surfaces are local.
- Chat, voice, and embedding lanes normally use configured providers through `open_llm_auth`.
- The current repo is best described as a local-state autonomous agent with provider-flexible model execution and first-class operator control planes.

## Unreleased

- Future release-note entries go here as the release bundle evolves.
