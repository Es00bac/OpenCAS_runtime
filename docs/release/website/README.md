# OpenCAS Release Website

This directory contains the static release site for the current OpenCAS repo state.

The site is intentionally factual:

- it matches the current dashboard, API, and runtime surfaces
- it includes the newer voice, phone, schedule, platform, logs, and telemetry features
- it avoids placeholder copy and stale roadmap language

## Local Preview

Serve this directory from a simple HTTP server:

```bash
cd docs/release/website
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Site Layout

- `index.html` is the landing page
- `installation.html`, `usage.html`, `features.html`, `terminology.html`, and `api/index.html` are the doc pages
- `styles.css` provides the shared visual system
- `media-player.js` and `media-player.css` drive the optional media button

## Keep It Accurate

When the runtime changes, update the markdown docs in `../` and mirror the same facts here in the website pages so the site does not drift from the actual system.
