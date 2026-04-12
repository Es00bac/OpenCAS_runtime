# OpenCAS - Autonomous Agent System Documentation

This repository contains the official release documentation and operator handbook for **OpenCAS**, an advanced autonomous agent framework. 

*Note: The core framework source code is currently maintained privately. This repository serves to publicly document the system's architecture, capabilities, and API surfaces.*

## 🧠 Core Capabilities

OpenCAS is designed with a highly advanced cognitive architecture, featuring:

* **Persistent Memory & Retrieval Fusion:** SQLite-backed episodic and distilled memory, edge graph storage, and retrieval logic that combines semantic, keyword, recency, and emotional resonance signals.
* **Somatic & Relational State Modeling:** Tracks arousal, fatigue, tension, and valence to modulate the agent's pacing and retrieval context, while maintaining deep user continuity.
* **Executive State & Autonomy:** Maintains active goals, persistent plans, and handles risk-tiered self-approval for autonomous subagent actions.
* **Idle Daydreaming:** Idle-time reflection, conflict tracking, and creative artifact generation.
* **Operator Surfaces:** A comprehensive dashboard for monitoring token telemetry, system hygiene, session history, and direct process/PTY control.
* **Telegram Integration:** Built-in pairing, DM policies, and remote configuration management via a Telegram bot interface.

## 🚀 Viewing the Documentation

### Live Website
You can view the compiled documentation live via GitHub Pages: 
👉 **[Replace this with your GitHub Pages URL, e.g., https://Es00bac.github.io/OpenCAS/]**

### Local Hosting
To view the documentation locally on your own machine, clone this repository and start a basic HTTP server:

```bash
git clone https://github.com/Es00bac/OpenCAS-Docs.git
cd OpenCAS-Docs
python3 -m http.server 8000
```
Then navigate to `http://localhost:8000` in your web browser.

## 🏗️ Architecture Stack

* **Embeddings:** Provider-backed via `open_llm_auth` (defaulting to `google/gemini-embedding-2-preview`) with deterministic local fallbacks.
* **Storage:** SQLite graph databases.
* **UI:** JetBrains Mono / Inter typography with responsive CSS grids.

---
*Developed by OpenCAS Contributors*
