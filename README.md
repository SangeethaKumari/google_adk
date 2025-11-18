# Patient Advocacy Voice Agent

An end-to-end Google ADK reference project that provides a voice-first patient advocacy assistant. The system combines a FastAPI backend running Google ADK agents with a React front end that captures microphone input and webcam snapshots for richer context.

## Features

- **Conversational voice interface** – users speak with the assistant; speech is transcribed via Whisper and routed through a Google ADK agent.
- **Webcam snapshot capture** – the front end can capture images (e.g., visible symptoms) and forward them to the agent as contextual attachments.
- **Session-aware conversations** – every browser session maintains stateful chat history so the agent can ask follow-up questions.

## Project Layout

```
patientadvocacy/
├── pyproject.toml           # FastAPI + Google ADK dependencies
├── README.md                # Project documentation
├── src/
│   └── patientadvocacy/
│       ├── __init__.py
│       └── app.py           # FastAPI service powering the agent experience
└── ui/
    ├── package.json         # React + Vite front end
    ├── vite.config.js
    └── src/
        ├── App.jsx          # Voice/chat UI with camera capture
        ├── main.jsx
        └── index.css
```

## Prerequisites

- Python 3.12+
- Node.js 18+
- `ffmpeg` installed locally (required by Whisper)
- Ollama running with `qwen3:8b` pulled (or update the model in `app.py`)
- Environment variable `GEMINI_API_KEY` set if you plan to use Google GenAI features beyond ADK (optional for baseline run)

## Backend Setup

```bash
cd /home/sangeethagsk/agent_bootcamp/google_adk/patientadvocacy
UV_HTTP_TIMEOUT=120 uv sync
uv run python -m patientadvocacy.app
```

The FastAPI server listens on `http://localhost:8100` by default. Modify the port in `app.py` if needed.

## Frontend Setup

```bash
cd /home/sangeethagsk/agent_bootcamp/google_adk/patientadvocacy/ui
npm install
npm run dev -- --port 5174
```

Visit `http://localhost:5174`. When prompted, allow microphone and camera access. Use the “Start Voice” button to speak and “Capture Snapshot” to capture webcam imagery. Messages and images are associated with the current session and dispatched to the backend.

Configure a different API base URL by setting `VITE_API_BASE` in a `.env` file inside `ui/`.

## Next Steps

- Integrate additional ADK tools (e.g., medical knowledge retrieval).
- Persist sessions using an external database rather than the in-memory service.
- Deploy the FastAPI backend behind HTTPS so browsers trust microphone/camera usage in production.

