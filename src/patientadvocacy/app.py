from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "patient_advocacy_app"
DEFAULT_MODEL = "ollama_chat/qwen3:8b"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
IMAGES_DIR = DATA_DIR / "images"
AUDIO_DIR = DATA_DIR / "audio"


class MessageRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    user_id: str = Field(default="patient", description="User identifier.")
    text: str = Field(..., min_length=1, description="User's transcribed message.")
    image_ids: List[str] = Field(default_factory=list, description="Optional list of image IDs to attach.")


class MessageResponse(BaseModel):
    reply: str
    transcript_file: str


class ImageUploadRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    image_base64: str = Field(..., description="data URL or raw base64 of the captured image.")


class ImageUploadResponse(BaseModel):
    image_id: str
    file_path: str


class TranscriptResponse(BaseModel):
    session_id: str
    history: List[dict]


def ensure_directories() -> None:
    for directory in (TRANSCRIPTS_DIR, IMAGES_DIR, AUDIO_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_agent() -> LlmAgent:
    return LlmAgent(
        name="PatientAdvocacyAgent",
        model=LiteLlm(model=DEFAULT_MODEL),
        instruction=(
            "You are a compassionate patient advocacy assistant. "
            "Gather details about the patient's concerns, ask thoughtful follow-up questions, "
            "and always encourage consulting healthcare professionals for diagnosis or treatment. "
            "Provide concise, empathetic responses (2-3 sentences)."
        ),
        output_key="assistant_reply",
    )


whisper_model = whisper.load_model("small")
agent = build_agent()
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
created_sessions: set[tuple[str, str]] = set()

app = FastAPI(title="Patient Advocacy Voice Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def ensure_session(user_id: str, session_id: str) -> None:
    key = (user_id, session_id)
    if key in created_sessions:
        return
    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    created_sessions.add(key)


async def transcribe_audio_bytes(audio_data: bytes) -> str:
    loop = asyncio.get_running_loop()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name

    def _whisper(path: str) -> str:
        result = whisper_model.transcribe(path, fp16=False)
        return result.get("text", "").strip()

    try:
        text = await loop.run_in_executor(None, _whisper, tmp_path)
        return text
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


def save_transcript(session_id: str, user_text: str, assistant_text: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    file_path = TRANSCRIPTS_DIR / f"{session_id}_{timestamp}.txt"
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] USER: {user_text}\n")
        handle.write(f"[{timestamp}] ASSISTANT: {assistant_text}\n")
    return file_path


def save_image(session_id: str, image_base64: str) -> Path:
    if image_base64.startswith("data:"):
        header, encoded = image_base64.split(",", 1)
    else:
        encoded = image_base64
    image_bytes = base64.b64decode(encoded)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    file_path = IMAGES_DIR / f"{session_id}_{timestamp}.png"
    file_path.write_bytes(image_bytes)
    return file_path


async def invoke_agent(
    *,
    text: str,
    user_id: str,
    session_id: str,
    image_paths: List[Path],
) -> str:
    content_parts: List[types.Part] = [types.Part(text=text)]
    for path in image_paths:
        try:
            image_bytes = path.read_bytes()
        except FileNotFoundError:
            continue
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes))
        )
        content_parts.append(
            types.Part(
                text=(
                "The attached image was captured by the patient. If relevant, comment on any visible observations."
                )
            )
        )

    events = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=content_parts),
    )
    responses: list[str] = []
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            text_part = event.content.parts[0].text
            if text_part:
                responses.append(text_part)

    if not responses:
        raise HTTPException(status_code=502, detail="Agent produced no response")

    return " ".join(responses).strip()


@app.on_event("startup")
async def startup_event() -> None:
    ensure_directories()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": agent.name}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict[str, str]:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="No audio received")
    text = await transcribe_audio_bytes(data)
    if not text:
        raise HTTPException(status_code=500, detail="Unable to transcribe audio")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    audio_path = AUDIO_DIR / f"{timestamp}.webm"
    audio_path.write_bytes(data)

    return {"transcript": text, "audio_file": str(audio_path)}


@app.post("/api/upload-image", response_model=ImageUploadResponse)
async def upload_image(request: ImageUploadRequest) -> ImageUploadResponse:
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="Image payload missing")
    file_path = save_image(request.session_id, request.image_base64)
    image_id = file_path.stem
    return ImageUploadResponse(image_id=image_id, file_path=str(file_path))


@app.post("/api/message", response_model=MessageResponse)
async def message_endpoint(request: MessageRequest) -> MessageResponse:
    await ensure_session(request.user_id, request.session_id)

    image_paths: List[Path] = []
    for image_id in request.image_ids:
        candidate = IMAGES_DIR / f"{image_id}.png"
        if candidate.exists():
            image_paths.append(candidate)

    agent_reply = await invoke_agent(
        text=request.text,
        user_id=request.user_id,
        session_id=request.session_id,
        image_paths=image_paths,
    )
    transcript_path = save_transcript(
        session_id=request.session_id,
        user_text=request.text,
        assistant_text=agent_reply,
    )
    return MessageResponse(reply=agent_reply, transcript_file=str(transcript_path))


@app.get("/api/session/{user_id}/{session_id}", response_model=TranscriptResponse)
async def session_history(user_id: str, session_id: str) -> TranscriptResponse:
    session = await session_service.get_session_or_none(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    history: List[dict] = []
    for message in session.messages:
        part_texts = []
        for part in message.parts or []:
            if getattr(part, "text", None):
                part_texts.append(part.text)
        history.append(
            {
                "role": message.role,
                "text": " ".join(part_texts),
            }
        )

    return TranscriptResponse(session_id=session_id, history=history)


class AllMessagesResponse(BaseModel):
    total_sessions: int
    total_user_messages: int
    all_user_messages: List[dict]
    combined_text: str


@app.get("/api/all-messages", response_model=AllMessagesResponse)
async def get_all_messages() -> AllMessagesResponse:
    """
    Retrieve all voice-to-text user messages from all conversations.
    Reads from transcript files and active sessions.
    """
    all_user_messages: List[dict] = []
    
    # 1. Read from transcript files
    if TRANSCRIPTS_DIR.exists():
        for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
            try:
                # Extract session_id from filename (format: session_xxx_timestamp.txt)
                # Find the last underscore before the timestamp (which starts with a date)
                filename_parts = transcript_file.stem.split("_")
                # Session ID is everything before the last part (which is the timestamp)
                # Timestamp format: YYYYMMDDTHHMMSS...Z
                session_id_parts = []
                for part in filename_parts:
                    # If part looks like a timestamp (starts with digits and has T), stop
                    if part and part[0].isdigit() and "T" in part:
                        break
                    session_id_parts.append(part)
                session_id = "_".join(session_id_parts) if session_id_parts else transcript_file.stem
                
                with transcript_file.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip().startswith("[") and "USER:" in line:
                            # Extract timestamp and message
                            parts = line.split("USER:", 1)
                            if len(parts) == 2:
                                timestamp_part = parts[0].strip()
                                message_text = parts[1].strip()
                                all_user_messages.append({
                                    "session_id": session_id,
                                    "message": message_text,
                                    "timestamp": timestamp_part,
                                    "source": "transcript_file"
                                })
            except Exception as e:
                print(f"Error reading transcript file {transcript_file}: {e}")
    
    # 2. Also get from active sessions (if available)
    # Note: InMemorySessionService doesn't have a method to list all sessions,
    # so we use the created_sessions set to track them
    for user_id, session_id in created_sessions:
        try:
            session = await session_service.get_session_or_none(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            if session:
                for message in session.messages:
                    if message.role == "user":
                        part_texts = []
                        for part in message.parts or []:
                            if getattr(part, "text", None) and not part.text.startswith("The attached image"):
                                part_texts.append(part.text)
                        if part_texts:
                            all_user_messages.append({
                                "session_id": session_id,
                                "message": " ".join(part_texts),
                                "timestamp": "active_session",
                                "source": "active_session"
                            })
        except Exception as e:
            print(f"Error reading session {session_id}: {e}")
    
    # Get unique sessions
    unique_sessions = set(msg["session_id"] for msg in all_user_messages)
    
    # Combine all messages into one text
    combined_text = "\n".join([f"[{msg['session_id']}] {msg['message']}" for msg in all_user_messages])
    
    # Print to console
    print("\n" + "="*80)
    print("ALL VOICE-TO-TEXT MESSAGES FROM ALL CONVERSATIONS")
    print("="*80)
    print(f"Total Sessions: {len(unique_sessions)}")
    print(f"Total User Messages: {len(all_user_messages)}")
    print("-"*80)
    for msg in all_user_messages:
        print(f"Session: {msg['session_id']}")
        print(f"Message: {msg['message']}")
        print(f"Timestamp: {msg['timestamp']}")
        print("-"*80)
    print("="*80 + "\n")
    
    return AllMessagesResponse(
        total_sessions=len(unique_sessions),
        total_user_messages=len(all_user_messages),
        all_user_messages=all_user_messages,
        combined_text=combined_text
    )


class AllImagesResponse(BaseModel):
    total_images: int
    images: List[dict]


@app.get("/api/all-images", response_model=AllImagesResponse)
async def get_all_images() -> AllImagesResponse:
    """
    List all images saved in the images directory.
    """
    images: List[dict] = []
    
    if IMAGES_DIR.exists():
        for image_file in sorted(IMAGES_DIR.glob("*.png")):
            try:
                # Extract session_id from filename (format: session_xxx_timestamp.png)
                # Find the last underscore before the timestamp (which starts with a date)
                filename_parts = image_file.stem.split("_")
                # Session ID is everything before the last part (which is the timestamp)
                session_id_parts = []
                for part in filename_parts:
                    # If part looks like a timestamp (starts with digits and has T), stop
                    if part and part[0].isdigit() and "T" in part:
                        break
                    session_id_parts.append(part)
                session_id = "_".join(session_id_parts) if session_id_parts else image_file.stem
                timestamp = filename_parts[-1] if len(filename_parts) > len(session_id_parts) else "unknown"
                
                file_size = image_file.stat().st_size
                images.append({
                    "session_id": session_id,
                    "filename": image_file.name,
                    "file_path": str(image_file),
                    "timestamp": timestamp,
                    "size_bytes": file_size,
                    "size_kb": round(file_size / 1024, 2)
                })
            except Exception as e:
                print(f"Error reading image file {image_file}: {e}")
    
    # Print to console
    print("\n" + "="*80)
    print("ALL IMAGES SAVED IN DIRECTORY")
    print("="*80)
    print(f"Total Images: {len(images)}")
    print("-"*80)
    for img in images:
        print(f"Session: {img['session_id']}")
        print(f"Filename: {img['filename']}")
        print(f"Path: {img['file_path']}")
        print(f"Size: {img['size_kb']} KB")
        print("-"*80)
    print("="*80 + "\n")
    
    return AllImagesResponse(
        total_images=len(images),
        images=images
    )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "patientadvocacy.app:app",
        host=os.environ.get("PATIENTADVOCACY_HOST", "0.0.0.0"),
        port=int(os.environ.get("PATIENTADVOCACY_PORT", "8100")),
        reload=False,
    )


if __name__ == "__main__":
    run()

