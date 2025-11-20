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


class AllMessagesResponse(BaseModel):
    total_sessions: int
    total_user_messages: int
    all_user_messages: List[dict]
    combined_text: str


class AllImagesResponse(BaseModel):
    total_images: int
    images: List[dict]


class SessionSummaryResponse(BaseModel):
    session_id: str
    total_messages: int
    all_messages: List[dict]
    combined_text: str
    total_images: int
    images: List[dict]


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

@app.get("/api/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """
    Get all messages and images for a specific session.
    - The API response contains all messages.
    - Only user messages are saved to the conversation_summary file.
    - UI can display only user messages using combined_user_text.
    """
    try:
        all_messages: List[dict] = []
        session_images: List[dict] = []

        print(f"\n[DEBUG] Looking for session: {session_id}")
        print(f"[DEBUG] Transcripts directory: {TRANSCRIPTS_DIR}")
        print(f"[DEBUG] Images directory: {IMAGES_DIR}")

        # --- 1. Load messages from transcript files ---
        if TRANSCRIPTS_DIR.exists():
            matching_files = list(TRANSCRIPTS_DIR.glob(f"{session_id}_*.txt"))
            print(f"[DEBUG] Found {len(matching_files)} transcript files for session {session_id}")

            for transcript_file in sorted(matching_files):
                try:
                    with transcript_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip().startswith("["):
                                if "USER:" in line:
                                    parts = line.split("USER:", 1)
                                    if len(parts) == 2:
                                        timestamp = parts[0].strip()
                                        message = parts[1].strip()
                                        all_messages.append({
                                            "role": "user",
                                            "message": message,
                                            "timestamp": timestamp,
                                            "source": "transcript_file"
                                        })
                                elif "ASSISTANT:" in line:
                                    parts = line.split("ASSISTANT:", 1)
                                    if len(parts) == 2:
                                        timestamp = parts[0].strip()
                                        message = parts[1].strip()
                                        all_messages.append({
                                            "role": "assistant",
                                            "message": message,
                                            "timestamp": timestamp,
                                            "source": "transcript_file"
                                        })
                except Exception as e:
                    print(f"[ERROR] Reading transcript {transcript_file}: {e}")

        # --- 2. Load messages from active session ---
        for user_id, sess_id in created_sessions:
            if sess_id == session_id:
                try:
                    session = session_service.get_session_sync(
                        app_name=APP_NAME, user_id=user_id, session_id=session_id
                    )
                    if session and hasattr(session, 'messages') and session.messages:
                        for message in session.messages:
                            parts_text = []
                            for part in message.parts or []:
                                text = getattr(part, "text", "")
                                if text and not text.startswith("The attached image"):
                                    parts_text.append(text)
                            if parts_text:
                                role = message.role.lower()
                                all_messages.append({
                                    "role": role,
                                    "message": " ".join(parts_text),
                                    "timestamp": "active_session",
                                    "source": "active_session"
                                })
                except Exception as e:
                    print(f"[ERROR] Reading active session {session_id}: {e}")

        # --- 3. Combine only user messages into a single string ---
        user_messages = [msg["message"] for msg in all_messages if msg.get("role") == "user"]
        combined_user_text = " ".join(user_messages)

        # --- 4. Save user conversation to summary file ---
        summary_file = TRANSCRIPTS_DIR / f"{session_id}_conversation_summary.txt"
        with summary_file.open("w", encoding="utf-8") as f:  # overwrite each time
            f.write(combined_user_text)
        print(f"[DEBUG] Saved user conversation to {summary_file}")

        # --- 5. Load images for session ---
        if IMAGES_DIR.exists():
            matching_images = list(IMAGES_DIR.glob(f"{session_id}_*.png"))
            for image_file in sorted(matching_images):
                try:
                    image_bytes = image_file.read_bytes()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    file_size = image_file.stat().st_size
                    timestamp = image_file.stem.split("_")[-1] if "_" in image_file.stem else "unknown"
                    session_images.append({
                        "filename": image_file.name,
                        "file_path": str(image_file),
                        "timestamp": timestamp,
                        "size_bytes": file_size,
                        "size_kb": round(file_size / 1024, 2),
                        "base64": f"data:image/png;base64,{image_base64}"
                    })
                except Exception as e:
                    print(f"[ERROR] Reading image {image_file}: {e}")

        # --- 6. Combine all messages for API response (full conversation) ---
        combined_text = "\n".join(
            f"[{msg.get('timestamp', 'unknown')}] {msg.get('role', '').upper()}: {msg.get('message', '')}"
            for msg in all_messages
        ) if all_messages else "No messages found for this session."

        print("\n" + "="*80)
        print(f"USER CONVERSATION ONLY for session: {session_id}")
        print("-"*80)
        print(combined_user_text)  # for UI display, only user messages
        print("="*80 + "\n")

        if not all_messages and not session_images:
            raise HTTPException(status_code=404, detail=f"No data found for session {session_id}")

        return SessionSummaryResponse(
            session_id=session_id,
            total_messages=len(all_messages),
            all_messages=all_messages,
            #combined_text=combined_text,  # full conversation
            combined_text=combined_user_text, #only user conversation
            total_images=len(session_images),
            images=session_images 
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_session_summary for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving session summary: {str(e)}")



@app.get("/api/session/{session_id}/summary_old2", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """
    Get all messages and images for a specific session.
    Only user messages are appended to the conversation_summary file.
    """
    try:
        all_messages: List[dict] = []
        session_images: List[dict] = []

        print(f"\n[DEBUG] Looking for session: {session_id}")
        print(f"[DEBUG] Transcripts directory: {TRANSCRIPTS_DIR}")
        print(f"[DEBUG] Images directory: {IMAGES_DIR}")

        # --- 1. Load messages from transcript files ---
        if TRANSCRIPTS_DIR.exists():
            matching_files = list(TRANSCRIPTS_DIR.glob(f"{session_id}_*.txt"))
            print(f"[DEBUG] Found {len(matching_files)} transcript files for session {session_id}")

            for transcript_file in sorted(matching_files):
                try:
                    with transcript_file.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip().startswith("["):
                                if "USER:" in line:
                                    parts = line.split("USER:", 1)
                                    if len(parts) == 2:
                                        timestamp = parts[0].strip()
                                        message = parts[1].strip()
                                        all_messages.append({
                                            "role": "user",
                                            "message": message,
                                            "timestamp": timestamp,
                                            "source": "transcript_file"
                                        })
                                elif "ASSISTANT:" in line:
                                    parts = line.split("ASSISTANT:", 1)
                                    if len(parts) == 2:
                                        timestamp = parts[0].strip()
                                        message = parts[1].strip()
                                        all_messages.append({
                                            "role": "assistant",
                                            "message": message,
                                            "timestamp": timestamp,
                                            "source": "transcript_file"
                                        })
                except Exception as e:
                    print(f"[ERROR] Reading transcript {transcript_file}: {e}")

        # --- 2. Load messages from active session (if any) ---
        for user_id, sess_id in created_sessions:
            if sess_id == session_id:
                try:
                    session = session_service.get_session_sync(
                        app_name=APP_NAME, user_id=user_id, session_id=session_id
                    )
                    if session and hasattr(session, 'messages') and session.messages:
                        for message in session.messages:
                            parts_text = []
                            for part in message.parts or []:
                                text = getattr(part, "text", "")
                                if text and not text.startswith("The attached image"):
                                    parts_text.append(text)
                            if parts_text:
                                role = message.role.lower()  # normalize role
                                all_messages.append({
                                    "role": role,
                                    "message": " ".join(parts_text),
                                    "timestamp": "active_session",
                                    "source": "active_session"
                                })
                except Exception as e:
                    print(f"[ERROR] Reading active session {session_id}: {e}")

        # --- 3. Append only USER messages to conversation_summary ---
        summary_file = TRANSCRIPTS_DIR / f"{session_id}_conversation_summary.txt"
        user_messages = [msg for msg in all_messages if msg.get("role") == "user"]
        if user_messages:
            with summary_file.open("a", encoding="utf-8") as f:
                for msg in user_messages:
                    f.write(f"[{msg['timestamp']}] USER: {msg['message']}\n")
            print(f"[DEBUG] Appended {len(user_messages)} user messages to {summary_file}")

        # --- 4. Load images for session ---
        if IMAGES_DIR.exists():
            matching_images = list(IMAGES_DIR.glob(f"{session_id}_*.png"))
            for image_file in sorted(matching_images):
                try:
                    image_bytes = image_file.read_bytes()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    file_size = image_file.stat().st_size
                    timestamp = image_file.stem.split("_")[-1] if "_" in image_file.stem else "unknown"
                    session_images.append({
                        "filename": image_file.name,
                        "file_path": str(image_file),
                        "timestamp": timestamp,
                        "size_bytes": file_size,
                        "size_kb": round(file_size / 1024, 2),
                        "base64": f"data:image/png;base64,{image_base64}"
                    })
                except Exception as e:
                    print(f"[ERROR] Reading image {image_file}: {e}")

        # --- 5. Combine messages for API response ---
        if all_messages:
            combined_text = "\n".join(
                f"[{msg.get('timestamp', 'unknown')}] {msg.get('role', '').upper()}: {msg.get('message', '')}"
                for msg in all_messages
            )
        else:
            combined_text = "No messages found for this session."

        print("\n" + "="*80)
        print(f"SESSION SUMMARY: {session_id}")
        print("="*80)
        print(f"Total Messages: {len(all_messages)}")
        print(f"Total Images: {len(session_images)}")
        print("-"*80)
        print("Combined Text:")
        print(combined_text)
        print("-"*80)
        print("Images:")
        for img in session_images:
            print(f"  - {img['filename']} ({img['size_kb']} KB)")
        print("="*80 + "\n")

        if not all_messages and not session_images:
            raise HTTPException(status_code=404, detail=f"No data found for session {session_id}")

        return SessionSummaryResponse(
            session_id=session_id,
            total_messages=len(all_messages),
            all_messages=all_messages,
            combined_text=combined_text,
            total_images=len(session_images),
            images=session_images
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_session_summary for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving session summary: {str(e)}")



@app.get("/api/session/{session_id}/summary_old", response_model=SessionSummaryResponse)
async def get_session_summary_old(session_id: str) -> SessionSummaryResponse:
    """
    Get all messages and images for a specific session.
    Returns combined text and all images captured in that session.
    IMPORTANT: This route must come BEFORE the /api/session/{user_id}/{session_id} route
    to avoid path parameter conflicts in FastAPI routing.
    """
    try:
        all_messages: List[dict] = []
        session_images: List[dict] = []
        
        print(f"\n[DEBUG] Looking for session: {session_id}")
        print(f"[DEBUG] Transcripts directory: {TRANSCRIPTS_DIR}")
        print(f"[DEBUG] Images directory: {IMAGES_DIR}")
        
        # 1. Get messages from transcript files for this session
        if TRANSCRIPTS_DIR.exists():
            print(f"[DEBUG] Searching for transcript files matching: {session_id}_*.txt")
            matching_files = list(TRANSCRIPTS_DIR.glob(f"{session_id}_*.txt"))
            print(f"[DEBUG] Found {len(matching_files)} matching transcript files")
            
            for transcript_file in sorted(matching_files):
                print(f"[DEBUG] Processing transcript: {transcript_file.name}")
                try:
                    with transcript_file.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip().startswith("["):
                                if "USER:" in line:
                                    parts = line.split("USER:", 1)
                                    if len(parts) == 2:
                                        timestamp_part = parts[0].strip()
                                        message_text = parts[1].strip()
                                        all_messages.append({
                                            "role": "user",
                                            "message": message_text,
                                            "timestamp": timestamp_part,
                                            "source": "transcript_file"
                                        })
                                elif "ASSISTANT:" in line:
                                    parts = line.split("ASSISTANT:", 1)
                                    if len(parts) == 2:
                                        timestamp_part = parts[0].strip()
                                        message_text = parts[1].strip()
                                        all_messages.append({
                                            "role": "assistant",
                                            "message": message_text,
                                            "timestamp": timestamp_part,
                                            "source": "transcript_file"
                                        })
                except Exception as e:
                    print(f"[ERROR] Reading transcript file {transcript_file}: {e}")
        else:
            print(f"[DEBUG] Transcripts directory does not exist: {TRANSCRIPTS_DIR}")
        
        # 2. Also get from active session if available
        for user_id, sess_id in created_sessions:
            if sess_id == session_id:
                print(f"[DEBUG] Found active session for user {user_id}")
                try:
                    session = session_service.get_session_sync(
                        app_name=APP_NAME, user_id=user_id, session_id=session_id
                    )
                    if session and hasattr(session, 'messages') and session.messages:
                        for message in session.messages:
                            part_texts = []
                            for part in message.parts or []:
                                if getattr(part, "text", None) and not part.text.startswith("The attached image"):
                                    part_texts.append(part.text)
                            if part_texts:
                                all_messages.append({
                                    "role": message.role,
                                    "message": " ".join(part_texts),
                                    "timestamp": "active_session",
                                    "source": "active_session"
                                })
                except Exception as e:
                    print(f"[ERROR] Reading active session {session_id}: {e}")
        
        # 3. Get all images for this session
        if IMAGES_DIR.exists():
            print(f"[DEBUG] Searching for images matching: {session_id}_*.png")
            matching_images = list(IMAGES_DIR.glob(f"{session_id}_*.png"))
            print(f"[DEBUG] Found {len(matching_images)} matching images")
            
            for image_file in sorted(matching_images):
                try:
                    file_size = image_file.stat().st_size
                    image_bytes = image_file.read_bytes()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Extract timestamp from filename
                    filename_parts = image_file.stem.split("_")
                    timestamp = filename_parts[-1] if len(filename_parts) > 1 else "unknown"
                    
                    session_images.append({
                        "filename": image_file.name,
                        "file_path": str(image_file),
                        "timestamp": timestamp,
                        "size_bytes": file_size,
                        "size_kb": round(file_size / 1024, 2),
                        "base64": f"data:image/png;base64,{image_base64}"
                    })
                except Exception as e:
                    print(f"[ERROR] Reading image file {image_file}: {e}")
        else:
            print(f"[DEBUG] Images directory does not exist: {IMAGES_DIR}")
        
        # 4. Combine all messages into text
        if all_messages:
            combined_text = "\n".join([
                f"[{msg.get('timestamp', 'unknown')}] {msg.get('role', 'unknown').upper()}: {msg.get('message', '')}"
                for msg in all_messages
            ])
        else:
            combined_text = "No messages found for this session."
        
        print("\n" + "="*80)
        print(f"SESSION SUMMARY: {session_id}")
        print("="*80)
        print(f"Total Messages: {len(all_messages)}")
        print(f"Total Images: {len(session_images)}")
        print("-"*80)
        print("Combined Text:")
        print(combined_text)
        print("-"*80)
        print("Images:")
        for img in session_images:
            print(f"  - {img['filename']} ({img['size_kb']} KB)")
        print("="*80 + "\n")
        
        if not all_messages and not session_images:
            print(f"[WARNING] No messages or images found for session: {session_id}")
            raise HTTPException(status_code=404, detail=f"No data found for session {session_id}")
        
        return SessionSummaryResponse(
            session_id=session_id,
            total_messages=len(all_messages),
            all_messages=all_messages,
            combined_text=combined_text,
            total_images=len(session_images),
            images=session_images
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_session_summary for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving session summary: {str(e)}")


@app.get("/api/session/{user_id}/{session_id}", response_model=TranscriptResponse)
async def session_history(user_id: str, session_id: str) -> TranscriptResponse:
    """
    Get session history from transcript files and active sessions.
    Falls back to transcript files if session is not in memory.
    """
    history: List[dict] = []
    
    # First, try to get from active session
    session = None
    try:
        session = session_service.get_session_sync(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except Exception as e:
        print(f"Error retrieving active session: {e}")
    
    # If session exists in memory, use it
    if session is not None and hasattr(session, 'messages') and session.messages:
        for message in session.messages:
            part_texts = []
            for part in message.parts or []:
                if getattr(part, "text", None):
                    part_texts.append(part.text)
            if part_texts:
                history.append(
                    {
                        "role": message.role,
                        "text": " ".join(part_texts),
                    }
                )
        return TranscriptResponse(session_id=session_id, history=history)
    
    # Fall back to transcript files if session not in memory
    print(f"Session not in memory, reading from transcript files for session_id: {session_id}")
    if TRANSCRIPTS_DIR.exists():
        for transcript_file in sorted(TRANSCRIPTS_DIR.glob(f"{session_id}_*.txt")):
            print(f"Found transcript file: {transcript_file}")
            try:
                with transcript_file.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip().startswith("["):
                            if "USER:" in line:
                                parts = line.split("USER:", 1)
                                if len(parts) == 2:
                                    message_text = parts[1].strip()
                                    history.append({
                                        "role": "user",
                                        "text": message_text,
                                    })
                            elif "ASSISTANT:" in line:
                                parts = line.split("ASSISTANT:", 1)
                                if len(parts) == 2:
                                    message_text = parts[1].strip()
                                    history.append({
                                        "role": "assistant",
                                        "text": message_text,
                                    })
            except Exception as e:
                print(f"Error reading transcript file {transcript_file}: {e}")
    
    if not history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found in memory or transcripts")
    
    print(f"Found {len(history)} messages for session {session_id}")
    return TranscriptResponse(session_id=session_id, history=history)


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
                filename_parts = transcript_file.stem.split("_")
                session_id_parts = []
                for part in filename_parts:
                    if part and part[0].isdigit() and "T" in part:
                        break
                    session_id_parts.append(part)
                session_id = "_".join(session_id_parts) if session_id_parts else transcript_file.stem
                
                with transcript_file.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip().startswith("[") and "USER:" in line:
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
    
    # 2. Also get from active sessions
    for user_id, session_id in created_sessions:
        try:
            session = session_service.get_session_sync(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
            if session and session.messages:
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
    
    unique_sessions = set(msg["session_id"] for msg in all_user_messages)
    combined_text = "\n".join([f"[{msg['session_id']}] {msg['message']}" for msg in all_user_messages])
    
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


class SessionSummaryResponse(BaseModel):
    session_id: str
    total_messages: int
    all_messages: List[dict]
    combined_text: str
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
                filename_parts = image_file.stem.split("_")
                session_id_parts = []
                for part in filename_parts:
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


@app.get("/api/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """
    Get all messages and images for a specific session.
    Returns combined text and all images captured in that session.
    """
    try:
        all_messages: List[dict] = []
        session_images: List[dict] = []
        
        print(f"\n[DEBUG] Looking for session: {session_id}")
        print(f"[DEBUG] Transcripts directory: {TRANSCRIPTS_DIR}")
        print(f"[DEBUG] Images directory: {IMAGES_DIR}")
        
        # 1. Get messages from transcript files for this session
        if TRANSCRIPTS_DIR.exists():
            print(f"[DEBUG] Searching for transcript files matching: {session_id}_*.txt")
            matching_files = list(TRANSCRIPTS_DIR.glob(f"{session_id}_*.txt"))
            print(f"[DEBUG] Found {len(matching_files)} matching transcript files")
            
            for transcript_file in sorted(matching_files):
                print(f"[DEBUG] Processing transcript: {transcript_file.name}")
                try:
                    with transcript_file.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip().startswith("["):
                                if "USER:" in line:
                                    parts = line.split("USER:", 1)
                                    if len(parts) == 2:
                                        timestamp_part = parts[0].strip()
                                        message_text = parts[1].strip()
                                        all_messages.append({
                                            "role": "user",
                                            "message": message_text,
                                            "timestamp": timestamp_part,
                                            "source": "transcript_file"
                                        })
                                elif "ASSISTANT:" in line:
                                    parts = line.split("ASSISTANT:", 1)
                                    if len(parts) == 2:
                                        timestamp_part = parts[0].strip()
                                        message_text = parts[1].strip()
                                        all_messages.append({
                                            "role": "assistant",
                                            "message": message_text,
                                            "timestamp": timestamp_part,
                                            "source": "transcript_file"
                                        })
                except Exception as e:
                    print(f"[ERROR] Reading transcript file {transcript_file}: {e}")
        else:
            print(f"[DEBUG] Transcripts directory does not exist: {TRANSCRIPTS_DIR}")
        
        # 2. Also get from active session if available
        for user_id, sess_id in created_sessions:
            if sess_id == session_id:
                print(f"[DEBUG] Found active session for user {user_id}")
                try:
                    session = session_service.get_session_sync(
                        app_name=APP_NAME, user_id=user_id, session_id=session_id
                    )
                    if session and hasattr(session, 'messages') and session.messages:
                        for message in session.messages:
                            part_texts = []
                            for part in message.parts or []:
                                if getattr(part, "text", None) and not part.text.startswith("The attached image"):
                                    part_texts.append(part.text)
                            if part_texts:
                                all_messages.append({
                                    "role": message.role,
                                    "message": " ".join(part_texts),
                                    "timestamp": "active_session",
                                    "source": "active_session"
                                })
                except Exception as e:
                    print(f"[ERROR] Reading active session {session_id}: {e}")
        
        # 3. Get all images for this session
        if IMAGES_DIR.exists():
            print(f"[DEBUG] Searching for images matching: {session_id}_*.png")
            matching_images = list(IMAGES_DIR.glob(f"{session_id}_*.png"))
            print(f"[DEBUG] Found {len(matching_images)} matching images")
            
            for image_file in sorted(matching_images):
                try:
                    file_size = image_file.stat().st_size
                    image_bytes = image_file.read_bytes()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Extract timestamp from filename
                    filename_parts = image_file.stem.split("_")
                    timestamp = filename_parts[-1] if len(filename_parts) > 1 else "unknown"
                    
                    session_images.append({
                        "filename": image_file.name,
                        "file_path": str(image_file),
                        "timestamp": timestamp,
                        "size_bytes": file_size,
                        "size_kb": round(file_size / 1024, 2),
                        "base64": f"data:image/png;base64,{image_base64}"
                    })
                except Exception as e:
                    print(f"[ERROR] Reading image file {image_file}: {e}")
        else:
            print(f"[DEBUG] Images directory does not exist: {IMAGES_DIR}")
        
        # 4. Combine all messages into text
        if all_messages:
            combined_text = "\n".join([
                f"[{msg.get('timestamp', 'unknown')}] {msg.get('role', 'unknown').upper()}: {msg.get('message', '')}"
                for msg in all_messages
            ])
        else:
            combined_text = "No messages found for this session."
        
        print("\n" + "="*80)
        print(f"SESSION SUMMARY: {session_id}")
        print("="*80)
        print(f"Total Messages: {len(all_messages)}")
        print(f"Total Images: {len(session_images)}")
        print("-"*80)
        print("Combined Text:")
        print(combined_text)
        print("-"*80)
        print("Images:")
        for img in session_images:
            print(f"  - {img['filename']} ({img['size_kb']} KB)")
        print("="*80 + "\n")
        
        if not all_messages and not session_images:
            print(f"[WARNING] No messages or images found for session: {session_id}")
            raise HTTPException(status_code=404, detail=f"No data found for session {session_id}")
        
        return SessionSummaryResponse(
            session_id=session_id,
            total_messages=len(all_messages),
            all_messages=all_messages,
            combined_text=combined_text,
            total_images=len(session_images),
            images=session_images
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] get_session_summary for session {session_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving session summary: {str(e)}")


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