# Patient Advocacy Project - Architecture Walkthrough

## 🎯 What This Project Does

This is a **voice-first patient advocacy assistant** that:
- Allows users to speak with an AI assistant about health concerns
- Captures webcam snapshots (e.g., visible symptoms) for context
- Maintains conversation history per session
- Uses Google ADK (Agent Development Kit) to power the AI agent

---

## 🏗️ Architecture Overview

```
┌─────────────────┐
│   React Frontend │  (Vite dev server on port 5174)
│   (App.jsx)      │
└────────┬─────────┘
         │ HTTP POST requests
         ▼
┌─────────────────┐
│  FastAPI Backend │  (Uvicorn server on port 8100)
│   (app.py)       │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Google ADK     │
│  - LlmAgent     │  ← The AI agent (patient advocacy assistant)
│  - Runner       │  ← Executes the agent
│  - SessionService│ ← Manages conversation state
└─────────────────┘
```

---

## 📋 Complete Flow: First Step to Last Step

### **STEP 1: Frontend Initialization** (`ui/src/App.jsx`)

When the page loads (`http://localhost:5174`):

1. **Session Creation** (line 26):
   ```javascript
   const [sessionId] = useState(() => generateSessionId());
   ```
   - Generates a unique session ID (e.g., `session_376b3b77-4225-4c31-a78e-1945814542c0`)

2. **Camera/Microphone Access** (lines 41-60):
   - Requests access to webcam and microphone
   - Sets up video stream for snapshot capture

3. **Speech Recognition Setup** (lines 62-107):
   - Initializes browser's Web Speech API
   - Sets up event handlers for voice input

**Status**: Frontend is ready, waiting for user input.

---

### **STEP 2: User Interaction - Voice Input**

When user clicks **"Start Voice"** button (line 173-183):

1. **Speech Recognition Starts**:
   ```javascript
   recognitionRef.current.start();
   ```

2. **User Speaks** → Browser transcribes speech

3. **On Speech Result** (lines 89-102):
   ```javascript
   recognition.onresult = async (event) => {
     const transcript = event.results[0][0].transcript;  // "I have a headache"
     appendLog({ role: "user", text: transcript });
     const result = await sendMessage(transcript);  // ← Calls backend
   }
   ```

---

### **STEP 3: Frontend Sends Message to Backend** (`App.jsx` line 123-139)

```javascript
async function sendMessage(message) {
  const response = await fetch(`${API_BASE}/api/message`, {  // http://localhost:8100/api/message
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      user_id: "patient",
      text: message,
      image_ids: capturedImages,  // Any captured images
    }),
  });
  return response.json();  // { reply: "...", transcript_file: "..." }
}
```

**What happens**: HTTP POST to `http://localhost:8100/api/message`

---

### **STEP 4: Backend Receives Request** (`app.py` line 220-241)

The FastAPI endpoint `/api/message` is triggered:

```python
@app.post("/api/message", response_model=MessageResponse)
async def message_endpoint(request: MessageRequest):
    # 1. Ensure session exists
    await ensure_session(request.user_id, request.session_id)
    
    # 2. Load any attached images
    image_paths: List[Path] = []
    for image_id in request.image_ids:
        candidate = IMAGES_DIR / f"{image_id}.png"
        if candidate.exists():
            image_paths.append(candidate)
    
    # 3. INVOKE THE AGENT ← THIS IS WHERE THE MAGIC HAPPENS
    agent_reply = await invoke_agent(
        text=request.text,
        user_id=request.user_id,
        session_id=request.session_id,
        image_paths=image_paths,
    )
    
    # 4. Save transcript
    transcript_path = save_transcript(...)
    
    return MessageResponse(reply=agent_reply, transcript_file=str(transcript_path))
```

---

### **STEP 5: Agent Invocation - THE CORE** (`app.py` line 143-182)

This is where **Google ADK agents are called**:

```python
async def invoke_agent(
    *,
    text: str,
    user_id: str,
    session_id: str,
    image_paths: List[Path],
) -> str:
    # 1. Build message content with text + images
    content_parts: List[types.Part] = [types.Part(text=text)]
    
    for path in image_paths:
        image_bytes = path.read_bytes()
        content_parts.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=image_bytes))
        )
        content_parts.append(
            types.Part(text="The attached image was captured by the patient...")
        )
    
    # 2. RUN THE AGENT ← THIS IS THE KEY LINE
    events = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=content_parts),
    )
    
    # 3. Stream events and collect response
    responses: list[str] = []
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            text_part = event.content.parts[0].text
            if text_part:
                responses.append(text_part)
    
    return " ".join(responses).strip()
```

**Key Components**:
- **`runner`** (line 81): Created at startup, wraps the agent
- **`agent`** (line 79): The `LlmAgent` instance (defined at line 64-75)
- **`runner.run_async()`**: This is where Google ADK executes the agent

---

### **STEP 6: Google ADK Agent Execution** (Behind the scenes)

When `runner.run_async()` is called:

1. **Session Lookup**: Runner retrieves/creates session from `session_service`
2. **Message History**: Adds new user message to session's message history
3. **Agent Processing**: 
   - The `LlmAgent` (defined at line 64-75) processes the message
   - Uses the model: `LiteLlm(model="ollama_chat/qwen3:8b")` (local Ollama)
   - Applies instruction: "You are a compassionate patient advocacy assistant..."
4. **State Management**: Updates session state automatically
5. **Event Streaming**: Returns async generator of events

**The Agent Definition** (line 64-75):
```python
def build_agent() -> LlmAgent:
    return LlmAgent(
        name="PatientAdvocacyAgent",
        model=LiteLlm(model=DEFAULT_MODEL),  # "ollama_chat/qwen3:8b"
        instruction=(
            "You are a compassionate patient advocacy assistant. "
            "Gather details about the patient's concerns, ask thoughtful follow-up questions, "
            "and always encourage consulting healthcare professionals for diagnosis or treatment. "
            "Provide concise, empathetic responses (2-3 sentences)."
        ),
        output_key="assistant_reply",
    )
```

---

### **STEP 7: Response Flows Back**

1. **Agent returns response** → `invoke_agent()` extracts text
2. **Backend saves transcript** → Creates a `.txt` file in `data/transcripts/`
3. **Backend returns JSON**:
   ```json
   {
     "reply": "I understand you're experiencing a headache...",
     "transcript_file": "/path/to/transcript.txt"
   }
   ```

---

### **STEP 8: Frontend Receives Response** (`App.jsx` line 94-97)

```javascript
const result = await sendMessage(transcript);
appendLog({ role: "assistant", text: result.reply });  // Add to conversation log
speak(result.reply);  // Text-to-speech: browser speaks the response
setStatus("Ready");
```

**Result**: 
- Response appears in conversation log
- Browser speaks the response aloud
- Status returns to "Ready"

---

## 🔍 Where Agents Are Called - Summary

| Location | Function | Line | What It Does |
|----------|----------|------|--------------|
| `app.py` | `invoke_agent()` | 167 | **Main agent call**: `runner.run_async()` |
| `app.py` | `message_endpoint()` | 230 | Calls `invoke_agent()` when API receives request |
| `app.py` | `build_agent()` | 64 | Defines the agent (called once at startup) |
| `app.py` | Module level | 79-81 | Creates agent, session service, and runner at startup |

**The agent is called in `invoke_agent()` → `runner.run_async()`**

---

## 🎬 Complete User Journey Example

1. **User opens** `http://localhost:5174`
   - Frontend loads, requests camera/mic permissions
   - Session ID generated: `session_abc123`

2. **User clicks "Start Voice"**
   - Browser starts listening

3. **User says**: "I have been having headaches for the past week"

4. **Frontend**:
   - Browser transcribes: "I have been having headaches for the past week"
   - Sends POST to `/api/message` with transcript

5. **Backend**:
   - Receives request at `/api/message`
   - Ensures session exists (creates if needed)
   - Calls `invoke_agent()` with the text

6. **Google ADK**:
   - `runner.run_async()` executes the agent
   - Agent (using Ollama qwen3:8b) processes the message
   - Considers conversation history from session
   - Generates response: "I understand you've been experiencing headaches..."

7. **Backend**:
   - Saves transcript to file
   - Returns JSON response

8. **Frontend**:
   - Displays response in conversation log
   - Browser speaks the response
   - Ready for next input

---

## 🖼️ Image Capture Flow

If user clicks **"Capture Snapshot"**:

1. **Frontend** (`App.jsx` line 141-171):
   - Captures frame from video stream
   - Converts to base64 data URL
   - POSTs to `/api/upload-image`

2. **Backend** (`app.py` line 211-217):
   - Saves image to `data/images/`
   - Returns `image_id`

3. **Next message**:
   - Frontend includes `image_ids` in message request
   - Backend loads images and attaches to agent message
   - Agent receives both text and images

---

## 🔑 Key Google ADK Components

1. **`LlmAgent`** (line 64-75):
   - The AI agent definition
   - Specifies model, instructions, behavior

2. **`Runner`** (line 81):
   - Executes agents
   - Manages message flow
   - Handles async event streaming

3. **`InMemorySessionService`** (line 80):
   - Stores conversation history
   - Maintains session state
   - Persists across multiple agent calls

4. **`runner.run_async()`** (line 167):
   - **This is the entry point** for agent execution
   - Takes user message, returns async event stream
   - Automatically updates session

---

## 🎯 The Role of Google ADK Agent - Detailed Explanation

### **What Google ADK Provides (The Framework Layer)**

Google ADK acts as an **abstraction layer** that handles the complex orchestration of LLM interactions. Here's what it does automatically:

#### 1. **Session & Conversation Management** (Automatic)
```python
# When you call runner.run_async(), Google ADK automatically:
# - Retrieves the session from session_service
# - Loads ALL previous messages in the conversation
# - Appends the new user message to the history
# - Sends the ENTIRE conversation context to the LLM
# - Saves the agent's response back to the session
```

**Without Google ADK**, you would need to:
- Manually maintain conversation history
- Format messages for the LLM API
- Track which messages belong to which session
- Handle context window limits
- Manage state updates

**With Google ADK**, you just call `runner.run_async()` and it handles all of this.

#### 2. **Event Streaming** (Automatic)
```python
events = runner.run_async(...)  # Returns async generator
async for event in events:
    if event.is_final_response():
        # Process response
```

Google ADK provides **streaming events** so you can:
- Show partial responses as they're generated
- Handle intermediate states
- React to different event types (start, progress, final, error)

#### 3. **Model Abstraction** (Automatic)
```python
LiteLlm(model="ollama_chat/qwen3:8b")  # Can switch to any model
```

Google ADK provides a **unified interface** for different LLM providers:
- Switch from Ollama to Gemini, OpenAI, etc. without changing code
- Handles different API formats automatically
- Manages authentication and connection details

#### 4. **State Management** (Automatic)
```python
output_key="assistant_reply"  # Agent output stored in session.state
```

The agent can store outputs in session state, enabling:
- Multi-turn reasoning
- Context preservation
- Stateful workflows

---

### **What the Application Code Handles (Your Code)**

Your application code (`app.py`) handles:

1. **Frontend Integration**:
   - Receives HTTP requests from React frontend
   - Parses JSON payloads
   - Returns responses

2. **Audio/Image Processing**:
   - Transcribes audio with Whisper (line 102-119)
   - Saves images to disk (line 131-140)
   - Formats images for the agent (line 150-165)

3. **Business Logic**:
   - Session creation tracking (line 94-99)
   - Transcript file saving (line 122-128)
   - Error handling

4. **Orchestration**:
   - Calls Google ADK at the right time (line 167)
   - Processes ADK events (line 173-177)
   - Formats responses for frontend

---

### **The Division of Responsibilities**

```
┌─────────────────────────────────────────────────────────┐
│ YOUR APPLICATION CODE (app.py)                          │
│                                                          │
│  ✓ HTTP API endpoints                                   │
│  ✓ Audio transcription (Whisper)                        │
│  ✓ Image handling                                       │
│  ✓ File I/O (transcripts, images)                       │
│  ✓ Business logic                                       │
│  ✓ Calls Google ADK when needed                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ Calls runner.run_async()
                       ▼
┌─────────────────────────────────────────────────────────┐
│ GOOGLE ADK (Framework)                                  │
│                                                          │
│  ✓ Session management                                   │
│  ✓ Conversation history tracking                        │
│  ✓ LLM API communication                                │
│  ✓ Event streaming                                      │
│  ✓ State management                                     │
│  ✓ Model abstraction                                    │
│  ✓ Message formatting                                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       │ Uses
                       ▼
┌─────────────────────────────────────────────────────────┐
│ LLM MODEL (Ollama qwen3:8b)                            │
│                                                          │
│  ✓ Generates responses                                  │
│  ✓ Processes conversation context                       │
│  ✓ Applies instructions                                 │
└─────────────────────────────────────────────────────────┘
```

---

### **Why Use Google ADK Instead of Direct LLM Calls?**

**Without Google ADK** (direct LLM API calls):
```python
# You would need to:
conversation_history = load_session(session_id)  # Manual
conversation_history.append({"role": "user", "content": text})  # Manual
response = ollama_client.chat(conversation_history)  # Manual API call
conversation_history.append({"role": "assistant", "content": response})  # Manual
save_session(session_id, conversation_history)  # Manual
# Handle errors, retries, streaming, etc. manually
```

**With Google ADK**:
```python
# Just this:
events = runner.run_async(
    user_id=user_id,
    session_id=session_id,
    new_message=types.Content(role="user", parts=[...])
)
# ADK handles: session loading, history management, API calls, 
#              state updates, error handling, streaming
```

---

### **Specific Role in This Project**

In the Patient Advocacy project, Google ADK's role is to:

1. **Maintain Conversation Context**:
   - When user says "I have a headache" → Agent responds
   - When user says "It's been 3 days" → Agent remembers the headache context
   - This happens automatically through session management

2. **Provide Consistent Interface**:
   - The same code works with different models (Ollama, Gemini, etc.)
   - Easy to switch models without rewriting logic

3. **Handle Complex Message Formats**:
   - Supports text + images in the same message (line 150-165)
   - Formats multi-modal content correctly for the LLM

4. **Enable Future Extensibility**:
   - Easy to add tools (e.g., medical knowledge retrieval)
   - Can chain multiple agents together
   - Can add workflows (sequential, parallel, loops)

---

### **Key Takeaway**

**Google ADK is the "orchestration layer"** that:
- **Simplifies** LLM integration (no manual API management)
- **Manages** conversation state automatically
- **Provides** a consistent interface across different models
- **Enables** advanced features (streaming, tools, workflows)

**Your code focuses on**:
- Business logic (patient advocacy)
- Frontend integration
- Audio/image processing
- Calling ADK when you need AI responses

**The agent itself** (the `LlmAgent`) is just a configuration that tells ADK:
- What model to use
- What instructions to follow
- How to behave

The **real value** is in the **Runner** and **SessionService** that handle all the complexity of maintaining stateful, conversational AI interactions.

---

## 📝 Summary

**First Step**: User opens frontend → Session created → Ready for input

**Agent Call**: Happens in `app.py` → `invoke_agent()` → `runner.run_async()`

**Last Step**: Agent response → Saved to transcript → Returned to frontend → Displayed and spoken

**The Google ADK agent is called every time a user sends a message**, and it maintains conversation context through the session service.


