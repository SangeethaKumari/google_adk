import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8100";
const DEFAULT_USER = "patient";

function generateSessionId_old() {
  return `session_${crypto.randomUUID()}`;
}

function generateSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `session_${crypto.randomUUID()}`;
  }
  const fallback = Math.random().toString(36).slice(2);
  return `session_${fallback}`;
}

function generateLogId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return Math.random().toString(36).slice(2);
}

export default function App() {
  const [sessionId] = useState(() => generateSessionId());
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState("Ready");
  const [isListening, setIsListening] = useState(false);
  const [capturedImages, setCapturedImages] = useState([]);
  const [imagePreview, setImagePreview] = useState(null);

  const videoRef = useRef(null);
  const recognitionRef = useRef(null);

  const speechSupported = useMemo(
    () => typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window),
    [],
  );

  useEffect(() => {
    let stream;
    async function initCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Camera access denied:", error);
        setStatus("Camera or microphone permission denied. Enable them to continue.");
      }
    }
    initCamera();
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!speechSupported) {
      setStatus("Web Speech API not supported. Try Chrome desktop.");
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.continuous = false;

    recognition.onstart = () => {
      setIsListening(true);
      setStatus("Listening...");
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event);
      setStatus(`Speech recognition error: ${event.error}`);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
      setStatus("Ready");
    };

    recognition.onresult = async (event) => {
      const transcript = event.results[0][0].transcript;
      appendLog({ role: "user", text: transcript });
      setStatus("Sending message...");
      try {
        const result = await sendMessage(transcript);
        appendLog({ role: "assistant", text: result.reply });
        speak(result.reply);
        setStatus("Ready");
      } catch (error) {
        console.error(error);
        setStatus(`Failed to contact backend: ${error.message}`);
      }
    };

    recognitionRef.current = recognition;

    return () => recognition.abort();
  }, [speechSupported]);

  function appendLog(entry) {
    setLogs((prev) => [...prev, { ...entry, id: generateLogId() }]);
  }

  function speak(text) {
    if (!("speechSynthesis" in window)) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  async function sendMessage(message) {
    const response = await fetch(`${API_BASE}/api/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        user_id: DEFAULT_USER,
        text: message,
        image_ids: capturedImages,
      }),
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || "Unexpected server error");
    }
    return response.json();
  }

  async function handleCapture() {
    if (!videoRef.current) {
      return;
    }
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");
    setImagePreview(dataUrl);
    setStatus("Uploading snapshot...");

    const response = await fetch(`${API_BASE}/api/upload-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        image_base64: dataUrl,
      }),
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      setStatus(errorBody.detail || "Snapshot upload failed");
      return;
    }
    const data = await response.json();
    setCapturedImages((prev) => [...prev, data.image_id]);
    setStatus("Snapshot attached for next message.");
  }

  function handleStartVoice() {
    if (!speechSupported || !recognitionRef.current) {
      setStatus("Speech recognition unavailable in this browser.");
      return;
    }
    if (isListening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  }

  async function handleManualSubmit(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const text = formData.get("message");
    if (!text) return;
    appendLog({ role: "user", text });
    setStatus("Sending message...");
    event.target.reset();
    try {
      const result = await sendMessage(text);
      appendLog({ role: "assistant", text: result.reply });
      speak(result.reply);
      setStatus("Ready");
    } catch (error) {
      console.error(error);
      setStatus(`Failed to contact backend: ${error.message}`);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Patient Advocacy Assistant</h1>
        <p>
          Session: <code>{sessionId}</code>
        </p>
        <p className="status">{status}</p>
      </header>

      <section className="controls">
        <button onClick={handleStartVoice} disabled={!speechSupported}>
          {isListening ? "Stop Listening" : "Start Voice"}
        </button>
        <button onClick={handleCapture}>Capture Snapshot</button>
      </section>

      <section className="camera">
        <video ref={videoRef} autoPlay playsInline muted />
        {imagePreview && (
          <div className="preview">
            <span>Last snapshot</span>
            <img src={imagePreview} alt="Snapshot preview" />
          </div>
        )}
      </section>

      <section className="manual-entry">
        <form onSubmit={handleManualSubmit}>
          <input name="message" type="text" placeholder="Type a follow-up..." />
          <button type="submit">Send</button>
        </form>
      </section>

      <section className="transcript">
        <h2>Conversation</h2>
        <ul>
          {logs.map((log) => (
            <li key={log.id} className={log.role}>
              <strong>{log.role === "user" ? "You" : "Assistant"}:</strong> {log.text}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

