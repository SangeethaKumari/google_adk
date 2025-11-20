# End Conversation Feature

## Overview

Added an "End Conversation" button that retrieves and displays all messages and images from the current session when clicked.

---

## What Was Added

### 1. **Backend Endpoint: `/api/session/{session_id}/summary`** (GET)

**Location**: `src/patientadvocacy/app.py` (line 440)

**What it does**:
- Retrieves all messages (user + assistant) from transcript files for the session
- Retrieves all images captured in that session
- Combines all messages into a single text
- Returns images as base64-encoded data
- Prints summary to console

**Response Format**:
```json
{
  "session_id": "session_abc123",
  "total_messages": 6,
  "all_messages": [
    {
      "role": "user",
      "message": "I have a headache",
      "timestamp": "[20251118T221813565799Z]",
      "source": "transcript_file"
    },
    {
      "role": "assistant",
      "message": "I understand you've been experiencing headaches...",
      "timestamp": "[20251118T221813565799Z]",
      "source": "transcript_file"
    }
  ],
  "combined_text": "[20251118T221813565799Z] USER: I have a headache\n[20251118T221813565799Z] ASSISTANT: I understand...",
  "total_images": 2,
  "images": [
    {
      "filename": "session_abc123_20251118T221813565799Z.png",
      "file_path": "/path/to/image.png",
      "timestamp": "20251118T221813565799Z",
      "size_bytes": 245760,
      "size_kb": 240.0,
      "base64": "data:image/png;base64,iVBORw0KGgo..."
    }
  ]
}
```

---

### 2. **Frontend: "End Conversation" Button**

**Location**: `ui/src/App.jsx`

**What was added**:
- New red "End Conversation" button in the controls section
- `handleEndConversation()` function that:
  - Calls `/api/session/{sessionId}/summary`
  - Displays the summary in a formatted section
  - Automatically downloads the combined text as a `.txt` file
  - Shows all captured images in a grid

**Button Location**: In the controls section, next to "Start Voice" and "Capture Snapshot"

---

## How It Works

### User Flow:

1. **User has a conversation** with the assistant
2. **User captures images** (optional)
3. **User clicks "End Conversation"** button
4. **Frontend calls** `GET /api/session/{sessionId}/summary`
5. **Backend**:
   - Reads all transcript files for that session
   - Reads all image files for that session
   - Combines messages into text
   - Encodes images as base64
   - Returns JSON response
6. **Frontend**:
   - Displays summary in a formatted section
   - Shows all messages in a scrollable text area
   - Displays all images in a grid
   - Automatically downloads `conversation_{sessionId}.txt` file

---

## Features

### ✅ What It Does:

1. **Retrieves All Messages**:
   - User messages (voice-to-text)
   - Assistant responses
   - From transcript files
   - From active session (if available)

2. **Retrieves All Images**:
   - All images captured in that session
   - Includes metadata (filename, size, timestamp)
   - Base64-encoded for easy display

3. **Combines Text**:
   - All messages formatted with timestamps
   - Ready for download/export

4. **Automatic Download**:
   - Downloads combined text as `.txt` file
   - Filename: `conversation_{sessionId}.txt`

5. **Visual Display**:
   - Summary section appears below conversation
   - Shows message count, image count
   - Displays combined text in scrollable area
   - Shows images in a grid layout

---

## Usage

### For Users:

1. Have a conversation with the assistant
2. Capture images if needed (optional)
3. Click the red **"End Conversation"** button
4. View the summary that appears
5. The text file downloads automatically
6. View all images in the summary section
7. Click "Close Summary" to hide it

### For Developers:

**API Call**:
```bash
curl http://localhost:8100/api/session/{session_id}/summary
```

**Frontend Function**:
```javascript
async function handleEndConversation() {
  const response = await fetch(`${API_BASE}/api/session/${sessionId}/summary`);
  const summary = await response.json();
  // Display summary, download file, etc.
}
```

---

## UI Components

### Button:
- Red background (`#dc3545`)
- White text
- Located in controls section
- Label: "End Conversation"

### Summary Section:
- Blue border (`#007bff`)
- Appears below conversation log
- Contains:
  - Session ID
  - Total messages count
  - Total images count
  - Combined text (scrollable)
  - Image grid (responsive)
  - Close button

---

## Example Output

When "End Conversation" is clicked:

**Console (Backend)**:
```
================================================================================
SESSION SUMMARY: session_abc123
================================================================================
Total Messages: 6
Total Images: 2
--------------------------------------------------------------------------------
Combined Text:
[20251118T221813565799Z] USER: I have a headache
[20251118T221813565799Z] ASSISTANT: I understand you've been experiencing headaches...
[20251118T221850569633Z] USER: It's been 3 days
[20251118T221850569633Z] ASSISTANT: Three days is significant...
--------------------------------------------------------------------------------
Images:
  - session_abc123_20251118T221813565799Z.png (240.0 KB)
  - session_abc123_20251118T221850569633Z.png (185.5 KB)
================================================================================
```

**Frontend**:
- Summary section appears
- Text file downloads: `conversation_session_abc123.txt`
- Images displayed in grid
- All messages shown in scrollable text area

---

## Technical Details

### Backend Implementation:

- **Endpoint**: `GET /api/session/{session_id}/summary`
- **Response Model**: `SessionSummaryResponse`
- **Data Sources**:
  - Transcript files: `data/transcripts/{session_id}_*.txt`
  - Image files: `data/images/{session_id}_*.png`
  - Active sessions: In-memory session service

### Frontend Implementation:

- **State Variables**:
  - `sessionSummary`: Stores the summary data
  - `showSummary`: Controls visibility of summary section

- **Functions**:
  - `handleEndConversation()`: Main handler function

- **Automatic Download**:
  - Creates Blob from combined text
  - Creates download link
  - Triggers download
  - Cleans up URL

---

## Notes

- **Images are base64-encoded** for easy display in the browser
- **Text file downloads automatically** when summary is generated
- **Summary can be closed** using the "Close Summary" button
- **All data comes from files** (transcripts and images), so it works even if session is no longer in memory
- **Console output** helps with debugging and monitoring

---

## Future Enhancements (Optional)

- Add option to download images as ZIP
- Add option to email summary
- Add option to print summary
- Add option to export as PDF
- Add option to share summary link
- Add option to save summary to database

