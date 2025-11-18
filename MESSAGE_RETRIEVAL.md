# Message and Image Retrieval Features

## Overview

Two new API endpoints have been added to capture and retrieve all voice-to-text messages from all conversations, plus list all saved images.

## New Endpoints

### 1. `/api/all-messages` (GET)

Retrieves all voice-to-text user messages from all conversations.

**Features:**
- Reads from transcript files in `data/transcripts/`
- Also retrieves from active sessions in memory
- Combines all messages into a single text
- Prints to console when called
- Returns JSON with all messages

**Response Format:**
```json
{
  "total_sessions": 5,
  "total_user_messages": 12,
  "all_user_messages": [
    {
      "session_id": "session_abc123",
      "message": "I have a headache",
      "timestamp": "[20251118T221813565799Z]",
      "source": "transcript_file"
    },
    ...
  ],
  "combined_text": "[session_abc123] I have a headache\n[session_xyz789] It's been 3 days\n..."
}
```

**Usage:**
```bash
# Via curl
curl http://localhost:8100/api/all-messages

# Via browser
http://localhost:8100/api/all-messages

# Via Python script
python get_all_messages.py
```

**Console Output:**
When called, it also prints to the server console:
```
================================================================================
ALL VOICE-TO-TEXT MESSAGES FROM ALL CONVERSATIONS
================================================================================
Total Sessions: 5
Total User Messages: 12
--------------------------------------------------------------------------------
Session: session_abc123
Message: I have a headache
Timestamp: [20251118T221813565799Z]
--------------------------------------------------------------------------------
...
================================================================================
```

---

### 2. `/api/all-images` (GET)

Lists all images saved in the images directory.

**Features:**
- Scans `data/images/` directory
- Extracts session ID from filenames
- Provides file size information
- Prints to console when called
- Returns JSON with all image metadata

**Response Format:**
```json
{
  "total_images": 8,
  "images": [
    {
      "session_id": "session_abc123",
      "filename": "session_abc123_20251118T221813565799Z.png",
      "file_path": "/path/to/data/images/session_abc123_20251118T221813565799Z.png",
      "timestamp": "20251118T221813565799Z",
      "size_bytes": 245760,
      "size_kb": 240.0
    },
    ...
  ]
}
```

**Usage:**
```bash
# Via curl
curl http://localhost:8100/api/all-images

# Via browser
http://localhost:8100/api/all-images

# Via Python script
python get_all_messages.py
```

**Console Output:**
When called, it also prints to the server console:
```
================================================================================
ALL IMAGES SAVED IN DIRECTORY
================================================================================
Total Images: 8
--------------------------------------------------------------------------------
Session: session_abc123
Filename: session_abc123_20251118T221813565799Z.png
Path: /path/to/data/images/session_abc123_20251118T221813565799Z.png
Size: 240.0 KB
--------------------------------------------------------------------------------
...
================================================================================
```

---

## Python Script

A helper script `get_all_messages.py` is provided to easily retrieve and save all data:

```bash
# Make sure backend is running first
cd /home/sangeethagsk/agent_bootcamp/google_adk/patientadvocacy
python get_all_messages.py
```

This will:
1. Fetch all messages
2. Fetch all images
3. Print results to console
4. Save JSON files: `all_messages.json` and `all_images.json`

---

## How It Works

### Message Retrieval

1. **Reads Transcript Files**: Scans all `.txt` files in `data/transcripts/`
   - Extracts session ID from filename
   - Parses USER: lines to get voice-to-text messages
   - Includes timestamps

2. **Reads Active Sessions**: Also checks in-memory sessions
   - Uses the `created_sessions` set to track active sessions
   - Extracts user messages from session history
   - Filters out image-related metadata

3. **Combines and Returns**: 
   - Deduplicates by session
   - Combines all messages into single text
   - Prints to console
   - Returns JSON response

### Image Listing

1. **Scans Images Directory**: Reads all `.png` files in `data/images/`
2. **Extracts Metadata**: 
   - Session ID from filename
   - Timestamp from filename
   - File size
3. **Returns List**: Provides complete image inventory

---

## Notes

- **Images are only saved in directory**: As noted, images are stored in `data/images/` directory. The `/api/all-images` endpoint lists all of them.
- **Messages are from transcripts**: Voice-to-text messages are captured from transcript files that are saved each time a message is sent.
- **Active sessions**: Also checks currently active sessions in memory for real-time messages.
- **Console printing**: Both endpoints print formatted output to the server console when called, making it easy to see all data at a glance.

---

## Example Usage

```python
import requests

# Get all messages
response = requests.get("http://localhost:8100/api/all-messages")
data = response.json()

print(f"Found {data['total_user_messages']} messages from {data['total_sessions']} sessions")
print("\nCombined text:")
print(data['combined_text'])

# Get all images
response = requests.get("http://localhost:8100/api/all-images")
images = response.json()

print(f"\nFound {images['total_images']} images")
for img in images['images']:
    print(f"  - {img['filename']} ({img['size_kb']} KB)")
```

