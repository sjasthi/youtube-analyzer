# ICS 499 - YouTube Analyzer

## What's new this version

<b>FP 3</b>
- Add analytics
- Use UC id to look up creators
- Able to pull at most 50 videos at once
- Shows creator total subscribers, total videos created, views, and posting trends
- Average views
- Average likes
- Average comments
- Average video duration
- Total views across the retrieved sample
- Engagement rate
- Average days between uploads
- Estimated uploads per month
- Most viewed video in the sample
- Most liked video in the sample
- Upload count by month
- Q&A section for asking questions about video content
- Example questions in the Q&A section

<b>FP4</b>
- Added pagination, search, and sort
- Removed video limit and retrieve all channel videos
- Removed video count selector
- Added database (SQLite) so channel/video data can be cached and reused
- Input supports channel IDs and YouTube URLs
- Added `youtube-transcript-api` for transcript retrieval testing

<b>FP5</b>
- The Recent Videos table now supports:
    - Clickable video titles that open the original YouTube video in a new tab
    - Copy
    - CSV export
    - Excel (.xlsx) export
    - PDF export
    - Print
- Exports include the currently searched/filtered rows across all DataTables pages and include a separate Video URL column.
- Dark Mode
    - The frontend includes a light/dark theme toggle in the top-right corner.
    - The selected theme is saved in browser `localStorage`.
    - Cards, forms, the Q&A area, DataTables controls, export buttons, and the video table follow the selected theme.
    - The Chart.js posting-trend chart updates its text and grid colors immediately when the theme changes.
- RAG implementation using a test file instead of actual transcripts (Youtube-transcript-api blocked me for too many request)
    - use Gemini API: model gemini-3.8-flash
    - FAISS implemented (find transcript chunks whose embeddings are most similar to the user's question embedding)
    - At most only 3 videos (for testing): limit can be removed at later date
- Add a test mode for RAG (to not hit transcript block)
- Added transcript retrieval
- Saved retrieved transcripts in the SQLite database for reuse
- Added source video links to answers
- Filtered out less relevant transcript chunks

## Authors
- Ashley Zenzola
- Chee Vang

## Objective
Build a website for analyzing a YouTube channel.

1. Collect and analyze information about the channel's videos.

2. Produce useful reports and statistics, such as:
    - Number of videos
    - Video lengths
    - Likes
    - Comments
    - Video-posting trends
    - Other meaningful metrics

3. Convert video/audio content to text when appropriate.

4. Build a RAG-based question-answering system over the channel's content.

5. Allow users to ask questions about information contained across the channel's videos.

## Project Overview
This is a web application that provides statistics, trends, and content analysis for a YouTube channel.

The application uses the YouTube Data API for channel/video information, SQLite for channel/video and raw-transcript caching, Chroma for persistent RAG-ready transcript chunks and embeddings, and Gemini for generated answers.

## Target Users
- YouTube creators
- Users interested in a YouTube channel

## Features
- Enter a supported YouTube channel ID or URL
- Resolve supported input to the canonical `UC...` channel ID
- Retrieve channel and video information
- Calculate video/channel statistics
- Analyze video-posting trends
- Search, sort, paginate, and export video information
- Light/Dark Mode
- Cache channel/video data with SQLite
- Retrieve and cache video transcripts with SQLite
- Store transcript chunks and embeddings persistently with Chroma
- Reuse already-indexed videos instead of reprocessing them
- Keep RAG results separated by `channel_id`
- Use Sentence Transformers for embeddings
- Use Chroma for semantic retrieval
- Use Gemini for generated answers
- Show unique source video links for RAG answers
- To test RAG without live transcript requests, set `RAG_TEST_MODE=true` in `.env`

## Scope
- Analyze one channel at a time
- No user accounts
- Answers are based on available channel content
- RAG processes the configured number of recent videos during indexing/testing
- Previously stored raw transcripts and Chroma chunks can be reused across FastAPI restarts

## Technologies Used (Subject to Change)
- Python
- FastAPI
- HTML
- CSS
- JavaScript
- SQLite
- Chroma
- Sentence Transformers
- Gemini API
- YouTube Data API
- youtube-transcript-api
- pandas
- jQuery / DataTables
- Chart.js
- Visual Studio Code

## External Libraries Used
- fastapi
- uvicorn[standard]
- google-api-python-client
- python-dotenv
- pandas
- youtube-transcript-api
- sentence-transformers
- chromadb
- google-genai

## Create the virtual environment to run
- This doesn't need to be run again if already done

```powershell
py -m venv .venv
```

## How to Install Dependencies
- Keep your existing `.env` and `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Environment Variables

Example `.env` settings:

```text
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
RAG_TRANSCRIPT_VIDEO_LIMIT=1
RAG_TEST_MODE=false
```

`RAG_TRANSCRIPT_VIDEO_LIMIT=0` means use all available videos. Any positive value uses only the newest N videos.

Optional Chroma/RAG settings:

```text
CHROMA_PATH=data/chroma_db
CHROMA_COLLECTION_NAME=youtube_transcript_chunks
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CHUNKING_VERSION=v1
```

`RAG_TEST_MODE=true`
- The **Index RAG Content** action uses `data/test_transcripts.json`
- Does not make live transcript requests
- Uses a separate Chroma test namespace

`RAG_TEST_MODE=false`
- **Index RAG Content** checks Chroma first
- For missing Chroma videos, the transcript service checks SQLite
- YouTube transcript retrieval is only attempted during indexing when the transcript is not saved in SQLite
- **Ask Question** searches Chroma only and never retrieves transcripts

## How to Run the Program

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

FastAPI docs:

```text
http://127.0.0.1:8000/docs
```

The main analytics endpoint is:

```text
GET /api/channel/{channel_id}/analytics
```

Index/retrieve missing transcript content for RAG:

```text
POST /api/rag/index
```

Ask a question using only already-indexed Chroma content:

```text
POST /api/question
```

RAG storage statistics:

```text
GET /api/rag/stats
```

For one channel:

```text
GET /api/rag/stats?channel_id=UC_CHANNEL_ID
```

## Expected Input
- YouTube channel ID or supported URL
- User questions about the analyzed channel
- Test transcript data when `RAG_TEST_MODE=true`

## Expected Output
- Channel statistics
- Video statistics
- Posting trends
- Hyperlinked videos
- RAG-generated answers
- Unique source video links

## Important Design Decisions
- SQLite stores normal channel/video data and full/raw transcripts.
- Chroma stores processed transcript chunks and embeddings used by RAG.
- Transcript indexing and question answering are separate operations.
- **Index RAG Content** is the only normal UI action that retrieves missing transcripts.
- **Ask Question** searches only the current channel's already-indexed Chroma content.
- Chroma is checked before transcript retrieval so an already-indexed video is not processed again.
- If Chroma does not contain a selected video during indexing, `transcript_service.py` checks SQLite before contacting YouTube.
- Chroma metadata includes `channel_id`, `channel_title`, `video_id`, title, publication date/timestamp, video duration, language, chunk timing, indexing time, transcript source/hash, embedding model, chunking version, and metadata version when available.
- Date-aware questions such as "What is the most recent video about?" use publication metadata to select the newest indexed video instead of relying only on semantic similarity.
- Test transcript chunks use a separate Chroma namespace so they cannot contaminate live RAG results.
- Multiple retrieved chunks from one video may still be used by Gemini, but the video is displayed once in the source list.

## Known Limitations
- `youtube-transcript-api` may still be blocked by YouTube after frequent requests or depending on the network/IP. The application now stops the batch and displays a UI warning instead of continuing live requests after a known block.
- Videos without an available transcript cannot currently be indexed through the normal transcript path.
- RAG currently considers only the configured number of recent videos for new indexing.
- Chroma data is local to `data/chroma_db/`; deployed persistent reuse requires persistent storage.
- Gemini requires a valid API key and can occasionally be unavailable because of provider demand.

## Testing Evidence
- Confirmed SQLite transcript reuse prints `Using cached transcript from SQLite: VIDEO_ID` instead of downloading the same transcript again.
- Chroma duplicate-index logic skips a video that is already stored.
- Chroma retrieval is filtered by `channel_id`.
- Multiple retrieved chunks from one video are deduplicated into one source-video entry.
- RAG can still be tested independently with `RAG_TEST_MODE=true`.
- `/api/rag/stats` can be used to inspect stored Chroma chunks/videos.

## What each file does

- `backend/main.py` - FastAPI routes and overall application flow; coordinates SQLite transcript caching, Chroma RAG storage, and Gemini generation
- `backend/youtube_service.py` - YouTube Data API communication and supported channel-input resolution
- `backend/database_service.py` - Handles SQLite caching for channels, videos, and full/raw transcripts
- `backend/analytics_service.py` - Statistics calculations with pandas
- `backend/transcript_service.py` - Checks SQLite for a saved transcript, then uses `youtube-transcript-api` only when needed
- `backend/rag_service.py` - Chunks transcripts, preserves source/date/language/timestamp metadata, creates Sentence Transformer embeddings, stores/reuses them in Chroma, retrieves channel/video-filtered chunks, builds the RAG prompt, and prepares unique timestamp-aware sources
- `backend/llm_service.py` - Handles Gemini answer generation
- `backend/requirements.txt` - Lists Python dependencies
- `frontend/index.html` - Page structure
- `frontend/style.css` - Appearance/layout, dark mode, and DataTables styling
- `frontend/script.js` - Calls FastAPI, updates the page, controls DataTables, explicitly starts RAG indexing, and sends Chroma-only RAG questions
- `data/youtube_analyzer.db` - SQLite cache created by the application
- `data/chroma_db/` - Persistent Chroma storage created by the application
- `data/test_transcripts.json` - Local transcript data used by RAG test mode
