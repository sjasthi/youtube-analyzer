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
- RAG implementation
    - Gemini API generates answers
    - Sentence Transformers create embeddings
    - FAISS was initially used for semantic retrieval
    - Added `RAG_TEST_MODE` to test without live transcript requests
- Added transcript retrieval
- Saved retrieved transcripts in SQLite for reuse
- Added source video links to answers

- Added analytics-aware chatbot queries
    - Questions such as `What is the most liked video?`, `Which video has the most views?`, and `Which video has the most comments?` are answered directly from cached SQLite video analytics rather than transcript text
    - Supports most/highest/top and least/fewest/lowest ranking forms, including requests such as `Top 5 most liked videos`
    - The selected analytics result becomes the active in-memory video scope, so a follow-up such as `What is it about?` can switch back to transcript RAG for that video
    - Analytics questions do not require the selected video to be indexed in Chroma; only transcript-content follow-ups require RAG indexing
- Removed the separate **Latest Sources** panel from the chatbot
    - Each assistant message keeps its own source video links/timestamps directly under that answer
    - This avoids duplicated or stale source information during follow-up questions and multi-video comparisons
- Filtered out less relevant transcript chunks

<b>FP6</b>
- Replaced FAISS with Chroma for persistent RAG vector storage
- Chroma stores transcript chunks, embeddings, and source metadata in `data/chroma_db/`
- SQLite still stores the full/raw transcript so the original transcript can be reused or reprocessed later
- RAG lookup order now uses:
    1. Chroma for already-indexed transcript chunks
    2. SQLite for saved raw transcripts
    3. YouTube transcript retrieval only when the video is not already stored
- Previously indexed videos are skipped instead of being re-chunked and re-embedded
- Chroma retrieval is filtered by canonical `channel_id` so one channel's chunks do not mix with another channel
- Test transcript data is stored under a separate Chroma namespace so it does not mix with live channel data
- Source video links are deduplicated so several retrieved chunks from the same video show one source video
- Added `/api/rag/stats` to inspect persistent Chroma storage
- Added safer live transcript batch retrieval
    - SQLite and Chroma cache hits remain immediate
    - New YouTube transcript requests are paced with a configurable delay and small random jitter
    - Default delay is 8 seconds plus up to 2 seconds of jitter between live YouTube requests
- Added stop-on-block behavior
    - `RequestBlocked` and `IpBlocked` stop the current transcript batch immediately
    - Successfully retrieved transcripts are saved to SQLite before the batch stops
    - Already-indexed Chroma videos remain available for RAG
- Added transcript status information to the Q&A UI
- Separated RAG indexing from RAG question answering
    - Added an explicit **Index RAG Content** button
    - Indexing is the only normal UI action that retrieves missing transcripts
    - Asking a question searches existing Chroma content only and never retries YouTube
- Missing/blocked videos are not automatically retried on every question
- Added RAG coverage information after channel analysis so users can see whether the channel already has indexed content
- Added `POST /api/rag/index` for explicit transcript retrieval + Chroma indexing
- Added richer RAG source metadata
    - Chroma chunks now support `published_at`, `language`, `start_time`, and `end_time`
    - New transcript retrievals preserve YouTube caption language and timing segments in SQLite
- Added timestamp-aware RAG sources
    - Source links jump to the approximate transcript chunk timestamp when timing data is available
    - The Q&A source list shows timestamp, language, publication date, and video duration when available
- Improved native-language transcript handling
    - Uses YouTube default audio/language metadata when available
    - Does not intentionally translate captions before indexing
- Improved "most recent/latest/newest video" questions
    - Uses video `published_at` metadata instead of semantic similarity alone
    - Restricts retrieval to the actual newest indexed video
    - If the newest video is not indexed, the chatbot explains that instead of silently answering from an older video
- Legacy metadata that cannot be reconstructed reliably is labeled `legacy_unknown` rather than guessed
- `/api/rag/stats` now reports stored embedding models, chunking versions, transcript sources, and metadata versions
- Source details can now also display video duration
- Added `RAG_CHUNKING_VERSION` to `.env.example` so future chunking-strategy changes can be tracked
- Added text resizing
- Added collapsible sections
- Added a Back to Top button
- Added clickable links in the Video Sample Analytics section
- Matched video link colors to the links in the Recent Videos section

<b>FP7</b>
- Thumbnails to 'Recent Videos' and 'Video Sample Analytics' sections
- Added multilingual RAG support
    - Default local embedding model changed to `intfloat/multilingual-e5-small` for cross-language semantic retrieval
    - Auto response mode answers in the same language as the user's question unless another language is explicitly requested
    - Leave the Answer language field blank for Auto, or type a language such as French, Japanese, Arabic, Hindi, Korean, or Portuguese (Brazil)
    - The Q&A UI accepts any target language name as free text; leave it blank for Auto mode
    - English questions can retrieve non-English transcript content and vice versa when the multilingual embeddings find relevant chunks
- Added specific-video question handling
    - Questions that explicitly contain a full video title or a quoted partial title are mapped to that video's `video_id`
    - Chroma retrieval is then restricted to that video only
    - If the requested video is not indexed, the chatbot says so instead of answering from another video
- Added a cache-only transcript viewer
    - Lists transcripts already saved in SQLite
    - Displays native/saved language metadata, publication date, duration, and timestamped segments when available
    - Legacy plain-text transcripts remain viewable without invented timestamps
    - Opening the transcript viewer never triggers a YouTube transcript request
- Added standalone utility: `youtube_transcriptor.py`
    - Processes channel videos outside the web UI
    - Retrieves original/native caption tracks and saves them to SQLite
    - Reuses cached transcripts and preserves successful work if YouTube blocks later requests
- Added standalone utility: `rag_builder.py`
    - Reads saved SQLite transcripts only
    - Chunks, embeds, and writes them to persistent Chroma
    - Supports `--rebuild` after embedding/chunking changes
- Added `rag_evaluation.py`
    - Exercises English, Spanish, Telugu, arbitrary target-language output (including Japanese), cross-language response, and single-video retrieval cases
    - Does not retrieve new YouTube transcripts

<b>FP8</b>
- Make chatbot/RAG smarter
    - a Popup chat box
    - Keep queried questions until session is terminated
    - Removed latest source as it was redundant
    - Follow-up questions can remember previous video(s)
    - Changed `clear chat` to `new chat`
    - RAG now recognize basic questions about analytics
    - Can decide when to use metadata or transcript content
    - Able to compare videos
    - Removed language seletion field
- Recent videos list
    - added status of transcript and RAG
    - Lets users know if video is short or regular
    - Can index a specific video by clicking on white index button




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
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
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

| Tool / Service | limitation | Impact on this project | Current mitigation |
| --- | --- | --- | --- |
| `youtube-transcript-api` | may still be blocked by YouTube after frequent requests or depending on the network/IP. | A large channel may stop processing before every transcript is retrieved. | Save every successful transcript to SQLite. Add a delay between request. RAG building is from saved transcripts.|
| YouTube Data API v3 | default daily quota cost(10,000/day). Each page require additional requests | Large channels, repeat analysis, or repeated development/testing can use a lot of daily quota and prevent channel/video metadata retrieval | Cache channel and video metadata in SQLite. Reuse previously retrieved data, avoid unneccessary repeated request and process what is needed during testing |
| Gemini API free tier | Gemini requires a valid API key and can occasionally be unavailable because of provider demand. | Chatbot may be unable to generate answers even when chroma retrieves the correct chunks. Large retrieved context uses more tokens | Keeps retrieveal and generation separate and preserve a retreval/fallback path so indexed content is not lost when Gemini is unavaliable |
| `intfloat/Multilingual-e5-small`/ Sentence Transformers | Low-performance PC may struggle because of CPU/RAM/Disk usage. Has 100 languages, but retrieval quality can vary for lower-resource language. Model has a finite input length, so long transcripts need to be chunked. | Inital setup can be slower. CPU only computer and multilingual retrieval quality is not identical for evey supported language | Chunk transcripts before embedding. Persist embeddings in Chroma so they're not recreated every question. track the embedding model in metadata and rebuild from cached SQLite transcripts when the embedding model changes |
| Chroma | Only limits are computer's disk pace, memory and processing speed | A large transcript collection can increase local storage and retrieval/indexing time. | Reuse persistent local collections, avoid duplicate indexing, store metadata, and use persistent storage if the app is deployed |
| SQLite | limited concurrent-write scalability compared with PostgreSQL or other server databases | may become a bottleneck if many users attempted to write transcripts/channel data at the same time in a production deployment | Keeps SQLite for local caching and preprocessing. A production version could replace the database service with PostgreSQL without changing the RAG architecture |


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


### Transcript utility

```powershell
python youtube_transcriptor.py UC_CHANNEL_ID --limit 0
```

This writes/reuses raw transcripts in SQLite. `--limit 0` means all available videos.

### RAG builder

```powershell
python rag_builder.py UC_CHANNEL_ID --limit 0
```

This reads SQLite transcripts and builds Chroma without requesting transcripts from YouTube. To rebuild after changing the embedding model:

```powershell
python rag_builder.py UC_CHANNEL_ID --rebuild
```

### Multilingual / RAG evaluation

Run it from the project root:

```powershell
python rag_evaluation.py UC_CHANNEL_ID
```

The evaluation covers English, Spanish, Telugu, an additional arbitrary target language (Japanese), explicit cross-language output, and a specific-video retrieval isolation check.
`rag_evaluation.py` now prints explicit PASS/FAIL checks for:

- multilingual retrieval/generation cases
- retrieval returning source chunks
- newest-video source isolation
- specific-video source isolation
- two-video comparison source coverage

Language fluency and semantic correctness still require human review; the automated harness verifies retrieval/scope behavior and whether an answer was produced.
