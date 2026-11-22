# ICS 499 - YouTube Analyzer

## Authors
- Ashley Zenzola
- Chee Vang

## Project Overview

The YouTube Analyzer is a web application for analyzing a YouTube channel and exploring both channel statistics and video content.

The project is designed to:
- Collect and analyze information about a channel's videos.
- Produce useful statistics such as video counts, lengths, likes, comments, and posting trends.
- Retrieve and cache video transcripts when available.
- Build a Retrieval-Augmented Generation (RAG) system over indexed transcript content.
- Allow users to ask questions about information contained across a channel's videos.
- Generate a channel-wide summary using already-indexed transcript content.

The application uses the YouTube Data API for channel and video information, SQLite for channel/video metadata and raw-transcript caching, Chroma for persistent RAG transcript chunks and embeddings, Sentence Transformers for embeddings, and Gemini for generated answers.

## Target Users
- YouTube creators
- Users interested in analyzing a YouTube channel

## Features

### Channel Analytics
- Enter a supported YouTube channel ID or URL.
- Resolve supported input to the canonical `UC...` channel ID.
- Retrieve channel and video information.
- Calculate video and channel statistics.
- Analyze video-posting trends.
- Search, sort, paginate, and export video information.

### Transcript Management
- Cache channel/video data with SQLite.
- Retrieve and cache full/raw video transcripts with SQLite.
- Reuse saved transcripts instead of downloading them again.
- View cached transcripts from the web interface.

### RAG and AI
- Store transcript chunks and embeddings persistently with Chroma.
- Reuse already-indexed videos instead of reprocessing them.
- Keep RAG results separated by `channel_id`.
- Use Sentence Transformers for multilingual embeddings.
- Use Chroma for semantic retrieval.
- Use Gemini for generated answers.
- Show unique source video links for RAG answers.
- Generate a grounded channel-wide summary across indexed videos.
- Run RAG without live transcript requests by setting `RAG_TEST_MODE=true`.

### Interface
- Light/Dark Mode.
- Searchable and sortable recent-video table.
- Transcript and RAG status indicators.
- Copy and clear actions for supported UI fields.

## System Architecture

```text
YouTube Data API
       |
       v
FastAPI Backend
   |        \
   v         v
SQLite      Chroma
(metadata   (transcript chunks
and raw      and embeddings)
transcripts)      |
      \           /
       \         /
        v       v
        RAG Service
             |
             v
           Gemini
             |
             v
        Web Frontend
```

## Data Flow

### Analyze Channel

```text
Channel ID / URL
      |
      v
YouTube Data API
      |
      v
Channel/video metadata
      |
      v
SQLite cache + analytics UI
```

### Index RAG Content

```text
Index RAG Content
      |
      v
Check Chroma
      |
      +-- already indexed --> skip
      |
      v
Check SQLite transcript cache
      |
      +-- transcript found --> reuse
      |
      v
Retrieve transcript from YouTube when missing
      |
      v
Chunk transcript
      |
      v
Create embeddings
      |
      v
Store in Chroma
```

### Ask a Question

```text
User question
      |
      v
Search current channel's Chroma content
      |
      v
Retrieve relevant chunks
      |
      v
Gemini generation
      |
      v
Answer + source videos
```

### Generate Channel Summary

```text
Generate Channel Summary
      |
      v
Existing indexed Chroma content only
      |
      v
Representative chunks from indexed videos
      |
      v
Bounded hierarchical Gemini summarization
      |
      v
Channel-wide summary + indexed source videos
```

The Channel Summary does not retrieve new transcripts from YouTube.

## Technology Stack

### Application
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
- `youtube-transcript-api`
- pandas
- jQuery / DataTables
- Chart.js
- Visual Studio Code

### Main Python Dependencies
- `fastapi`
- `uvicorn[standard]`
- `google-api-python-client`
- `python-dotenv`
- `pandas`
- `youtube-transcript-api`
- `sentence-transformers`
- `chromadb`
- `google-genai`

## Quick Start

From the project root:

1. Create the virtual environment if it does not already exist.
2. Install dependencies.
3. Configure `.env`.
4. Start FastAPI.
5. Open the application in a browser.

### 1. Create the Virtual Environment

```powershell
py -m venv .venv
```

This does not need to be repeated if `.venv` already exists.

### 2. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

### 3. Configure Environment Variables

Example `.env` settings:

```text
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.8-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite
GEMINI_503_MAX_RETRIES=2
GEMINI_503_RETRY_BASE_SECONDS=1.0
RAG_TRANSCRIPT_VIDEO_LIMIT=1
RAG_TEST_MODE=false
```

Optional Chroma/RAG settings:

```text
CHROMA_PATH=data/chroma_db
CHROMA_COLLECTION_NAME=youtube_transcript_chunks
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small
RAG_CHUNKING_VERSION=v1
```

If you already have a local `.env`, update it directly. Local `.env` values override defaults shown in `.env.example` and in the Python code.

### 4. Run the Application

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## How to Use

1. Enter a supported YouTube channel ID or URL.
2. Analyze the channel to retrieve or reuse cached channel/video metadata.
3. Use **Index RAG Content** to process transcript content that is not already indexed.
4. Ask questions through the Channel RAG Chatbot.
5. Use **Generate Channel Summary** to summarize currently indexed videos.
6. View cached transcripts when available.

## Cached vs. Indexed Content

**Cached** and **indexed** refer to different stages of the application's data flow.

- **Cached:** Channel/video metadata or a full/raw transcript is stored locally in SQLite so it can be reused without retrieving it again.
- **Indexed:** Transcript content has been chunked, embedded, and stored in Chroma so it can be searched and used by RAG features.

A video can be cached without being indexed. RAG questions and Channel Summary depend on indexed Chroma content.

## RAG Indexing and Question Answering

`RAG_TRANSCRIPT_VIDEO_LIMIT` controls the per-run batch size for videos that are not yet indexed in Chroma.

- Cached videos are considered newest-first.
- Videos already indexed in Chroma are skipped.
- The next N unindexed videos are processed.
- A value of `3` indexes up to the next 3 missing videos each time **Index RAG Content** is used.
- A value of `0` processes all currently unindexed cached videos.

Repeated indexing therefore continues farther down the cached channel instead of repeatedly inspecting the same newest videos. Newly uploaded videos receive priority once they appear in the cache because candidates are ordered newest-first.

### Test Mode

`RAG_TEST_MODE=true`
- **Index RAG Content** uses `data/test_transcripts.json`.
- No live transcript requests are made.
- A separate Chroma test namespace is used.

`RAG_TEST_MODE=false`
- **Index RAG Content** checks Chroma first.
- For missing Chroma videos, the transcript service checks SQLite.
- YouTube transcript retrieval is attempted only during indexing when a transcript is not already saved in SQLite.
- **Ask Question** searches Chroma only and never retrieves transcripts.

### RAG Design Details

- SQLite stores normal channel/video data and full/raw transcripts.
- Chroma stores processed transcript chunks and embeddings used by RAG.
- Transcript indexing and question answering are separate operations.
- **Index RAG Content** is the only normal UI action that retrieves missing transcripts.
- **Ask Question** searches only the current channel's already-indexed Chroma content.
- Chroma is checked before transcript retrieval so already-indexed videos are not processed again.
- If Chroma does not contain a selected video during indexing, `transcript_service.py` checks SQLite before contacting YouTube.
- Date-aware questions such as "What is the most recent video about?" use publication metadata to select the newest indexed video rather than relying only on semantic similarity.
- Test transcript chunks use a separate Chroma namespace so they cannot contaminate live RAG results.
- Multiple retrieved chunks from one video may be used by Gemini, but each video is displayed once in the source list.

Chroma metadata can include `channel_id`, `channel_title`, `video_id`, title, publication date/timestamp, video duration, language, chunk timing, indexing time, transcript source/hash, embedding model, chunking version, and metadata version when available.

## Channel Summary

**Generate Channel Summary** uses only existing indexed Chroma content.

- It never retrieves new transcripts from YouTube.
- Representative transcript chunks are sampled from indexed videos.
- Larger channels use bounded hierarchical Gemini synthesis rather than placing the entire channel into one oversized prompt.
- Indexed source videos are preserved and displayed with the summary.

## Gemini Reliability and Fallback

The default Gemini model chain is:

```text
Gemini 3.8 Flash
      |
      v
Gemini 3.5 Flash-Lite
```

`GEMINI_FALLBACK_MODELS` is an ordered, comma-separated fallback chain.

### Temporary 503 / High-Demand Errors

For `503 UNAVAILABLE`, high-demand, or temporary provider-unavailable errors:

- The backend retries the same model using exponential backoff.
- `GEMINI_503_MAX_RETRIES=2` gives up to three total attempts: the initial request, then retries after approximately 1 second and 2 seconds by default.
- If the model is still unavailable, the backend temporarily uses the next configured model for the rest of the current Channel Summary or hierarchical comparison operation.
- A new summary starts by trying the preferred model again.

### 429 / Model-Access Errors

For `429` rate/quota failures or explicit model-access/not-found failures:

- The backend moves to the next configured model.
- The lower-priority model remains active for the rest of the current FastAPI process.
- Restarting FastAPI resets the preferred model to `GEMINI_MODEL`.

### Errors That Do Not Trigger Fallback

The system does not use another model to bypass:
- Authentication errors
- Malformed requests
- Empty responses
- Content-safety or policy failures

Set:

```text
GEMINI_FALLBACK_MODELS=
```

to disable automatic model fallback.

Set:

```text
GEMINI_503_MAX_RETRIES=0
```

to disable same-model 503 retries.

### AI Availability Limitation

Channel Summary depends on the external Gemini service. If all configured Gemini models remain unavailable after retries, the analyzer preserves the indexed transcript evidence and source list but reports that synthesis could not be completed.

Multi-provider LLM failover was considered but intentionally deferred to keep the capstone MVP smaller, easier to test, and free of additional provider/API dependencies.

## API Endpoints

### Channel Analytics

```text
GET /api/channel/{channel_id}/analytics
```

### Index RAG Content

```text
POST /api/rag/index
```

### Ask a Question

```text
POST /api/question
```

### RAG Storage Statistics

```text
GET /api/rag/stats
```

For one channel:

```text
GET /api/rag/stats?channel_id=UC_CHANNEL_ID
```

## Testing

### Automated Gemini Fallback Tests

Run from the project root:

```powershell
python -m unittest tests.test_llm_fallback -v
```

The tests use mocked Gemini responses and do not make live Gemini API calls.

They verify:
- Same-model 503 retry and recovery.
- Temporary per-summary 503 fallback.
- Returning to the primary model for a new summary.
- Sticky 429/model-access downgrade behavior.
- Invalid-request handling.
- Safety/policy handling.

### RAG Evaluation

Run:

```powershell
python rag_evaluation.py UC_CHANNEL_ID
```

The evaluation covers:
- English retrieval/generation.
- Spanish retrieval/generation.
- Telugu retrieval/generation.
- An additional arbitrary target language such as Japanese.
- Explicit cross-language output.
- Retrieval returning source chunks.
- Newest-video source isolation.
- Specific-video source isolation.
- Two-video comparison source coverage.

`rag_evaluation.py` prints explicit PASS/FAIL checks for retrieval and scope behavior. Language fluency and semantic correctness still require human review.

### Manual Verification

The project has also been manually checked for the following behavior:
- SQLite transcript reuse prints `Using cached transcript from SQLite: VIDEO_ID` instead of downloading the same transcript again.
- Chroma duplicate-index logic skips videos that are already stored.
- Chroma retrieval is filtered by `channel_id`.
- Multiple retrieved chunks from one video are deduplicated into one source-video entry.
- RAG can be tested independently with `RAG_TEST_MODE=true`.
- `/api/rag/stats` can be used to inspect stored Chroma chunks and videos.

## Standalone Utilities

These utilities allow the transcript, RAG-building, and evaluation stages to be run independently of the web interface.

### Transcript Utility

```powershell
python youtube_transcriptor.py UC_CHANNEL_ID --limit 0
```

This writes or reuses raw transcripts in SQLite. `--limit 0` means all available videos.

### RAG Builder

```powershell
python rag_builder.py UC_CHANNEL_ID --limit 0
```

This reads SQLite transcripts and builds Chroma without requesting transcripts from YouTube.

To rebuild after changing the embedding model:

```powershell
python rag_builder.py UC_CHANNEL_ID --rebuild
```

### Multilingual / RAG Evaluation

```powershell
python rag_evaluation.py UC_CHANNEL_ID
```

This runs the RAG evaluation checks described in the Testing section.

## Project Structure

```text
backend/
  main.py
  youtube_service.py
  database_service.py
  analytics_service.py
  transcript_service.py
  rag_service.py
  llm_service.py
  requirements.txt

frontend/
  index.html
  style.css
  script.js

tests/
  test_llm_fallback.py

data/
  youtube_analyzer.db
  chroma_db/
  test_transcripts.json

youtube_transcriptor.py
rag_builder.py
rag_evaluation.py
```

### Main Components

- `backend/main.py` - FastAPI routes and overall application flow.
- `backend/youtube_service.py` - YouTube Data API communication and channel-input resolution.
- `backend/database_service.py` - SQLite caching for channels, videos, and full/raw transcripts.
- `backend/analytics_service.py` - Statistics calculations with pandas.
- `backend/transcript_service.py` - Reuses SQLite transcripts before requesting missing transcripts from YouTube.
- `backend/rag_service.py` - Chroma indexing, retrieval, RAG prompt preparation, and source handling.
- `backend/llm_service.py` - Gemini generation, retry, and fallback behavior.
- `frontend/index.html` - Main page structure.
- `frontend/style.css` - Layout, appearance, dark mode, and DataTables styling.
- `frontend/script.js` - Frontend behavior and FastAPI communication.
- `data/youtube_analyzer.db` - SQLite cache created by the application.
- `data/chroma_db/` - Persistent Chroma storage created by the application.
- `data/test_transcripts.json` - Local transcript data used by RAG test mode.

## Data Storage and Security Notes

- API keys are configured through `.env` and should not be committed to source control.
- Channel/video metadata and raw transcripts are cached locally in SQLite.
- RAG transcript chunks and embeddings are stored locally in Chroma.
- Retrieved transcript context is sent to Gemini when generating an AI response or channel summary.
- The current application analyzes one channel at a time and does not include user accounts.

## Known Limitations

| Tool / Service | Limitation | Impact on This Project | Current Mitigation |
| --- | --- | --- | --- |
| `youtube-transcript-api` | YouTube may block transcript requests after frequent requests or depending on the network/IP. | Large channels may stop processing before every transcript is retrieved. | Save every successful transcript to SQLite, reuse cached transcripts, and avoid unnecessary repeated requests. |
| YouTube Data API v3 | The API has a default daily quota of 10,000 units, and additional pages require additional requests. | Large channels, repeated analysis, or development/testing can consume the daily quota and prevent metadata retrieval. | Cache channel/video metadata in SQLite and reuse previously retrieved data. |
| Gemini API free tier | Gemini requires a valid API key and can be temporarily unavailable because of provider demand or rate/token limits. | Chatbot generation may fail even when Chroma retrieves the correct chunks. Large channel summaries may require several bounded Gemini calls. | Keep retrieval/indexing local, use bounded hierarchical batching, retry temporary failures, use the configured fallback model, and preserve indexed evidence when generation is unavailable. |
| `intfloat/multilingual-e5-small` / Sentence Transformers | CPU, RAM, and disk performance can affect embedding/indexing speed. Retrieval quality may vary across languages, and long transcripts must be chunked. | Initial setup and indexing can be slower on lower-performance computers. Multilingual retrieval quality is not identical for every language. | Chunk transcripts before embedding, persist embeddings in Chroma, track the embedding model in metadata, and rebuild from cached SQLite transcripts when the embedding model changes. |
| Chroma | Practical capacity depends on local disk space, memory, and processing performance. | Large transcript collections can increase storage use and retrieval/indexing time. | Reuse persistent collections, prevent duplicate indexing, store metadata, and use persistent storage when deployed. |
| SQLite | SQLite has limited concurrent-write scalability compared with server databases such as PostgreSQL. | It may become a bottleneck if many users write channel/transcript data concurrently in a production deployment. | Use SQLite for local caching and preprocessing. A production version could replace the database service with PostgreSQL without changing the overall RAG architecture. |

## Future Implementations
Persistent Chat History - Storing chatbot conversations in SQLite and tying them to channel_id so users can resume conversation after refreshing the page or a restart
