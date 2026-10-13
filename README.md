# ICS 499 - YouTube Analyzer

## What's new this version

<b>FP 3</b>
- Add analytics
- Use UC id to look up creators (Soon to be updated to use URLs and @handle)
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
- Q&A section for asking questions about video content (RAG not yet implemented)
- Example questions in the Q&A section

<b>FP4</b>
- added pagination, search, sort
- remove video limit and retrieve all channel videos
- removed video count selector
- Add database (SQLite) so it doesn't have to download the data again and just access previous called channels
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

## Target Users
- YouTube creators
- Users interested in a YouTube channel

## Features
- Enter a YouTube channel ID or URL
- Retrieve channel and video information
- Calculate statistics such as video length and likes
- Analyze video-posting trends
- Process video transcripts
- Allow users to ask questions about channel content
- To test RAG: go to .env and change `RAG_TEST_MODE`=true 

## Scope
- Analyze one channel at a time
- No user accounts
- Answers are based on available channel content

## Technologies Used (Subject to Change)
- Python
- FastAPI
- HTML
- CSS
- JavaScript
- SQLite
- RAG/AI libraries
- Visual Studio Code

## External Libraries Used
- fastapi
- uvicorn[standard]
- google-api-python-client
- python-dotenv
- panda

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
GET /api/channel/{channel_id}/analytics?limit=10
```


## Expected Input
- YouTube channel ID or URL
- User questions (Testing phase only, blocked by Youtube-transcript-api for frequent request)

## Expected Output
- Channel statistics
- Video statistics
- Posting trends
- Answers to user questions
- hyperlink videos

## Important Design Decisions
- 

## Known Limitations
- 

## Testing Evidence
- 

## What each file does

- `backend/main.py` - FastAPI routes and application flow
- `backend/youtube_service.py` - YouTube Data API communication
- `backend/database_services.py` - Handles SQLite cache
- `backend/analytics_service.py` - Statistics calculations with pandas
- `backend/transcript_service.py` - Uses youtube-transcript-api to retreive transcript for a video
- `backend/rag_service.py` - Takes transcripts, breaks them into chunks, creates embeddings with Sentence Transformers, builds the FAISS index, embeds the user's question, retrieves the most relevant transcript chunks, constructs the RAG prompt, and prepares sources for the answer.
- `backend/llm_service.py` - Handles Gemini generation
- `backend/requirements.txt` - Lists the Python libraries that need to be installed
- `frontend/index.html` - page structure
- `frontend/style.css` - appearance/layout
- `frontend/script.js` - calls FastAPI and updates the page
