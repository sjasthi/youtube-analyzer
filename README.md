# ICS 499 - YouTube Analyzer

## What's new this version

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
- PostgreSQL
- RAG/AI libraries
- Visual Studio Code

## External Libraries Used
- fastapi
- uvicorn[standard]
- google-api-python-client
- python-dotenv
- panda

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
- YouTube channel ID or URL (Only the UC id at the moment)
- User questions (Not Implemented Yet)

## Expected Output
- Channel statistics
- Video statistics
- Posting trends
- Answers to user questions

## Important Design Decisions
- 

## Known Limitations
- 

## Testing Evidence
-

## What each file does

- `backend/main.py` - FastAPI routes and application flow
- `backend/youtube_service.py` - YouTube Data API communication
- `backend/analytics_service.py` - statistics calculations with pandas
- `frontend/index.html` - page structure
- `frontend/style.css` - appearance/layout
- `frontend/script.js` - calls FastAPI and updates the page
