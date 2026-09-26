# YouTube Channel Analyzer

This version adds a Python analytics layer. In order to look up Creator analytics, use the UC.. id. Able to check at most 50 videos. Shows Subscribers, Total channel videos, Views and posting trends.

## What it calculates

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

## Engagement formula

```text
(likes + comments) / views * 100
```

## Install the dependencies

Keep your existing `.env` and `.venv`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

## Run

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

## What each file does

- `backend/main.py` - FastAPI routes and application flow
- `backend/youtube_service.py` - YouTube Data API communication
- `backend/analytics_service.py` - statistics calculations with pandas
- `frontend/index.html` - page structure
- `frontend/style.css` - appearance/layout
- `frontend/script.js` - calls FastAPI and updates the page
