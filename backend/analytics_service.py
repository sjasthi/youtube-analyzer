"""Calculate analytics from the videos returned by YouTube."""

import pandas as pd


def calculate_analytics(videos):
    """Return summary statistics for the retrieved video sample."""

    # If YouTube returned no videos, return safe defaults instead of
    # attempting averages that would cause errors.
    if not videos:
        return {
            "sample_size": 0,
            "total_views": 0,
            "total_likes": 0,
            "total_comments": 0,
            "average_views": 0,
            "average_likes": 0,
            "average_comments": 0,
            "average_duration_seconds": 0,
            "engagement_rate_percent": 0,
            "average_days_between_uploads": None,
            "estimated_uploads_per_month": None,
            "most_viewed_video": None,
            "most_liked_video": None,
            "uploads_by_month": [],
        }

    # A pandas DataFrame works like a table/spreadsheet.
    # Each video dictionary becomes one row.
    df = pd.DataFrame(videos)

    # Ensure values used in calculations are numeric.
    for column in ["view_count", "like_count", "comment_count", "duration_seconds"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    # Totals for the retrieved sample.
    total_views = int(df["view_count"].sum())
    total_likes = int(df["like_count"].sum())
    total_comments = int(df["comment_count"].sum())

    # Project engagement formula:
    # (likes + comments) / views * 100
    if total_views > 0:
        engagement_rate = ((total_likes + total_comments) / total_views) * 100
    else:
        engagement_rate = 0

    # Find the rows with the largest view and like counts.
    most_viewed = df.loc[df["view_count"].idxmax()]
    most_liked = df.loc[df["like_count"].idxmax()]

    # Convert YouTube date strings into real datetime values.
    df["published_datetime"] = pd.to_datetime(
        df["published_at"], errors="coerce", utc=True
    )

    # Calculate time between uploads using valid publication dates.
    valid_dates = df["published_datetime"].dropna().sort_values()
    avg_days = None
    uploads_per_month = None

    if len(valid_dates) >= 2:
        differences = valid_dates.diff().dropna()
        avg_days = differences.dt.total_seconds().mean() / 86400

        # 30.44 is approximately the average number of days in a month.
        if avg_days > 0:
            uploads_per_month = 30.44 / avg_days

    # Count how many retrieved videos were published in each month.
    uploads_by_month = []
    dated_rows = df.dropna(subset=["published_datetime"]).copy()

    if not dated_rows.empty:
        dated_rows["month"] = dated_rows["published_datetime"].dt.strftime("%Y-%m")
        month_counts = dated_rows.groupby("month").size().sort_index()

        uploads_by_month = [
            {"month": month, "video_count": int(count)}
            for month, count in month_counts.items()
        ]

    # FastAPI will automatically convert this Python dictionary to JSON.
    return {
        "sample_size": int(len(df)),
        "total_views": total_views,
        "total_likes": total_likes,
        "total_comments": total_comments,
        "average_views": round(float(df["view_count"].mean()), 2),
        "average_likes": round(float(df["like_count"].mean()), 2),
        "average_comments": round(float(df["comment_count"].mean()), 2),
        "average_duration_seconds": round(float(df["duration_seconds"].mean()), 2),
        "engagement_rate_percent": round(engagement_rate, 3),
        "average_days_between_uploads": round(avg_days, 2) if avg_days is not None else None,
        "estimated_uploads_per_month": round(uploads_per_month, 2) if uploads_per_month is not None else None,
        "most_viewed_video": {
            "video_id": most_viewed["video_id"],
            "title": most_viewed["title"],
            "view_count": int(most_viewed["view_count"]),
        },
        "most_liked_video": {
            "video_id": most_liked["video_id"],
            "title": most_liked["title"],
            "like_count": int(most_liked["like_count"]),
        },
        "uploads_by_month": uploads_by_month,
    }
