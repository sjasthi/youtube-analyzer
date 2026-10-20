"""Functions for retrieving YouTube video transcripts."""

from youtube_transcript_api import YouTubeTranscriptApi

from backend.database_service import get_saved_transcript, save_transcript

def get_video_transcript(video_id, channel_id):
    """Return a saved transcript or retrieve one from YouTube."""

    # Checks the database first
    saved = get_saved_transcript(video_id)

    if saved is not None:
        print(f"Using cached transcript from SQLite: {video_id}")
        return saved["transcript"]
    
    print(f"Fetching transcript from YouTube: {video_id}")

    # Retrieves the transcript from YouTube if it is not saved
    transcript = YouTubeTranscriptApi().fetch(video_id)
    transcript_text = " ".join(snippet.text for snippet in transcript)

    # Saves the transcript for future requests
    save_transcript(video_id, channel_id, transcript_text)

    return transcript_text

# Retrieves transcripts for multiple videos, skipping videos that fail
def get_video_transcripts(videos, channel_id):
    """Return transcripts for a list of videos."""
    transcripts = []

    for video in videos:
        video_id = video["video_id"]

        try:
            transcript = get_video_transcript(video_id, channel_id)

            transcripts.append({
                "video_id": video_id,
                "title": video["title"],
                "transcript": transcript,
            })

        except Exception as error:
            print(f"Transcript error for {video_id}: {error}")
            continue

    return transcripts

if __name__ == "__main__":
    from backend.youtube_service import get_all_videos

    channel_id = "UCBNvuaTVFUfQlVt9uLeMOfA"

    result = get_all_videos(channel_id)
    videos = result["videos"][:1]

    transcripts = get_video_transcripts(videos, channel_id)

    for item in transcripts:
        print("\nVIDEO:", item["title"])
        print(item["transcript"][:500])