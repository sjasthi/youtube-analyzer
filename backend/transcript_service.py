"""Functions for retrieving YouTube video transcripts."""

# FIXME: CURRENTLY TESTING - Will change later

from youtube_transcript_api import YouTubeTranscriptApi

def get_video_transcript(video_id):
    """Return the transcript text for one YouTube video."""
    transcript = YouTubeTranscriptApi().fetch(video_id)

    return " ".join(snippet.text for snippet in transcript)

def get_video_transcripts(videos):
    """Return transcripts for a list of videos."""
    transcripts = []

    for video in videos:
        video_id = video["video_id"]

        try:
            transcript = get_video_transcript(video_id)

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
    from youtube_service import get_all_videos

    channel_id = "UCBNvuaTVFUfQlVt9uLeMOfA"
    result = get_all_videos(channel_id)
    videos = result["videos"][:3]
    transcripts = get_video_transcripts(videos)

    for item in transcripts:
        print("\nVIDEO:", item["title"])
        print(item["transcript"][:500])