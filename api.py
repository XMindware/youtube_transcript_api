from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import subprocess, os

client = OpenAI()
app = FastAPI(title="YouTube Summarizer API")

class VideoRequest(BaseModel):
    video_id: str

def get_youtube_transcript(video_id: str):
    """Try to get YouTube captions."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([t['text'] for t in transcript])
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f"Transcript error: {e}")
        return None

def download_audio(video_id: str, output_file="audio.mp3"):
    """Download audio using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    subprocess.run([
        "yt-dlp", "-x", "--audio-format", "mp3", "-o", output_file, url
    ], check=True)
    return output_file

def transcribe_with_whisper(file_path: str):
    """Transcribe using OpenAI Whisper."""
    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(model="whisper-1", file=f)
    return response.text

@app.post("/summarize")
async def summarize_video(req: VideoRequest):
    """Main endpoint: fetch transcript or transcribe."""
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")

    transcript_text = get_youtube_transcript(video_id)
    if transcript_text:
        return {"source": "youtube", "transcript": transcript_text}

    # Fallback: download + transcribe
    try:
        audio_path = download_audio(video_id)
        transcript_text = transcribe_with_whisper(audio_path)
        os.remove(audio_path)
        return {"source": "whisper", "transcript": transcript_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
