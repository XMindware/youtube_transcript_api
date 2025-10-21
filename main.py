from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import subprocess, os, datetime

# Initialize
client = OpenAI()
app = FastAPI(title="YouTube Transcript API")
templates = Jinja2Templates(directory="templates")

# CORS restriction
origins = [
    "https://narrify.cloud",
    "http://localhost",
    "http://127.0.0.1"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Ensure folder exists
os.makedirs("transcripts", exist_ok=True)

class VideoRequest(BaseModel):
    video_id: str

def get_youtube_transcript(video_id: str):
    """Try to get YouTube captions."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([t['text'] for t in transcript])
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

def summarize_text(text: str):
    """Summarize transcript using GPT."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Summarize this YouTube transcript in 3-5 sentences:\n\n{text}"
        }]
    )
    return response.choices[0].message.content.strip()

def save_transcript(video_id: str, text: str, summary: str, source: str):
    """Save transcript locally."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"transcripts/{video_id}_{timestamp}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"[Source: {source}]\n\n")
        f.write(f"Summary:\n{summary}\n\n---\n\n{text}")
    return path

@app.post("/summarize")
async def summarize_video(req: VideoRequest):
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="Missing video_id")

    text = get_youtube_transcript(video_id)
    source = "youtube"

    if not text:
        try:
            audio_path = download_audio(video_id)
            text = transcribe_with_whisper(audio_path)
            os.remove(audio_path)
            source = "whisper"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    summary = summarize_text(text)
    file_path = save_transcript(video_id, text, summary, source)
    return {"video_id": video_id, "source": source, "summary": summary, "file": file_path}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    files = sorted(os.listdir("transcripts"), reverse=True)
    entries = []
    for f in files:
        path = os.path.join("transcripts", f)
        with open(path, encoding="utf-8") as fp:
            first_line = fp.readline().strip()
            summary = ""
            for line in fp:
                if line.startswith("Summary:"):
                    summary = fp.readline().strip()
                    break
            entries.append({
                "filename": f,
                "source": first_line.replace("[Source:", "").replace("]", "").strip(),
                "summary": summary
            })
    return templates.TemplateResponse("index.html", {"request": request, "entries": entries})
