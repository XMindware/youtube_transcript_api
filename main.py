from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
import subprocess, os, datetime, json

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
    """Try to get YouTube captions in the video's own language. Returns (text, language_code)."""
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred = None
        for tr in transcripts:
            if not getattr(tr, "is_generated", False):
                preferred = tr
                break
        if preferred is None:
            preferred = next(iter(transcripts))
        entries = preferred.fetch()
        text = " ".join([t['text'] for t in entries])
        return text, preferred.language_code
    except (TranscriptsDisabled, NoTranscriptFound):
        return None, None
    except Exception as e:
        print(f"Transcript error: {e}")
        return None, None

def get_video_info(video_id: str):
    """Return (duration_in_seconds, language_code|None) using yt-dlp JSON output."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-warnings",
                "--dump-json",
                "--skip-download",
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Some videos/playlists may output multiple lines; take the first JSON object
        first_line = result.stdout.splitlines()[0] if result.stdout else "{}"
        data = json.loads(first_line)
        duration = data.get("duration")
        language_code = data.get("language")
        return (int(duration) if duration is not None else 0, language_code)
    except Exception as e:
        print(f"Duration fetch error: {e}")
        return (0, None)

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
        messages=[
            {
                "role": "system",
                "content": """You are an assistant that summarizes YouTube videos in a structured, clear, and insightful way.
Always respond in the same language as the provided transcript.
Your tone should be concise and intelligent, occasionally using emojis as visual markers.
When the video is argumentative or analytical, extract and format the content like this:

🎯 Main Argument:
<1–2 sentence core idea>

🔍 Key Takeaways & Themes:
- Use bold subheadings and emojis for categories
- Provide brief but complete bullets (each with its own short explanation)
- Add names, events, or facts when relevant

📌 Conclusion:
<Wrap-up of core takeaway or what this means going forward>

If the video is a tutorial or guide, instead focus on:
- 🎓 Purpose
- 🔧 Steps / Instructions
- ✅ Final Outcome

Avoid unnecessary fluff. Focus on clarity and high-level insight.""",
            },
            {
            "role": "user",
            "content": f"This is the transcript of a YouTube video. Please give a short, clear explanation of what the video is about:\n\n{text}"
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

    text, detected_language = get_youtube_transcript(video_id)
    source = "youtube"

    if not text:
        try:
            # Enforce 10-minute max for free tier when falling back to audio transcription
            duration_sec, lang_code = get_video_info(video_id)
            if duration_sec and duration_sec > 600:
                msg = (
                    "La versión gratuita solo admite audios de menos de 10 minutos."
                    if (lang_code or "").startswith("es") else
                    "Free version only supports less than 10min audios."
                )
                raise HTTPException(status_code=400, detail=msg)
            audio_path = download_audio(video_id)
            text = transcribe_with_whisper(audio_path)
            os.remove(audio_path)
            source = "whisper"
            detected_language = lang_code
        except HTTPException as http_err:
            # Propagate intended HTTP errors (e.g., 400 for duration limit)
            raise http_err
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
