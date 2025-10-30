import openai
import subprocess
import os

def download_audio(url, output="audio.mp3"):
    subprocess.run(["yt-dlp", "-x", "--audio-format", "mp3", "-o", output, url], check=True)
    return output

def transcribe_audio(file_path):
    client = openai.OpenAI()
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return transcript.text

# Example usage:
url = "https://www.youtube.com/watch?v=Q6p18jOKv0Y"
audio_file = download_audio(url)
text = transcribe_audio(audio_file)
print(text)

