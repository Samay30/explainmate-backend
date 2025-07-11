from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
import traceback
import tempfile
import time
import ssl
import re

# Optional: Uncomment for local SSL bypass only
# if os.environ.get("SKIP_SSL_VERIFY") == "1":
#     ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "key.env"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_NAME_TO_ID = {
    "Rachelle": "ZT9u07TYPVl83ejeLakq",
}

class CodeInput(BaseModel):
    code: str
    language: str = "en"
    speed: float = 0.85
    voice: str = "Rachelle"
    mode: str = "default"
    max_sentences: int | None = None   # ⇦ NEW (allow caller to override trimming)

class FollowUpInput(BaseModel):
    question: str
    explanation: str
    code: str

# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────

def split_text(text: str, max_len: int = 700):
    """Split cleaned text into ≤ max_len‑char chunks on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        # +1 for the space we removed in split
        if len(current) + len(sentence) + 1 <= max_len:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def clean_for_tts(text: str, max_sentences: int | None = 5) -> str:
    """Remove markdown/emoji and optionally limit to N sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if max_sentences:
        sentences = sentences[:max_sentences]
    text = " ".join(sentences)
    text = re.sub(r"[`_*#~\[\]{}<>]", "", text)                # markdown / brackets
    text = re.sub(r"[^\w\s.,!?\'\"()\-]", "", text)         # emojis / misc
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ────────────────────────────────────────────────────────────────────────────
# ElevenLabs wrapper
# ────────────────────────────────────────────────────────────────────────────

def speak_text(text: str, *, voice: str = "Rachelle", speed: float = 0.85) -> str:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import VoiceSettings

    client   = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    voice_id = VOICE_NAME_TO_ID.get(voice, voice)

    audio_bytes = b""
    for chunk in split_text(text):
        stream = client.text_to_speech.convert(
            text=chunk,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True,
                speed=speed,
            ),
        )
        for part in stream:  # consume fully
            audio_bytes += part

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

# ────────────────────────────────────────────────────────────────────────────
# API Routes
# ────────────────────────────────────────────────────────────────────────────

@app.post("/explain")
def explain_code(payload: CodeInput):
    try:
        code_lines, depth = [], "beginner"
        for line in payload.code.splitlines():
            if line.startswith("Depth:"):
                depth = line.replace("Depth:", "").strip()
            else:
                code_lines.append(line)
        code = "\n".join(code_lines)

        prompt_map = {
            "eli5":   "Explain this code simply and strictly in 150 words, as if to a 5‑year‑old in a conversational tone.",
            "spoken": "Summarize this code in 150 words in a conversational, natural tone.",
        }
        format_prompt = prompt_map.get(payload.mode, f"Explain the code in {depth}-level terms with a concise, spoken‑friendly explanation in 300 words.")
        prompt = f"{format_prompt}\n\n{code}"

        res = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_tokens": 400,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            verify=False,
            timeout=45,
        )
        explanation = res.json().get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no explanation generated.")
        return {"explanation": explanation}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speak")
def speak_code(payload: CodeInput):
    """Generate full‑length audio. Caller can set max_sentences via payload (None = no limit)."""
    cleaned = clean_for_tts(payload.code, max_sentences=payload.max_sentences)
    audio   = speak_text(cleaned, voice=payload.voice, speed=payload.speed)
    return FileResponse(audio, media_type="audio/mpeg", filename="speech.mp3")

@app.post("/ask")
def ask_followup(input: FollowUpInput):
    try:
        prompt = (
            f"The user had a question about the code:\n\n{input.code}\n\n"
            f"Previous explanation:\n\n{input.explanation}\n\n"
            f"User's question:\n\"{input.question}\"\n"
            "Please clarify or expand to help the user understand clearly."
        )
        res = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_tokens": 650,
                "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}],
            },
            verify=False,
            timeout=45,
        )
        answer = res.json().get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no answer generated.")
        return {"answer": answer}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
