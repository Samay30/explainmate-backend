# ---------------- main.py (Updated Backend with FastAPI) ----------------

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
import whisper
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "key.env"))

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper ASR model
model = whisper.load_model("base")  # You can switch to "medium" for better accuracy

# ElevenLabs voice mapping
VOICE_NAME_TO_ID = {
    "Rachelle": "ZT9u07TYPVl83ejeLakq",
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Domi": "AZnzlk1XvdvUeBnXmlld"
}

# ---------------- API Models ----------------

class CodeInput(BaseModel):
    code: str
    language: str = "en"
    speed: float = 1.0
    voice: str = "default"
    mode: str = "default"

class FollowUpInput(BaseModel):
    question: str
    explanation: str
    code: str

# ---------------- Utility Functions ----------------

def speak_text(text: str, voice: str = "Rachel") -> str:
    try:
        start_time = time.time()
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        voice_id = VOICE_NAME_TO_ID.get(voice, voice)

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.75,
                style=0.0,
                use_speaker_boost=True
            )
        )

        audio_bytes = b"".join(audio_stream)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(audio_bytes)
        temp_file.close()

        print(f"[✅] Audio generated in {time.time() - start_time:.2f} seconds")
        return temp_file.name

    except Exception as e:
        print("🛑 ElevenLabs TTS Error:", str(e))
        raise RuntimeError(f"Text-to-speech failed: {str(e)}")

# ---------------- API Endpoints ----------------

@app.post("/explain")
def explain_code(payload: CodeInput):
    try:
        start_time = time.time()

        code_lines = []
        depth = "beginner"
        fmt = "default"

        for line in payload.code.splitlines():
            if line.startswith("Depth:"):
                depth = line.replace("Depth:", "").strip()
            elif line.startswith("Format:"):
                fmt = line.replace("Format:", "").strip()
            else:
                code_lines.append(line)

        code = "\n".join(code_lines)

        if payload.mode == "eli5":
            format_prompt = "Explain this code simply and briefly in 300 words, as if to a 5-year-old."
        elif payload.mode == "spoken":
            format_prompt = "Summarize this code in 300 words in a conversational, natural tone."
        else:
            format_prompt = f"Explain the code in {depth}-level terms with a concise, spoken-friendly explanation in 300 words."

        prompt = f"{format_prompt}\n\n{code}"

        print("[📤] Sending prompt to Together.ai:\n", prompt)

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_tokens": 400,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]
            },
            verify=False
        )

        result = response.json()
        explanation = result.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no explanation generated.")

        print(f"[✅] Explanation generated in {time.time() - start_time:.2f} seconds")
        return {"explanation": explanation}

    except Exception as e:
        print("[❌] Error in /explain:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/speak")
def speak_code(payload: CodeInput):
    try:
        print("📥 Received for TTS:", payload.dict())
        audio_path = speak_text(payload.code, payload.voice)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="speech.mp3")

    except Exception as e:
        print("[❌] Error in /speak:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask")
def ask_followup(input: FollowUpInput):
    try:
        prompt = (
            f"The user had a question about the code:\n\n{input.code}\n\n"
            f"Previous explanation:\n\n{input.explanation}\n\n"
            f"User's question:\n\"{input.question}\"\n"
            "Please clarify or expand to help the user understand clearly."
        )

        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "max_tokens": 300,
                "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}]
            },
            verify=False
        )

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no answer generated.")
        return {"answer": answer}

    except Exception as e:
        print("[❌] Error in /ask:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/transcribe")
def transcribe_question(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio.file.read())
            tmp_path = tmp.name

        print(f"[🎙️] Transcribing {tmp_path}")
        result = model.transcribe(tmp_path)
        return {"transcript": result["text"]}

    except Exception as e:
        print("[❌] Error in /transcribe:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
