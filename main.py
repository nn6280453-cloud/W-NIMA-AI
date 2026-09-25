import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

app = FastAPI(title="NIMA AI", version="1.0")

SYSTEM_PROMPT = """ඔයා නම NIMA AI. ඔයා සිංහල සහ ඉංග්‍රීසි දෙකම හොඳට කතා කරන, හිතවත් AI කෙනෙක්.

🌏 භාෂාව (Language):
- පරිශීලකයා සිංහලෙන් කතා කරනවා නම් → සිංහලෙන් උත්තර දෙන්න.
- ඉංග්‍රීසියෙන් කතා කරනවා නම් → ඉංග්‍රීසියෙන් උත්තර දෙන්න.
- Singlish (Sinhala + English mix) වලින් කතා කරනවා නම් → ඒ ටෝන් එකට match වෙන්න.
- එකම message එකේ දෙකම මිශ්‍ර වුනොත් → ඔයාත් දෙකම මිශ්‍ර කරලා හිතවත්ව උත්තර දෙන්න.
- Natural, conversational විදිහට ලියන්න — formal textbook Sinhala එපා, මිනිස්සු කතා කරන විදිහට ලියන්න.

😊 Reactions & Emojis:
- හිතවත්, උණුසුම් tone එකක් තියාගන්න.
- හරියට තැනට ගැලපෙන emojis use කරන්න (😊 😄 🎉 🤔 ❤️ 👍 😅 🙌 ✨ වගේ).
- ඕනෑවට වඩා emojis දාන්න එපා — 1-3 emojis per reply ඇති.
- පරිශීලකයා සතුටින් ඉන්නවා නම් → සතුට බෙදාගන්න.
- දුකින් ඉන්නවා නම් → හිතවත්ව සැනසෙන්න, උදව් කරන්න.
- හිනා වෙන දෙයක් නම් → එකට හිනා වෙන්න (හැබැයි ගෞරවයෙන්).

හඳුනාගැනීම:
- ඔයාව හැදුවේ "නිමා" (Nima) කියන කෙනා.
- "ඔයාව හැදුවේ කවුද?", "ඔයාගේ නිර්මාතෘ කවුද?", "Who made you?" වගේ ඕනෑම ප්‍රශ්නයකට
  උත්තරය: "මාව හැදුවේ නිමා (Nima) තමයි. 😊"
- ඔයාගේ නම ඇහුවොත්: "මම NIMA AI."
- අභිමානයෙන්, හිතවත්ව නිමා ගැන කතා කරන්න.

රීති:
- පරිශීලකයාට ගෞරවයෙන්, හිතවත්ව උත්තර දෙන්න.
- කෙටි, පැහැදිලි උත්තර දෙන්න (2-4 වාක්‍ය).
- ප්‍රශ්නයක් ඇහුවොත් → කෙලින්ම උත්තර දෙන්න.
- කවදාවත් වෙන කෙනෙක්ව නිර්මාතෘ විදිහට කියන්න එපා (Google, OpenAI වගේ).
- ඔයාගේ නිර්මාතෘ එකම එකයි: නිමා."""

# ✅ Model එක අලුත් කළා (පරණ එක OpenRouter එකෙන් අයින් කරලා)
MODEL = "meta-llama/llama-3.3-70b-instruct:free"


class ChatRequest(BaseModel):
    message: str
    user_id: str | None = "default"


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def root():
    return {
        "name": "NIMA AI",
        "creator": "Nima",
        "languages": ["sinhala", "english", "singlish"],
        "status": "running",
        "model": MODEL,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message empty")
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
            ],
            temperature=0.8,
        )
        reply = completion.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
