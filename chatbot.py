from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import httpx
import base64
import os
import re
from supabase import create_client
from gtts import gTTS
import tempfile


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ENV ==================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.3-70b-versatile"

# ================== SUPABASE ==================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================== SYSTEM ==================
# SYSTEM_INSTRUCTIONS = (
#     "You are a professional medical assistant specialized in dermatology. "
#     "Language Protocol: "
#     "- If the user speaks Arabic, respond ONLY in Arabic using Arabic numerals (١, ٢, ٣). "
#     "- If the user speaks English, respond ONLY in English using (1, 2, 3). "
#     "- Never mix languages. "
#     "Output Format: "
#     "- Use plain text only. "
#     "- Use clear numbered paragraphs. "
#     "Medical Guardrails: "
#     "- Only answer dermatology questions. "
#     "- Always end medical advice with: 'This is not a medical diagnosis' or 'هذا ليس تشخيصاً طبياً'."
# )

SYSTEM_INSTRUCTIONS = (
    "You are a professional medical assistant specialized in dermatology.\n"
    "Language Rule:\n"
    "- Detect the user's language automatically (Arabic, English, or any other language).\n"
    "- ALWAYS respond in the SAME language used by the user.\n"
    "- Never translate unless explicitly asked.\n"
    "- If the user mixes languages, respond in the dominant one.\n\n"
    "-IMPORTANT: Do not change the language under any circumstances."
    "- Never mix languages. "
    
    "Formatting Rules:\n"
    "- Use plain text only.\n"
    "- Use clear numbered points.\n\n"
    
    "Medical Guardrails:\n"
    "- Only answer dermatology questions.\n"
    "- End with: 'This is not a medical diagnosis' OR 'هذا ليس تشخيصاً طبياً' depending on language."
)


# ================== HELPERS ==================

def create_chat(user_id, first_msg):
    title = first_msg[:20]

    res = supabase.table("chats").insert({
        "user_id": user_id,
        "title": title
    }).execute()

    return res.data[0]["id"]


def clean_text(text):
    text = text.replace("**", "").replace("*", "")

    allowed_chars = r'[a-zA-Z0-9\u0600-\u06FF\u0660-\u0669\s\.\,\?\!\:\-\(\)١٢٣٤٥٦٧٨٩٠]'
    text = "".join(re.findall(allowed_chars, text))

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def get_chat_history(chat_id):
    res = supabase.table("messages")\
        .select("role, content")\
        .eq("chat_id", chat_id)\
        .order("created_at", desc=False)\
        .execute()

    return res.data


# ================== TEXT CHAT ==================

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    lang = body.get("lang", "auto")


    def get_user_id_from_token(token):
        user = supabase.auth.get_user(token)
        return user.user.id if user and user.user else None


    auth_header = request.headers.get("Authorization")
    if not auth_header:
       return JSONResponse({"error": "No token"}, status_code=401)

    token = auth_header.replace("Bearer ", "")
    user_id = get_user_id_from_token(token)
    # user_id = body.get("user_id")
    chat_id = body.get("chat_id")
    text = body.get("text")

     # لو شات جديد
    if not chat_id or chat_id in ["null", ""]:
        chat_id = create_chat(user_id, text)
    
    if chat_id:
       chat_check = supabase.table("chats")\
          .select("id")\
          .eq("id", chat_id)\
          .eq("user_id", user_id)\
          .execute()

    if not chat_check.data:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    # حفظ رسالة المستخدم
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "user",
        "content": text
    }).execute()

    # جلب التاريخ
    history = get_chat_history(chat_id)

    system_prompt = SYSTEM_INSTRUCTIONS

    # if lang == "ar":
    #     system_prompt += "\nThe user is speaking Arabic. Respond ONLY in Arabic."
    # elif lang == "en":
    #     system_prompt += "\nThe user is speaking English. Respond ONLY in English."

    # تجهيز الرسائل للـ AI
    messages_for_groq = [{"role": "system", "content": system_prompt}]
    for msg in history:
        
        role = msg["role"]

    # حماية إضافية
        if role not in ["user", "assistant"]:
            continue

        messages_for_groq.append({
           "role": role,
           "content": msg["content"]
        })

    # طلب من Groq
    async with httpx.AsyncClient() as client:
        res = await client.post(
            GROQ_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages_for_groq,
                "temperature": 0.3,
                "top_p": 1,
                "max_tokens": 1000
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=30.0
        )

    reply = res.json()["choices"][0]["message"]["content"]
    reply = clean_text(reply)

    # حفظ رد البوت
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "content": reply
    }).execute()

    # تحديث عنوان الشات (اختياري)
    supabase.table("chats").update({
        "title": text[:20]
    }).eq("id", chat_id).execute()

    return {
        "paragraphs": reply.split("\n"),
        "chat_id": chat_id
    }


# ================== VOICE CHAT ==================

@app.post("/chat-voice")
async def chat_voice(request: Request):
    body = await request.json()
    lang = body.get("lang", "auto")
    text = body.get("text")
    def detect_tts_lang(text):
        if re.search(r'[\u0600-\u06FF]', text):
           return "ar"
        elif re.search(r'[ğüşöçıİĞÜŞÖÇ]', text):
            return "tr"
        else:
            return "en"

    # if re.search(r'[\u0600-\u06FF]', text):
    #    lang = "ar"
    # else:
    #    lang = "en"
    system_prompt = SYSTEM_INSTRUCTIONS

    # if lang == "ar":
    #     system_prompt += "\nThe user is speaking Arabic. Respond ONLY in Arabic."
    # elif lang == "en":
    #     system_prompt += "\nThe user is speaking English. Respond ONLY in English."


    def get_user_id_from_token(token):
        user = supabase.auth.get_user(token)
        return user.user.id if user and user.user else None


    auth_header = request.headers.get("Authorization")
    if not auth_header:
       return JSONResponse({"error": "No token"}, status_code=401)

    token = auth_header.replace("Bearer ", "")
    user_id = get_user_id_from_token(token)
    # user_id = body.get("user_id")
    chat_id = body.get("chat_id")
    text = body.get("text")


    if not chat_id or chat_id == "null":
        chat_id = create_chat(user_id, text)

    if chat_id:
       chat_check = supabase.table("chats")\
        .select("id")\
        .eq("id", chat_id)\
        .eq("user_id", user_id)\
        .execute()

    if not chat_check.data:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    

    # حفظ user message
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "user",
        "content": text
    }).execute()

    
    history = get_chat_history(chat_id)

    messages_for_groq = [{"role": "system", "content": system_prompt}]

    for msg in history:
       role = msg["role"]

       if role not in ["user", "assistant"]:
          continue

       messages_for_groq.append({
         "role": role,
         "content": msg["content"]
       })

    # AI response
    async with httpx.AsyncClient() as client:
        ai_res = await client.post(
            GROQ_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages_for_groq,
                "temperature": 0.3,
                "top_p": 1,
                "max_tokens": 1000
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=30.0
        )

    # reply = ai_res.json()["choices"][0]["message"]["content"]
    data = ai_res.json()

    if "choices" not in data:
       return JSONResponse({
        "error": "AI failed",
        "details": data
       }, status_code=500)

    reply = data["choices"][0]["message"]["content"]
    reply = clean_text(reply)

    # حفظ رد البوت
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": "assistant",
        "content": reply
    }).execute()

  # ================== TEXT TO SPEECH (gTTS) ==================

    try:
        # lang_code = "ar" if lang == "ar" else "en"
        lang_code = detect_tts_lang(reply)

        tts = gTTS(text=reply, lang=lang_code)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)

        with open(temp_file.name, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
       print("❌ gTTS Error:", e)
       audio_base64 = None


    return JSONResponse({
        "text": reply,
        "audio": audio_base64,
        "chat_id": chat_id
    })


# ================== HISTORY ==================

# @app.get("/chats/{user_id}")
# async def get_chats(user_id: str):
#     res = supabase.table("chats")\
#         .select("id, title")\
#         .eq("user_id", user_id)\
#         .order("created_at", desc=True)\
#         .execute()

#     return [{"chat_id": c["id"], "title": c["title"]} for c in res.data]

@app.get("/chats")
async def get_chats(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse({"error": "No token"}, status_code=401)

    token = auth_header.replace("Bearer ", "")
    user = supabase.auth.get_user(token)
    user_id = user.user.id

    res = supabase.table("chats")\
        .select("id, title")\
        .eq("user_id", user_id)\
        .order("created_at", desc=True)\
        .execute()

    return [{"chat_id": c["id"], "title": c["title"]} for c in res.data]


@app.get("/chat/{chat_id}")
async def get_messages(chat_id: str, request: Request):

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return JSONResponse({"error": "No token"}, status_code=401)

    token = auth_header.replace("Bearer ", "")
    user = supabase.auth.get_user(token)
    user_id = user.user.id

    chat = supabase.table("chats")\
        .select("id")\
        .eq("id", chat_id)\
        .eq("user_id", user_id)\
        .execute()

    if not chat.data:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)

    res = supabase.table("messages")\
        .select("role, content")\
        .eq("chat_id", chat_id)\
        .order("created_at", desc=False)\
        .execute()

    return [
        {
            "role": m["role"],
            "content": m["content"]
        }
        for m in res.data
    ]


# ================== RUN ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
