import os
import glob
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Cedar Creek 1985 - Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "gemini-3.8-flash"

# Cachés activas organizadas por api_key y modelo: { (api_key, model_name): cache_name }
active_caches = {}

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class ContinuityUpdateRequest(BaseModel):
    new_entry: str

def get_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=401, 
            detail="Falta la API Key de Google AI Studio. Ingrésala en la app."
        )
    return genai.Client(api_key=key), key

def read_canon_context():
    main_prompt = "Eres el co-piloto narrativo de Cedar Creek 1985. Mantén la ambientación rigurosa de 1985, coherencia total y la voz de los personajes."
    if os.path.exists("main_instructions.txt"):
        with open("main_instructions.txt", "r", encoding="utf-8") as f:
            main_prompt = f.read()

    canon_blocks = []
    files = glob.glob("canon/**/*.txt", recursive=True) + glob.glob("canon/**/*.md", recursive=True) + glob.glob("canon/*.*")
    files = sorted(list(set(files)))
    
    for path in files:
        if os.path.isfile(path) and not path.endswith(".py"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        filename = os.path.basename(path)
                        canon_blocks.append(f"=== [CANON / EXPEDIENTE: {filename}] ===\n{content}\n")
            except Exception as e:
                print(f"Error al leer {path}: {e}")

    full_canon = "\n".join(canon_blocks)
    return main_prompt, full_canon, len(canon_blocks)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cedar Creek Engine Activo</h1><p>Falta templates/index.html</p>"

@app.post("/api/sync-cache")
def sync_cache(payload: Optional[ChatRequest] = None):
    api_key = payload.api_key if payload else None
    model_name = payload.model if (payload and payload.model) else DEFAULT_MODEL
    client, key = get_client(api_key)
    
    main_prompt, full_canon, total_files = read_canon_context()
    if not full_canon.strip():
        raise HTTPException(
            status_code=400, 
            detail="No se encontraron archivos en la carpeta /canon/."
        )

    try:
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                contents=[full_canon],
                system_instruction=main_prompt,
                ttl="86400s",
            )
        )
        active_caches[(key, model_name)] = cache.name
        return {
            "status": "success",
            "cache_id": cache.name,
            "total_files": total_files,
            "model": model_name,
            "message": f"¡Caché montada en {model_name} con {total_files} archivos!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando la caché: {str(e)}")

@app.post("/api/chat")
def chat(req: ChatRequest):
    client, key = get_client(req.api_key)
    model_name = req.model or DEFAULT_MODEL
    cache_id = active_caches.get((key, model_name))
    
    if not cache_id:
        main_prompt, full_canon, _ = read_canon_context()
        if full_canon.strip():
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    contents=[full_canon],
                    system_instruction=main_prompt,
                    ttl="86400s",
                )
            )
            cache_id = cache.name
            active_caches[(key, model_name)] = cache_id

    try:
        if cache_id:
            response = client.models.generate_content(
                model=model_name,
                contents=req.message,
                config=types.GenerateContentConfig(cached_content=cache_id)
            )
        else:
            response = client.models.generate_content(
                model=model_name,
                contents=req.message
            )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
