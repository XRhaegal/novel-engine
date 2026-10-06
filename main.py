import os
import glob
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, PlainTextResponse
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

MODEL_NAME = "gemini-2.5-flash"

# Almacén en memoria para cachés activas (asociadas a la API Key)
active_caches = {}

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None

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
    """Lee el main_instructions y todos los archivos dentro de la carpeta canon/"""
    main_prompt = "Eres el co-piloto narrativo de Cedar Creek 1985. Mantén la ambientación rigurosa de 1985, coherencia total y la voz de los personajes."
    if os.path.exists("main_instructions.txt"):
        with open("main_instructions.txt", "r", encoding="utf-8") as f:
            main_prompt = f.read()

    canon_blocks = []
    # Busca archivos de texto y markdown en la carpeta canon/
    files = glob.glob("canon/**/*.txt", recursive=True) + glob.glob("canon/**/*.md", recursive=True) + glob.glob("canon/*.*")
    
    # Quitar duplicados si hay
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
    """Sirve la interfaz web para PC y Celular si existe index.html"""
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cedar Creek Engine Activo</h1><p>Falta subir templates/index.html</p>"

@app.post("/api/sync-cache")
def sync_cache(payload: Optional[ChatRequest] = None):
    """Compila los 40+ archivos y los monta en el Context Caching de Gemini"""
    api_key = payload.api_key if payload else None
    client, key = get_client(api_key)
    
    main_prompt, full_canon, total_files = read_canon_context()
    if not full_canon.strip():
        raise HTTPException(
            status_code=400, 
            detail="No se encontraron archivos en la carpeta /canon/. Sube tus capítulos o fichas primero."
        )

    try:
        # Crea caché por 24 horas (86400 segundos) en Google AI Studio
        cache = client.caches.create(
            model=MODEL_NAME,
            config=types.CreateCachedContentConfig(
                contents=[full_canon],
                system_instruction=main_prompt,
                ttl="86400s",
            )
        )
        active_caches[key] = cache.name
        return {
            "status": "success",
            "cache_id": cache.name,
            "total_files": total_files,
            "message": f"¡Caché montada exitosamente con {total_files} archivos!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando la caché: {str(e)}")

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Envía el turno usando la caché en memoria para velocidad y ahorro de tokens"""
    client, key = get_client(req.api_key)
    cache_id = active_caches.get(key)
    
    # Si la caché expiró o no se ha creado, intentamos crearla al vuelo
    if not cache_id:
        main_prompt, full_canon, _ = read_canon_context()
        if full_canon.strip():
            cache = client.caches.create(
                model=MODEL_NAME,
                config=types.CreateCachedContentConfig(
                    contents=[full_canon],
                    system_instruction=main_prompt,
                    ttl="86400s",
                )
            )
            cache_id = cache.name
            active_caches[key] = cache_id

    try:
        if cache_id:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=req.message,
                config=types.GenerateContentConfig(
                    cached_content=cache_id
                )
            )
        else:
            # Fallback en caso de que no haya archivos en canon aún
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=req.message
            )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/append-continuity")
def append_continuity(req: ContinuityUpdateRequest):
    """Agrega un nuevo hito al archivo continuity.txt para no olvidar nada"""
    os.makedirs("canon", exist_ok=True)
    continuity_path = os.path.join("canon", "continuity.txt")
    with open(continuity_path, "a", encoding="utf-8") as f:
        f.write(f"\n- {req.new_entry}")
    return {"status": "success", "message": "Continuidad actualizada en el canon."}
