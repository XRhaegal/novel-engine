import os
import re
import glob
import base64
import requests
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
active_caches = {}

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

class AutoCloseRequest(BaseModel):
    github_token: str
    repo_owner: str
    repo_name: str
    chapter_num: str
    full_text: str
    api_key: Optional[str] = None
    model: Optional[str] = DEFAULT_MODEL

def get_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=401, 
            detail="Falta la API Key de Google AI Studio. Ingrésala en la app."
        )
    return genai.Client(api_key=key), key

def read_canon_context():
    main_prompt = "Eres el co-piloto narrativo de Cedar Creek 1985."
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

def commit_to_github(token: str, owner: str, repo: str, path: str, content: str, commit_msg: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    sha = None
    get_res = requests.get(url, headers=headers)
    if get_res.status_code == 200:
        sha = get_res.json().get("sha")

    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    data = {"message": commit_msg, "content": encoded}
    if sha:
        data["sha"] = sha

    put_res = requests.put(url, headers=headers, json=data)
    if put_res.status_code not in [200, 201]:
        raise HTTPException(status_code=put_res.status_code, detail=f"GitHub Error: {put_res.text}")
    return True

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
        raise HTTPException(status_code=400, detail="No se encontraron archivos en /canon/.")

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
        raise HTTPException(status_code=500, detail=f"Error creando caché: {str(e)}")

@app.post("/api/auto-close-chapter")
def auto_close_chapter(req: AutoCloseRequest):
    """Extrae la continuidad, actualiza main_instructions.txt, guarda el capítulo y sincroniza."""
    # 1. Separar prosa de continuidad si viene incluida
    continuity_match = re.search(r"═+\s*CONTINUITY STATE.*", req.full_text, re.DOTALL)
    continuity_block = ""
    prose_text = req.full_text

    if continuity_match:
        continuity_block = continuity_match.group(0).strip()
        prose_text = req.full_text[:continuity_match.start()].strip()

    # 2. Guardar el archivo en canon/capitulo_XX.txt en GitHub
    cap_filename = f"capitulo_{req.chapter_num.zfill(2)}.txt"
    commit_to_github(
        req.github_token, req.repo_owner, req.repo_name, 
        f"canon/{cap_filename}", prose_text, 
        f"Cierre automático: Guardar {cap_filename}"
    )

    # Guardar en local del contenedor
    os.makedirs("canon", exist_ok=True)
    with open(f"canon/{cap_filename}", "w", encoding="utf-8") as f:
        f.write(prose_text)

    # 3. Si hay bloque de continuidad, actualizar la Sección 15 de main_instructions.txt en GitHub
    if continuity_block and os.path.exists("main_instructions.txt"):
        with open("main_instructions.txt", "r", encoding="utf-8") as f:
            current_main = f.read()

        # Reemplazar la Sección 15 conservando el resto
        pattern = r"(## 15\. ESTADO ACTUAL DE CONTINUIDAD.*?)(?=## 16\. CONFLICTO TERRITORIAL|$)"
        replacement = f"## 15. ESTADO ACTUAL DE CONTINUIDAD (ACTUALIZADO AUTOMÁTICAMENTE)\n\n```\n{continuity_block}\n```\n\n---\n\n"
        
        new_main = re.sub(pattern, replacement, current_main, flags=re.DOTALL)
        
        commit_to_github(
            req.github_token, req.repo_owner, req.repo_name,
            "main_instructions.txt", new_main,
            f"Cierre automático: Actualizar continuidad Cap {req.chapter_num}"
        )
        with open("main_instructions.txt", "w", encoding="utf-8") as f:
            f.write(new_main)

    # 4. Re-sincronizar la memoria caché de Gemini de una vez
    model_name = req.model or DEFAULT_MODEL
    client, key = get_client(req.api_key)
    main_prompt, full_canon, total_files = read_canon_context()

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
    except Exception as e:
        print(f"Advertencia al re-sincronizar caché: {e}")

    return {
        "status": "success",
        "message": f"Capítulo {req.chapter_num} archivado, Sección 15 actualizada y Caché refrescada con éxito."
    }

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
