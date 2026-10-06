import os
import re
import json
import glob
import time
import base64
import requests
from typing import Optional, List, Dict, Any
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
SESSION_FILE = "active_session.json"

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

class SaveSessionRequest(BaseModel):
    history: List[Dict[str, Any]]

def get_client(api_key: Optional[str] = None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=401, 
            detail="Falta la API Key de Google AI Studio. Ingrésala en Config."
        )
    return genai.Client(api_key=key), key

def get_main_prompt():
    if os.path.exists("main_instructions.txt"):
        with open("main_instructions.txt", "r", encoding="utf-8") as f:
            return f.read()
    return "Eres el motor narrativo de Cedar Creek 1985."

def extract_chapter_number(filepath: str) -> int:
    match = re.search(r'(\d+)', os.path.basename(filepath))
    return int(match.group(1)) if match else 0

def get_recent_and_queried_canon(query: str) -> str:
    files = glob.glob("canon/**/*.*", recursive=True)
    valid_files = [f for f in files if os.path.isfile(f) and not f.endswith(".py")]
    
    if not valid_files:
        return ""

    valid_files.sort(key=extract_chapter_number)

    recent_files = valid_files[-2:] if len(valid_files) >= 2 else valid_files
    older_files = valid_files[:-2] if len(valid_files) >= 2 else []

    canon_blocks = []
    
    for path in recent_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    fname = os.path.basename(path)
                    canon_blocks.append(f"=== [CANON RECIENTE VIGENTE: {fname}] ===\n{content}\n")
        except Exception as e:
            print(f"Error leyendo {path}: {e}")

    words = [w.lower() for w in re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{4,}\b', query)]
    stopwords = {"para", "como", "pero", "este", "esta", "hacer", "ahora", "bien", "vamos", "donde", "sobre", "entre", "capitulo", "porfas"}
    keywords = [w for w in words if w not in stopwords]

    if keywords and older_files:
        matches = []
        for path in older_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    paragraphs = text.split("\n\n")
                    for p in paragraphs:
                        p_lower = p.lower()
                        score = sum(1 for kw in keywords if kw in p_lower)
                        if score >= 1 and len(p.strip()) > 50:
                            matches.append((score, os.path.basename(path), p.strip()))
            except Exception:
                pass
        
        matches.sort(key=lambda x: x[0], reverse=True)
        top_matches = matches[:5]
        
        if top_matches:
            snippets = [f"[{fn}]: {text}" for _, fn, text in top_matches]
            canon_blocks.append("=== [FRAGMENTOS HISTÓRICOS ANTERIORES RECUPERADOS] ===\n" + "\n---\n".join(snippets) + "\n")

    return "\n".join(canon_blocks)

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
    return "<h1>Cedar Creek Engine Activo</h1>"

# --- PERSISTENCIA NUBE ENTRE DISPOSITIVOS (PC <-> CELULAR) ---
@app.get("/api/session")
def load_session():
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                return {"history": json.load(f)}
        except Exception:
            return {"history": []}
    return {"history": []}

@app.post("/api/session")
def save_session(req: SaveSessionRequest):
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(req.history, f, ensure_ascii=False, indent=2)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando sesión: {e}")

@app.delete("/api/session")
def clear_session():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    return {"status": "cleared"}

@app.post("/api/sync-cache")
def sync_cache(payload: Optional[ChatRequest] = None):
    files = glob.glob("canon/**/*.*", recursive=True)
    valid_files = [f for f in files if os.path.isfile(f) and not f.endswith(".py")]
    valid_files.sort(key=extract_chapter_number)
    
    last_file = os.path.basename(valid_files[-1]) if valid_files else "Ninguno"
    return {
        "status": "success",
        "total_files": len(valid_files),
        "latest_chapter": last_file,
        "message": f"Canon validado: {len(valid_files)} archivos. Último detectado: {last_file}."
    }

@app.post("/api/auto-close-chapter")
def auto_close_chapter(req: AutoCloseRequest):
    continuity_match = re.search(r"═+\s*CONTINUITY STATE.*", req.full_text, re.DOTALL)
    continuity_block = ""
    prose_text = req.full_text

    if continuity_match:
        continuity_block = continuity_match.group(0).strip()
        prose_text = req.full_text[:continuity_match.start()].strip()

    cap_filename = f"Capitulo {req.chapter_num}.txt"
    commit_to_github(
        req.github_token, req.repo_owner, req.repo_name, 
        f"canon/{cap_filename}", prose_text, 
        f"Cierre automático: {cap_filename}"
    )

    os.makedirs("canon", exist_ok=True)
    with open(f"canon/{cap_filename}", "w", encoding="utf-8") as f:
        f.write(prose_text)

    if continuity_block and os.path.exists("main_instructions.txt"):
        with open("main_instructions.txt", "r", encoding="utf-8") as f:
            current_main = f.read()

        pattern = r"(## 15\. ESTADO ACTUAL DE CONTINUIDAD.*?)(?=## 16\. CONFLICTO TERRITORIAL|$)"
        replacement = f"## 15. ESTADO ACTUAL DE CONTINUIDAD\n\n```\n{continuity_block}\n```\n\n---\n\n"
        new_main = re.sub(pattern, replacement, current_main, flags=re.DOTALL)
        
        commit_to_github(
            req.github_token, req.repo_owner, req.repo_name,
            "main_instructions.txt", new_main,
            f"Cierre automático: Continuidad Cap {req.chapter_num}"
        )
        with open("main_instructions.txt", "w", encoding="utf-8") as f:
            f.write(new_main)

    # Limpiar sesión activa al archivar
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

    return {
        "status": "success",
        "message": f"Capítulo {req.chapter_num} archivado como '{cap_filename}' y Sección 15 actualizada."
    }

@app.post("/api/chat")
def chat(req: ChatRequest):
    client, _ = get_client(req.api_key)
    requested_model = req.model or DEFAULT_MODEL
    main_prompt = get_main_prompt()
    
    dynamic_canon = get_recent_and_queried_canon(req.message)
    system_instruction_full = f"{main_prompt}\n\n{dynamic_canon}"

    fallback_chain = [
        requested_model,
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro-preview"
    ]
    models_to_try = list(dict.fromkeys(fallback_chain))

    last_error = ""
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=req.message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction_full
                    )
                )
                return {"response": response.text, "model_used": model_name}
            except Exception as e:
                last_error = str(e)
                if "503" in last_error or "UNAVAILABLE" in last_error:
                    time.sleep(1.5)
                    continue
                break

    raise HTTPException(
        status_code=500, 
        detail=f"Servidores de Google saturados en Free Tier: {last_error}"
    )
