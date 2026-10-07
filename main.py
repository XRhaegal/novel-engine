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
EMBED_MODEL = "gemini-embedding-001"

SESSION_FILE = "active_session.json"          # Historial de burbujas para la UI (sync entre dispositivos) — NO se toca.
CHAT_HISTORY_FILE = "chat_history.json"       # Historial real que se manda al modelo para generar (nuevo).
INDEX_FILE = "canon_index.json"               # Índice de fragmentos + embeddings de todo el canon (nuevo).
ECONOMY_FILE = "economy_state.json"           # Saldo real, calculado por Python, nunca por el modelo (nuevo).

# Documentos de apoyo que SIEMPRE van completos (no pasan por embeddings, son chicos y estables).
STATIC_DOCS = ["Resumenes_por_Capitulo.md", "Ficha_Voz_Personajes.md", "Biblia_Continuidad.md"]

TOP_K_SEMANTIC = 6          # Cuántos fragmentos antiguos relevantes se traen por significado.
CHUNK_MAX_CHARS = 900       # Tamaño aproximado de cada fragmento al trocear un capítulo.

# Nombres del elenco, para el linter de vocativos de relleno.
CAST_NAMES = [
    "Edson", "Iselin", "Ise", "Claire", "Grace", "Rebecca", "Sabine", "Kirra",
    "Klara", "Jill", "Leon", "Chris", "Jamie", "Ryo", "Liam", "Barry",
    "Ashley", "Miller", "Arthur", "Martha", "Walt", "Estrada"
]

ECONOMY_TAG_RE = re.compile(
    r"<<ECONOMIA:\s*tipo=(gasto|ingreso)\s+monto=([\d.]+)\s+origen=(mano|maleta)(?:\s+nota=\"([^\"]*)\")?\s*>>",
    re.IGNORECASE
)


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


class RebuildIndexRequest(BaseModel):
    github_token: str
    repo_owner: str
    repo_name: str
    api_key: Optional[str] = None


# ----------------------------------------------------------------------
# Utilidades básicas (cliente Gemini, Main prompt, GitHub)
# ----------------------------------------------------------------------

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


def get_static_docs_block() -> str:
    """Resúmenes, ficha de voz y Biblia de continuidad: siempre completos,
    nunca pasan por la búsqueda semántica porque son chicos y cambian poco."""
    blocks = []
    for fname in STATIC_DOCS:
        if os.path.exists(fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    blocks.append(f"=== [{fname}] ===\n{content}\n")
            except Exception as e:
                print(f"Error leyendo {fname}: {e}")
    return "\n".join(blocks)


def extract_chapter_number(filepath: str) -> int:
    match = re.search(r'(\d+)', os.path.basename(filepath))
    return int(match.group(1)) if match else 0


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


# ----------------------------------------------------------------------
# Historial de conversación real (lo que se manda al modelo para generar)
# ----------------------------------------------------------------------

def load_chat_history() -> List[Dict[str, str]]:
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_chat_history(history: List[Dict[str, str]]):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)


def history_to_contents(history: List[Dict[str, str]], new_message: str):
    """Convierte el historial guardado en el formato que pide la API de Gemini."""
    contents = []
    for turn in history:
        contents.append(
            types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        )
    contents.append(types.Content(role="user", parts=[types.Part(text=new_message)]))
    return contents


# ----------------------------------------------------------------------
# Embeddings y búsqueda semántica (reemplaza la búsqueda por palabras sueltas)
# ----------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """Trocea un capítulo en fragmentos por párrafo, agrupando párrafos cortos
    hasta llegar a un tamaño razonable, sin cortar frases a la mitad."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buffer = ""
    for p in paragraphs:
        if len(buffer) + len(p) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{p}" if buffer else p
        else:
            if buffer:
                chunks.append(buffer)
            buffer = p
    if buffer:
        chunks.append(buffer)
    return chunks


def embed_texts(client, texts: List[str]) -> List[List[float]]:
    """Pide a Gemini la 'huella digital' de cada texto. Se llama poco
    (solo al cerrar capítulo o al reconstruir el índice), así que un
    loop simple es suficiente — no hace falta optimizar velocidad aquí."""
    vectors = []
    for t in texts:
        if not t.strip():
            vectors.append([])
            continue
        result = client.models.embed_content(model=EMBED_MODEL, contents=t)
        vectors.append(result.embeddings[0].values)
    return vectors


def embed_query(client, text: str) -> List[float]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_index() -> List[Dict[str, Any]]:
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_index(index: List[Dict[str, Any]]):
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)


def get_canon_files_sorted() -> List[str]:
    files = glob.glob("canon/**/*.*", recursive=True)
    valid_files = [f for f in files if os.path.isfile(f) and not f.endswith(".py")]
    valid_files.sort(key=extract_chapter_number)
    return valid_files


def get_recent_and_relevant_canon(client, query: str) -> str:
    """Reemplazo directo de la función vieja de palabras sueltas.
    - Los últimos 2 capítulos van completos, siempre (fidelidad de voz/tono).
    - Todo lo demás se busca por significado real contra el índice de embeddings."""
    valid_files = get_canon_files_sorted()
    if not valid_files:
        return ""

    recent_files = valid_files[-2:] if len(valid_files) >= 2 else valid_files
    recent_names = {os.path.basename(f) for f in recent_files}

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

    # Documentos de apoyo siempre completos (Resúmenes, Ficha de Voz, Biblia).
    static_block = get_static_docs_block()
    if static_block:
        canon_blocks.insert(0, static_block)

    index = load_index()
    # Solo buscamos en fragmentos que NO pertenezcan a los capítulos recientes
    # (esos ya van completos arriba, no hace falta repetirlos).
    searchable = [entry for entry in index if entry.get("file") not in recent_names]

    if searchable:
        try:
            query_vec = embed_query(client, query)
            scored = []
            for entry in searchable:
                score = cosine_sim(query_vec, entry.get("embedding", []))
                scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:TOP_K_SEMANTIC]
            if top:
                snippets = [f"[{e['file']}] (relevancia {s:.2f}): {e['text']}" for s, e in top]
                canon_blocks.append(
                    "=== [FRAGMENTOS RELEVANTES RECUPERADOS POR SIGNIFICADO] ===\n"
                    + "\n---\n".join(snippets) + "\n"
                )
        except Exception as e:
            print(f"Error en búsqueda semántica: {e}")

    return "\n".join(canon_blocks)


def add_chapter_to_index(client, filename: str, chapter_text: str):
    """Trocea y embebe un capítulo recién cerrado, y lo agrega al índice existente."""
    index = load_index()
    # Si por alguna razón ya existían fragmentos de este archivo, los quitamos primero
    # (evita duplicados si se vuelve a cerrar el mismo capítulo).
    index = [e for e in index if e.get("file") != filename]

    chunks = chunk_text(chapter_text)
    if not chunks:
        save_index(index)
        return index

    vectors = embed_texts(client, chunks)
    for chunk, vector in zip(chunks, vectors):
        if vector:
            index.append({"file": filename, "text": chunk, "embedding": vector})

    save_index(index)
    return index


# ----------------------------------------------------------------------
# Economía exacta (el servidor hace la suma, el modelo solo reporta)
# ----------------------------------------------------------------------

def load_economy() -> Dict[str, float]:
    if os.path.exists(ECONOMY_FILE):
        try:
            with open(ECONOMY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"mano": float(data.get("mano", 0)), "maleta": float(data.get("maleta", 0))}
        except Exception:
            pass
    return {"mano": 0.0, "maleta": 0.0}


def save_economy(state: Dict[str, float]):
    with open(ECONOMY_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def apply_economy_tags(text: str) -> (str, Dict[str, float], List[str]):
    """Busca etiquetas <<ECONOMIA: ...>> en la respuesta del modelo, aplica
    la suma/resta real en Python (nunca confiando en la aritmética del
    modelo), las borra del texto visible, y regresa el texto limpio +
    el nuevo saldo + una lista de movimientos aplicados (para log/debug)."""
    state = load_economy()
    movements = []

    for match in ECONOMY_TAG_RE.finditer(text):
        tipo, monto_str, origen, nota = match.groups()
        monto = float(monto_str)
        if tipo.lower() == "gasto":
            state[origen] = round(state[origen] - monto, 2)
        else:
            state[origen] = round(state[origen] + monto, 2)
        movements.append(f"{tipo} ${monto:.2f} ({origen}){' - ' + nota if nota else ''}")

    clean_text = ECONOMY_TAG_RE.sub("", text).rstrip()
    if movements:
        save_economy(state)
    return clean_text, state, movements


def economy_ground_truth_block() -> str:
    state = load_economy()
    total = round(state["mano"] + state["maleta"], 2)
    return (
        "## ESTADO ECONÓMICO REAL (verdad absoluta, calculada por el sistema — "
        "NUNCA asumas ni recalcules otro número, usa exactamente este)\n"
        f"Efectivo en mano: ${state['mano']:.2f}\n"
        f"Reserva en maleta: ${state['maleta']:.2f}\n"
        f"Total: ${total:.2f}\n\n"
        "Cada vez que Edson gaste, cobre o reciba dinero en la escena, al FINAL de tu "
        "respuesta (después de toda la prosa, invisible para el lector) agrega una línea "
        "exacta con este formato, una por cada movimiento:\n"
        "<<ECONOMIA: tipo=gasto monto=12.50 origen=mano nota=\"cena en Martha's\">>\n"
        "<<ECONOMIA: tipo=ingreso monto=35.00 origen=mano nota=\"medio turno con Iselin\">>\n"
        "tipo es 'gasto' o 'ingreso'; origen es 'mano' o 'maleta'. No hagas tú la resta en la "
        "prosa ni anuncies el saldo nuevo en el texto narrativo — el sistema la calcula. Si no "
        "hubo movimiento de dinero en el turno, no agregues ninguna etiqueta."
    )


@app.post("/api/economy/adjust")
def adjust_economy(mano: Optional[float] = None, maleta: Optional[float] = None):
    """Ajuste manual de emergencia, por si el saldo se desincroniza y hay
    que corregirlo a mano desde fuera del flujo narrativo."""
    state = load_economy()
    if mano is not None:
        state["mano"] = round(mano, 2)
    if maleta is not None:
        state["maleta"] = round(maleta, 2)
    save_economy(state)
    return {"status": "updated", "state": state}


@app.get("/api/economy")
def get_economy():
    state = load_economy()
    return {"state": state, "total": round(state["mano"] + state["maleta"], 2)}


# ----------------------------------------------------------------------
# Linter de estilo (reglas duras del Main, revisadas por código, no por IA)
# ----------------------------------------------------------------------

def lint_response(text: str) -> List[str]:
    warnings = []

    # 1) Vocativo de relleno al final de una línea de diálogo: ", Nombre." o ", Nombre?" etc.
    for name in CAST_NAMES:
        pattern = re.compile(rf",\s*{re.escape(name)}[.!?]", re.IGNORECASE)
        for m in pattern.finditer(text):
            snippet = text[max(0, m.start() - 40):m.end()]
            warnings.append(f"Posible vocativo de relleno: \"...{snippet}\"")

    # 2) Em-dash cortando a mitad de frase: —texto— dentro de la misma línea de diálogo.
    for m in re.finditer(r"—[^—\n]{2,80}—", text):
        warnings.append(f"Posible em-dash a mitad de frase: \"{m.group(0)}\"")

    # 3) Palabras de la lista negra (Sección 2 del Main).
    blacklist = ["fierros", "sobremesa"]
    for word in blacklist:
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            warnings.append(f"Palabra vetada en la lista negra: \"{word}\"")

    return warnings


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_home():
    if os.path.exists("templates/index.html"):
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Cedar Creek Engine Activo</h1>"


# --- PERSISTENCIA DE SESIÓN (burbujas de la UI, sync entre dispositivos) ---
# Sin cambios respecto a la versión anterior — el frontend sigue funcionando igual.

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
    # Al limpiar la sesión de la UI, también reiniciamos el historial real de generación.
    clear_chat_history()
    return {"status": "cleared"}


@app.post("/api/sync-cache")
def sync_cache(payload: Optional[ChatRequest] = None):
    valid_files = get_canon_files_sorted()
    last_file = os.path.basename(valid_files[-1]) if valid_files else "Ninguno"
    index = load_index()
    return {
        "status": "success",
        "total_files": len(valid_files),
        "indexed_fragments": len(index),
        "latest_chapter": last_file,
        "message": f"Canon validado: {len(valid_files)} archivos, {len(index)} fragmentos indexados. Último: {last_file}."
    }


@app.post("/api/rebuild-index")
def rebuild_index(req: RebuildIndexRequest):
    """Llamar UNA SOLA VEZ para construir el índice de embeddings a partir
    de todos los capítulos que ya existen en canon/. De ahí en adelante,
    auto-close-chapter lo va actualizando solo, capítulo por capítulo."""
    client, _ = get_client(req.api_key)
    valid_files = get_canon_files_sorted()
    if not valid_files:
        return {"status": "success", "message": "No hay archivos en canon/ todavía.", "fragments": 0}

    full_index = []
    for path in valid_files:
        fname = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content)
        if not chunks:
            continue
        vectors = embed_texts(client, chunks)
        for chunk, vector in zip(chunks, vectors):
            if vector:
                full_index.append({"file": fname, "text": chunk, "embedding": vector})

    save_index(full_index)
    commit_to_github(
        req.github_token, req.repo_owner, req.repo_name,
        INDEX_FILE, json.dumps(full_index, ensure_ascii=False),
        "Reconstrucción completa del índice de embeddings"
    )
    return {
        "status": "success",
        "message": f"Índice reconstruido: {len(valid_files)} capítulos, {len(full_index)} fragmentos.",
        "fragments": len(full_index)
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

    # Actualizar el índice de embeddings con el capítulo recién cerrado.
    try:
        client, _ = get_client(req.api_key)
        updated_index = add_chapter_to_index(client, cap_filename, prose_text)
        commit_to_github(
            req.github_token, req.repo_owner, req.repo_name,
            INDEX_FILE, json.dumps(updated_index, ensure_ascii=False),
            f"Índice actualizado: {cap_filename}"
        )
    except Exception as e:
        print(f"Aviso: no se pudo actualizar el índice de embeddings: {e}")

    # Respaldar el saldo económico real en GitHub (sobrevive a reinicios de Render).
    try:
        economy_state = load_economy()
        commit_to_github(
            req.github_token, req.repo_owner, req.repo_name,
            ECONOMY_FILE, json.dumps(economy_state, ensure_ascii=False),
            f"Respaldo de saldo al cerrar Cap {req.chapter_num}"
        )
    except Exception as e:
        print(f"Aviso: no se pudo respaldar el saldo: {e}")

    # Limpiar sesión activa e historial de generación al archivar el capítulo.
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    clear_chat_history()

    return {
        "status": "success",
        "message": f"Capítulo {req.chapter_num} archivado como '{cap_filename}', Sección 15 e índice actualizados."
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    client, _ = get_client(req.api_key)
    requested_model = req.model or DEFAULT_MODEL

    main_prompt = get_main_prompt()
    dynamic_canon = get_recent_and_relevant_canon(client, req.message)
    economy_block = economy_ground_truth_block()
    system_instruction_full = f"{main_prompt}\n\n{economy_block}\n\n{dynamic_canon}"

    history = load_chat_history()
    contents = history_to_contents(history, req.message)

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
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction_full
                    )
                )
                clean_text, new_economy_state, movements = apply_economy_tags(response.text)
                warnings = lint_response(clean_text)

                # Guardar el turno en el historial real de generación (ya sin la etiqueta de economía).
                history.append({"role": "user", "text": req.message})
                history.append({"role": "model", "text": clean_text})
                save_chat_history(history)

                return {
                    "response": clean_text,
                    "model_used": model_name,
                    "economy": new_economy_state,
                    "economy_movements": movements,
                    "lint_warnings": warnings
                }
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
