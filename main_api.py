import asyncio
import ast
import gzip
import hashlib
import json
import os
import re
import zlib
from functools import lru_cache
from typing import Optional
from groq import Groq
import chromadb
import httpx
from bson.codec_options import CodecOptions
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
import google.generativeai as genai

print("RUNNING FILE:", __file__)
print("CWD:", os.getcwd())

load_dotenv()

# =========================================================
# ENV CONFIG
# =========================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "legaldb")
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:70b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_legal")
os.makedirs(CHROMA_PATH, exist_ok=True)

# performance knobs (не променят логиката, само изпълнението)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "180"))
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
ES_BULK_BATCH = int(os.getenv("ES_BULK_BATCH", "200"))
CHROMA_QUERY_TOP_K = int(os.getenv("CHROMA_QUERY_TOP_K", "12"))

# =========================================================
# DB + SEARCH CONFIG
# =========================================================
BASE_COLLECTIONS = [
    "constitution",
    "codex",
    "laws",
    "implementableRegulations",
    "regulations",
    "rules",
]

EU_COLLECTIONS = [
    "regulations-eu-bg",
    "regulations-eu-en",
    "directives-eu-bg",
    "directives-eu-en",
    "agreement-eu-bg",
    "agreement-eu-en",
]

MONGO_COLLECTIONS = BASE_COLLECTIONS + EU_COLLECTIONS

MAX_RETRIES = 5
VECTOR_DOC_LIMIT = 12
PER_VECTOR_DOC_BLOCK_LIMIT = 6
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client.get_database(
    MONGO_DB,
    codec_options=CodecOptions(unicode_decode_error_handler="ignore"),
)

es = Elasticsearch(
    ES_URL,
    verify_certs=False,
    request_timeout=30,
    retry_on_timeout=True,
    max_retries=2,
    http_compress=True,
)

# =========================================================
# CHROMA INIT
# =========================================================
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(
    name="legal_docs",
    metadata={"hnsw:space": "cosine"},
)

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI(
    title="JusticIA API",
    description="Legal AI Assistant API (Mongo + Elasticsearch + Chroma + Ollama)",
    version="4.3.1-fast",
)

# =========================================================
# GLOBAL ASYNC HTTP CLIENT
# =========================================================
http_client: Optional[httpx.AsyncClient] = None

# in-memory caches
OLLAMA_CHAT_CACHE: dict[str, str] = {}
OLLAMA_EMBED_CACHE: dict[str, list[float]] = {}
BLOCK_SPLIT_CACHE: dict[str, list[str]] = {}
EXTRACT_TERM_CACHE: dict[str, tuple[str | None, list[str]]] = {}
DSL_CACHE: dict[str, dict] = {}

# =========================================================
# REGEX / PRECOMPILED
# =========================================================
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINES_RE = re.compile(r"\n{3,}")
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

ARTICLE_PATTERNS = [
    re.compile(r"(Чл\.\s*\d+[а-яА-Яa-zA-Z0-9\-]*)", flags=re.IGNORECASE),
    re.compile(r"(Член\s*\d+[а-яА-Яa-zA-Z0-9\-]*)", flags=re.IGNORECASE),
    re.compile(r"(Article\s*\d+[a-zA-Z0-9\-]*)", flags=re.IGNORECASE),
]

# =========================================================
# COLLECTION DESCRIPTIONS
# =========================================================
COLLECTION_DESCRIPTIONS = {
    "constitution": "Конституцията на Република България.",
    "codex": "Кодексите (напр. НК, НПК, КТ и др.).",
    "laws": "Законите.",
    "implementableRegulations": "Правилници по прилагане.",
    "regulations": "Правилници.",
    "rules": "Наредби.",
    "regulations-eu-bg": "Европейските регламенти на български език.",
    "regulations-eu-en": "European Union regulations in English.",
    "directives-eu-bg": "Директивите на Европейския съюз на български език.",
    "directives-eu-en": "European Union directives in English.",
    "agreement-eu-bg": (
        "Договорите за ЕС представляват обвързващи споразумения между страните. "
        "Включват: Договор за Европейския съюз, Договор за функционирането на ЕС, "
        "Договор за Евратом, Харта на основните права на ЕС."
    ),
    "agreement-eu-en": (
        "EU treaties are binding agreements between the parties. Includes: "
        "Treaty on European Union, Treaty on the Functioning of the EU, "
        "Euratom Treaty, Charter of Fundamental Rights of the EU."
    ),
}


# =========================================================
# REQUEST MODEL
# =========================================================
class GenerateRequest(BaseModel):
    question: str
    options: str = Field(default="all", pattern="^(all|bg|en|old)$")
    conversation_id: Optional[str] = "default"


# =========================================================
# APP LIFECYCLE
# =========================================================
@app.on_event("startup")
async def startup():
    global http_client
    timeout = httpx.Timeout(
        timeout=HTTP_TIMEOUT,
        connect=HTTP_CONNECT_TIMEOUT,
        read=HTTP_READ_TIMEOUT,
        write=HTTP_TIMEOUT,
        pool=HTTP_TIMEOUT,
    )
    http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        http2=False,
    )

    print("🚀 Startup indexing check...")

    try:
        if not elastic_has_any_data():
            print("📦 Elasticsearch is empty -> indexing...")
            index_mongo_to_es()
        else:
            print("✅ Elasticsearch already has data -> skipping")

        if not chroma_has_any_data():
            print("📦 Chroma is empty -> indexing...")
            await index_mongo_to_chroma()
        else:
            print("✅ Chroma already has data -> skipping")

    except Exception as e:
        print("⚠️ Startup indexing failed:", e)


@app.on_event("shutdown")
async def shutdown():
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None


# =========================================================
# HELPERS
# =========================================================
def normalize_to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return " ".join(normalize_to_str(i) for i in x if i is not None)
    if isinstance(x, dict):
        for k in ("term", "value", "text", "name", "query"):
            if k in x:
                return normalize_to_str(x[k])
        return " ".join(normalize_to_str(v) for v in x.values() if v is not None)
    return str(x)


def clean_for_embedding(text: str) -> str:
    if text is None:
        return ""

    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="ignore")
        except Exception:
            text = str(text)

    text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINES_RE.sub("\n\n", text)
    return text.strip()


def safe_text(x):
    if x is None:
        return ""

    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8")
        except Exception:
            pass

        if isinstance(x, bytes):
            try:
                x = x.decode("cp1251")
            except Exception:
                pass

        if isinstance(x, bytes):
            try:
                x = gzip.decompress(x).decode("utf-8", errors="ignore")
            except Exception:
                pass

        if isinstance(x, bytes):
            try:
                x = zlib.decompress(x).decode("utf-8", errors="ignore")
            except Exception:
                pass

        if isinstance(x, bytes):
            x = x.decode("latin-1", errors="ignore")
    else:
        x = normalize_to_str(x)

    return clean_for_embedding(x)


def safe_parse_json(output: str):
    if not output:
        return None
    m = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def normalize_collection_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [normalize_to_str(i) for i in x]
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [normalize_to_str(i) for i in parsed]
        except Exception:
            pass
        return [x]
    return [normalize_to_str(x)]


def output_language(options: str) -> str:
    return "en" if (options or "").strip().lower() == "en" else "bg"


def allowed_collections_for_options(options: str):
    options = (options or "all").strip().lower()

    if options == "old":
        return BASE_COLLECTIONS
    if options == "bg":
        return ["regulations-eu-bg", "directives-eu-bg"]
    if options == "en":
        return ["regulations-eu-en", "directives-eu-en"]

    return BASE_COLLECTIONS + ["regulations-eu-bg", "directives-eu-bg"]


def agreements_index_for_options(options: str):
    options = (options or "all").strip().lower()
    return "agreement-eu-en" if options == "en" else "agreement-eu-bg"


def is_eu_regs_or_directives(indices: list[str]):
    indices_l = [i.lower() for i in indices if isinstance(i, str)]
    return any(
        idx in {
            "regulations-eu-bg",
            "directives-eu-bg",
            "regulations-eu-en",
            "directives-eu-en",
        }
        for idx in indices_l
    )


# =========================================================
# TOKEN / BLOCK UTILS
# =========================================================
@lru_cache(maxsize=50000)
def tokenize_cached(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall((text or "").lower()))


def tokenize(text: str) -> list[str]:
    return list(tokenize_cached(text or ""))


@lru_cache(maxsize=50000)
def extract_article_label(block: str) -> str:
    if not block:
        return ""
    for pattern in ARTICLE_PATTERNS:
        m = pattern.search(block)
        if m:
            return m.group(1).strip()
    return ""


def _block_cache_key(description: str, header_regex: str) -> str:
    return hashlib.md5(f"{header_regex}::{description}".encode("utf-8", errors="ignore")).hexdigest()


def _extract_blocks(description: str, header_regex: str) -> list[str]:
    if not description:
        return []

    cache_key = _block_cache_key(description, header_regex)
    cached = BLOCK_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pattern = rf"({header_regex}.*?)(?={header_regex}|$)"
    blocks = [m.strip() for m in re.findall(pattern, description, flags=re.DOTALL | re.IGNORECASE)]
    BLOCK_SPLIT_CACHE[cache_key] = blocks
    return blocks


def split_into_blocks(description: str, collection: str) -> list[str]:
    collection = (collection or "").lower()

    if collection in {"regulations-eu-bg", "directives-eu-bg", "agreement-eu-bg"}:
        blocks = _extract_blocks(description, r"Член\s*\d+[^\n]*")
        if not blocks:
            blocks = _extract_blocks(description, r"Чл\.\s*\d+[^\n]*")
        return blocks

    if collection in {"regulations-eu-en", "directives-eu-en", "agreement-eu-en"}:
        return _extract_blocks(description, r"Article\s*\d+[^\n]*")

    return _extract_blocks(description, r"Чл\.\s*\d+[^\n]*")


def extract_context_chl(description: str, term: str) -> list[str]:
    term_l = term.lower()
    return [b for b in _extract_blocks(description, r"Чл\.\s*\d+[^\n]*") if term_l in b.lower()]


def extract_context_chlen(description: str, term: str) -> list[str]:
    term_l = term.lower()
    blocks = _extract_blocks(description, r"Член\s*\d+[^\n]*")
    if not blocks:
        blocks = _extract_blocks(description, r"Чл\.\s*\d+[^\n]*")
    return [b for b in blocks if term_l in b.lower()]


def extract_context_article(description: str, term: str) -> list[str]:
    term_l = term.lower()
    return [b for b in _extract_blocks(description, r"Article\s*\d+[^\n]*") if term_l in b.lower()]


def pick_extractor_by_index(index_name: str):
    idx = (index_name or "").lower()
    if idx in {"regulations-eu-bg", "directives-eu-bg", "agreement-eu-bg"}:
        return extract_context_chlen
    if idx in {"regulations-eu-en", "directives-eu-en", "agreement-eu-en"}:
        return extract_context_article
    return extract_context_chl


# =========================================================
# OLLAMA API
# =========================================================
def _chat_cache_key(messages: list[dict], temperature: float) -> str:
    raw = json.dumps(
        {
            "model": OLLAMA_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "num_ctx": OLLAMA_NUM_CTX,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


async def groq_final_summary(prompt: str) -> str:
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Groq final summary failed: {e}")
        raise RuntimeError("Groq final summary failed") from e


async def ask_gemini_summary(prompt: str) -> str:
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned empty response")
        return text.strip()
    except Exception as e:
        print(f"❌ Gemini final summary failed: {e}")
        raise RuntimeError("Gemini final summary failed") from e


async def ollama_chat(messages: list[dict], temperature: float = 0.1) -> str:
    global http_client
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized")

    cache_key = _chat_cache_key(messages, temperature)
    if cache_key in OLLAMA_CHAT_CACHE:
        return OLLAMA_CHAT_CACHE[cache_key]

    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    try:
        r = await http_client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        content = data["message"]["content"].strip()
        OLLAMA_CHAT_CACHE[cache_key] = content
        return content
    except httpx.HTTPStatusError as e:
        body = ""
        try:
            body = e.response.text[:1000]
        except Exception:
            pass
        print(f"❌ Ollama HTTP error: {e} | body={body}")
        raise RuntimeError(f"Ollama chat failed with status {e.response.status_code}") from e
    except Exception as e:
        print(f"❌ Ollama chat failed: {e}")
        raise RuntimeError("Ollama chat failed") from e


async def ollama_embed_texts(texts: list[str]) -> list[list[float]]:
    global http_client
    if http_client is None:
        raise RuntimeError("HTTP client is not initialized")

    cleaned = [clean_for_embedding(t) for t in texts]

    results: list[Optional[list[float]]] = [None] * len(cleaned)
    uncached_indexes = []
    uncached_texts = []

    for i, text in enumerate(cleaned):
        key = hashlib.md5(f"{OLLAMA_EMBED_MODEL}::{text}".encode("utf-8")).hexdigest()
        cached = OLLAMA_EMBED_CACHE.get(key)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indexes.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        for start in range(0, len(uncached_texts), EMBED_BATCH_SIZE):
            batch = uncached_texts[start:start + EMBED_BATCH_SIZE]
            payload = {
                "model": OLLAMA_EMBED_MODEL,
                "input": batch,
            }

            try:
                r = await http_client.post(f"{OLLAMA_HOST}/api/embed", json=payload)
                r.raise_for_status()
                data = r.json()
                batch_embeddings = data["embeddings"]
            except Exception as e:
                print(f"❌ Ollama embed batch failed: {e}")
                raise RuntimeError("Ollama embedding failed") from e

            batch_indices = uncached_indexes[start:start + EMBED_BATCH_SIZE]
            for idx, text, emb in zip(batch_indices, batch, batch_embeddings):
                key = hashlib.md5(f"{OLLAMA_EMBED_MODEL}::{text}".encode("utf-8")).hexdigest()
                OLLAMA_EMBED_CACHE[key] = emb
                results[idx] = emb

    return [r for r in results if r is not None]


# =========================================================
# INDEX CHECKERS
# =========================================================
def elastic_has_any_data() -> bool:
    try:
        for idx in MONGO_COLLECTIONS:
            if es.indices.exists(index=idx.lower()):
                if es.count(index=idx.lower())["count"] > 0:
                    return True
        return False
    except Exception as e:
        print("⚠️ elastic_has_any_data error:", e)
        return False


def chroma_has_any_data() -> bool:
    try:
        return chroma_collection.count() > 0
    except Exception as e:
        print("⚠️ chroma_has_any_data error:", e)
        return False


# =========================================================
# ES INDEXING
# =========================================================
def index_mongo_to_es():
    for coll_name in MONGO_COLLECTIONS:
        index_name = coll_name.lower()
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
                print(f"🧹 Deleted old index '{index_name}' before reindexing.")
        except Exception as e:
            print(f"⚠️ Could not delete index '{index_name}': {e}")

    for coll_name in MONGO_COLLECTIONS:
        collection = mongo_db[coll_name]
        actions = []
        indexed = 0
        skipped = 0

        print(f"🚀 Indexing collection: {coll_name}")

        try:
            cursor = collection.find({}, {"title": 1, "description": 1}, no_cursor_timeout=True).batch_size(200)
            for doc in cursor:
                try:
                    doc_id = str(doc.get("_id"))
                    cleaned_doc = {
                        "title": safe_text(doc.get("title", "")),
                        "description": safe_text(doc.get("description", "")),
                    }

                    actions.append({
                        "_index": coll_name.lower(),
                        "_id": doc_id,
                        "_source": cleaned_doc,
                    })

                    if len(actions) >= ES_BULK_BATCH:
                        helpers.bulk(es, actions, raise_on_error=False, request_timeout=60)
                        indexed += len(actions)
                        actions = []
                except Exception as e:
                    skipped += 1
                    print(f"⚠️ Skipped ES doc in {coll_name}: {e}")
        except Exception as e:
            print(f"⚠️ Cursor failed in {coll_name}: {e}")

        if actions:
            try:
                helpers.bulk(es, actions, raise_on_error=False, request_timeout=60)
                indexed += len(actions)
            except Exception as e:
                print(f"⚠️ Final ES bulk failed in {coll_name}: {e}")

        print(f"✅ Indexed {indexed} docs from {coll_name} | Skipped: {skipped}")

    print("🎯 MongoDB -> Elasticsearch finished.")


# =========================================================
# CHROMA INDEXING (FAIL-SAFE)
# =========================================================
async def index_mongo_to_chroma():
    print("🚀 Indexing MongoDB -> Chroma")

    try:
        chroma_client.delete_collection("legal_docs")
    except Exception:
        pass

    global chroma_collection
    chroma_collection = chroma_client.get_or_create_collection(
        name="legal_docs",
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    skipped = 0

    batch_ids = []
    batch_docs = []
    batch_metas = []

    async def flush_batch(batch_ids, batch_docs, batch_metas, coll_name):
        nonlocal total, skipped

        if not batch_docs:
            return [], [], []

        try:
            cleaned_docs = [clean_for_embedding(d) for d in batch_docs]
            embeddings = await ollama_embed_texts(cleaned_docs)
            chroma_collection.add(
                ids=batch_ids,
                documents=cleaned_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            total += len(cleaned_docs)
            print(f"✅ Chroma batch indexed: {total}")
            return [], [], []

        except Exception as e:
            print(f"⚠️ Batch failed in {coll_name}: {e}")
            print("🔁 Falling back to one-by-one indexing...")

            for single_id, single_doc, single_meta in zip(batch_ids, batch_docs, batch_metas):
                try:
                    cleaned_doc = clean_for_embedding(single_doc)
                    single_embedding = await ollama_embed_texts([cleaned_doc])
                    chroma_collection.add(
                        ids=[single_id],
                        documents=[cleaned_doc],
                        embeddings=single_embedding,
                        metadatas=[single_meta],
                    )
                    total += 1
                    print(f"✅ Single doc indexed: {single_id}")
                except Exception as inner_e:
                    skipped += 1
                    print(
                        f"❌ Skipped broken doc | collection={coll_name} | "
                        f"mongo_id={single_id} | title={single_meta.get('title', '')[:120]} | error={inner_e}"
                    )

            return [], [], []

    for coll_name in MONGO_COLLECTIONS:
        collection = mongo_db[coll_name]
        print(f"📚 Reading collection for Chroma: {coll_name}")

        try:
            cursor = collection.find({}, {"title": 1, "description": 1}, no_cursor_timeout=True).batch_size(200)
            for doc in cursor:
                try:
                    mongo_id = str(doc.get("_id"))
                    title = safe_text(doc.get("title", ""))
                    description = safe_text(doc.get("description", ""))

                    if not description.strip():
                        skipped += 1
                        print(f"⚠️ Empty description skipped | collection={coll_name} | mongo_id={mongo_id}")
                        continue

                    language = "en" if coll_name.endswith("-en") else "bg"

                    batch_ids.append(mongo_id)
                    batch_docs.append(description)
                    batch_metas.append({
                        "mongo_id": mongo_id,
                        "title": title,
                        "collection": coll_name,
                        "language": language,
                    })

                    if len(batch_docs) >= EMBED_BATCH_SIZE:
                        batch_ids, batch_docs, batch_metas = await flush_batch(
                            batch_ids, batch_docs, batch_metas, coll_name
                        )

                except Exception as e:
                    skipped += 1
                    print(
                        f"❌ Failed preparing doc | collection={coll_name} | "
                        f"mongo_id={doc.get('_id')} | error={e}"
                    )

        except Exception as e:
            skipped += 1
            print(f"❌ Failed iterating collection {coll_name}: {e}")

        batch_ids, batch_docs, batch_metas = await flush_batch(
            batch_ids, batch_docs, batch_metas, coll_name
        )

    print(f"🎯 MongoDB -> Chroma finished. Total indexed: {total} | Skipped: {skipped}")


# =========================================================
# QUERY UNDERSTANDING
# =========================================================
async def extract_term_and_collection(question: str, allowed_collections: list[str], lang: str):
    cache_key = hashlib.md5(
        json.dumps(
            {"q": question, "allowed": allowed_collections, "lang": lang},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cached = EXTRACT_TERM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    allowed_payload = [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in allowed_collections]
    allowed_json = json.dumps(allowed_payload, ensure_ascii=False)

    if lang == "en":
        prompt = f"""
You are a legal routing assistant.

Allowed collections:
{allowed_json}

Question:
"{question}"

Return ONLY JSON:
{{
  "term": "main legal term",
  "collection": ["one_collection_from_allowed_list"]
}}
"""
    else:
        prompt = f"""
Ти си правен routing асистент.

Разрешени колекции:
{allowed_json}

Въпрос:
"{question}"

Върни САМО JSON:
{{
  "term": "основен правен термин",
  "collection": ["една_валидна_колекция"]
}}
"""

    output = await ollama_chat([{"role": "user", "content": prompt.strip()}], temperature=0.1)

    try:
        parsed = safe_parse_json(output)
        if not parsed:
            raise ValueError("No valid JSON found")

        term = normalize_to_str(parsed.get("term", "")).strip()
        collection = normalize_collection_list(parsed.get("collection", []))
        collection = [c.strip().lower() for c in collection if c and normalize_to_str(c).strip()]

        allowed_set = {c.lower() for c in allowed_collections}
        collection = [c for c in collection if c in allowed_set]

        result = (term, collection)
        EXTRACT_TERM_CACHE[cache_key] = result
        return result
    except Exception as e:
        print("⚠️ Failed to parse term/collection:", e)
        return None, []


def _find_matching_index(index_name: str, term: str) -> Optional[str]:
    if not index_name:
        return None
    try:
        res = es.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": term,
                        "fields": ["title^3", "description"],
                    }
                },
                "size": 1,
                "_source": False,
            },
        )
        if res.get("hits", {}).get("total", {}).get("value", 0) > 0:
            return index_name
    except Exception as e:
        print(f"⚠️ Error searching in index '{index_name}': {e}")
    return None


def find_matching_indices(term: str, indices: list[str]):
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, _find_matching_index, idx, term) for idx in indices if idx]
    if not tasks:
        return []

    done = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True)) if not loop.is_running() else None
    if done is not None:
        return [x for x in done if isinstance(x, str)]

    # fallback ако loop already runs
    matched = []
    for idx in indices:
        found = _find_matching_index(idx, term)
        if found:
            matched.append(found)
    return matched


async def find_matching_indices_async(term: str, indices: list[str]) -> list[str]:
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _find_matching_index, idx, term) for idx in indices if idx]
    if not tasks:
        return []
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [x for x in results if isinstance(x, str)]


async def generate_term_with_retries(question: str, allowed_cols: list[str], lang: str):
    failed_terms = []

    for _ in range(MAX_RETRIES):
        term, collection = await extract_term_and_collection(question, allowed_cols, lang=lang)
        if not isinstance(collection, list):
            collection = [collection]
        if not term or not collection:
            continue

        matched = await find_matching_indices_async(term, collection)
        if matched:
            return term, matched, failed_terms

        failed_terms.append(term)

    return None, [], failed_terms


# =========================================================
# DSL
# =========================================================
async def generate_detailed_dsl(question: str, term: str, indices: list[str], lang: str,
                                excluded_terms: list[str] = []):
    cache_key = hashlib.md5(
        json.dumps(
            {
                "question": question,
                "term": term,
                "indices": indices,
                "lang": lang,
                "excluded": excluded_terms,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cached = DSL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if lang == "en":
        prompt = f"""
Question: "{question}"
Current term: "{term}"

Generate Elasticsearch DSL JSON:
- multi_match on title^3 and description
- highlight description
- size 100

Return ONLY JSON.
"""
    else:
        prompt = f"""
Изходен въпрос: "{question}"
Текущ термин: "{term}"

Генерирай Elasticsearch DSL JSON:
- multi_match върху title^3 и description
- highlight на description
- size 100

Върни САМО JSON.
"""

    output = await ollama_chat([{"role": "user", "content": prompt.strip()}], temperature=0.1)

    try:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        dsl = json.loads(output[json_start:json_end])

        if "query" not in dsl:
            raise ValueError("Incomplete DSL")

        if "highlight" not in dsl:
            dsl["highlight"] = {
                "fields": {
                    "description": {
                        "number_of_fragments": 1000,
                        "fragment_size": 500,
                    }
                }
            }

        dsl["size"] = dsl.get("size", 100)
        DSL_CACHE[cache_key] = dsl
        return dsl
    except Exception:
        dsl = {
            "query": {"multi_match": {"query": term, "fields": ["title^3", "description"]}},
            "highlight": {
                "fields": {
                    "description": {
                        "number_of_fragments": 1000,
                        "fragment_size": 500,
                    }
                }
            },
            "size": 100,
        }
        DSL_CACHE[cache_key] = dsl
        return dsl


# =========================================================
# ES SEARCH + BLOCK EXTRACTION
# =========================================================
def extract_highlighted_article_blocks(description: str, highlighted_fragments: list[str], collection: str) -> list[
    str]:
    if not description or not highlighted_fragments:
        return []

    blocks = split_into_blocks(description, collection)
    if not blocks:
        return []

    results = []
    normalized_blocks = [(block, re.sub(r"\s+", " ", block).strip().lower()) for block in blocks]

    for _, clean_block in normalized_blocks:
        pass

    clean_frags = []
    for frag in highlighted_fragments:
        frag_clean = re.sub(r"</?em>", "", frag)
        frag_clean = re.sub(r"\s+", " ", frag_clean).strip().lower()
        if frag_clean:
            clean_frags.append(frag_clean)

    for block, clean_block in normalized_blocks:
        for frag_clean in clean_frags:
            if frag_clean in clean_block:
                results.append(block)
                break

    return list(dict.fromkeys(results))


def search_index_for_term(index_name: str, question: str, term: str, lang: str):
    dsl = {
        "query": {"multi_match": {"query": term, "fields": ["title^3", "description"]}},
        "highlight": {
            "fields": {
                "description": {
                    "number_of_fragments": 1000,
                    "fragment_size": 500,
                }
            }
        },
        "size": 100,
    }

    try:
        res = es.search(index=index_name, body=dsl)
        return res.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"⚠️ Failed searching prerequisite index '{index_name}': {e}")
        return []


# =========================================================
# VECTOR SEARCH
# =========================================================
@lru_cache(maxsize=50000)
def token_set(text: str) -> frozenset[str]:
    return frozenset(tokenize_cached(text or ""))


def block_score(block: str, standalone_question: str, legal_query: str, keywords: list[str], title: str = "") -> float:
    block_tokens = token_set(block)
    score = 0.0
    score += len(block_tokens & token_set(standalone_question)) * 2.0
    score += len(block_tokens & token_set(legal_query)) * 3.0
    score += len(block_tokens & token_set(" ".join(keywords))) * 2.5
    score += len(block_tokens & token_set(title)) * 0.5

    if legal_query and legal_query.lower() in block.lower():
        score += 8.0

    if 100 <= len(block) <= 2500:
        score += 1.0

    return score


async def search_chroma_docs(standalone_question: str, allowed_collections: list[str],
                             n_results: int = VECTOR_DOC_LIMIT) -> list[dict]:
    question_embedding = (await ollama_embed_texts([standalone_question]))[0]

    results = chroma_collection.query(
        query_embeddings=[question_embedding],
        n_results=min(n_results, CHROMA_QUERY_TOP_K),
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

    allowed_set = set(allowed_collections)
    out = []
    for doc_id, document, meta, distance in zip(ids, docs, metas, distances):
        collection = normalize_to_str(meta.get("collection"))
        if collection not in allowed_set:
            continue

        out.append({
            "vector_id": doc_id,
            "doc_id": normalize_to_str(meta.get("mongo_id")),
            "title": normalize_to_str(meta.get("title", "Без заглавие")),
            "collection": collection,
            "language": normalize_to_str(meta.get("language")),
            "description": normalize_to_str(document),
            "distance": distance,
        })
    return out


def extract_blocks_from_vector_docs(vector_docs: list[dict], standalone_question: str, legal_query: str,
                                    keywords: list[str], per_doc_top_n: int = PER_VECTOR_DOC_BLOCK_LIMIT) -> list[dict]:
    all_blocks = []

    for doc in vector_docs:
        description = doc.get("description", "")
        collection = doc.get("collection", "")
        title = doc.get("title", "")
        doc_id = doc.get("doc_id", "")

        blocks = split_into_blocks(description, collection)
        scored_blocks = []

        for block in blocks:
            scored_blocks.append({
                "source_types": ["vector"],
                "doc_id": doc_id,
                "title": title,
                "collection": collection,
                "article_label": extract_article_label(block),
                "content": block,
                "vector_distance": doc.get("distance"),
                "local_score": block_score(block, standalone_question, legal_query, keywords, title),
            })

        scored_blocks.sort(key=lambda x: x["local_score"], reverse=True)
        all_blocks.extend(scored_blocks[:per_doc_top_n])

    return all_blocks


def dedupe_blocks(blocks: list[dict]) -> list[dict]:
    merged = {}

    for b in blocks:
        key = (
            normalize_to_str(b.get("doc_id")),
            normalize_to_str(b.get("article_label")),
            hashlib.md5(normalize_to_str(b.get("content")).encode("utf-8")).hexdigest(),
        )

        if key not in merged:
            merged[key] = b
        else:
            existing = merged[key]
            existing["source_types"] = list(set(existing.get("source_types", []) + b.get("source_types", [])))
            existing["local_score"] = max(existing.get("local_score", 0), b.get("local_score", 0))

    return list(merged.values())


# =========================================================
# FINAL ANSWER
# =========================================================
async def final_answer_from_blocks(question: str, blocks: list[dict], lang: str):
    blocks_text = []
    for i, b in enumerate(blocks, start=1):
        blocks_text.append(
            f"""SOURCE {i}
TITLE: {b.get("title", "")}
COLLECTION: {b.get("collection", "")}
ARTICLE: {b.get("article_label", "")}
TEXT:
{b.get("content", "")}
"""
        )

    joined = "\n\n".join(blocks_text)

    MAX_FINAL_CONTEXT_CHARS = 999000

    if len(joined) > MAX_FINAL_CONTEXT_CHARS:
        joined = joined[:MAX_FINAL_CONTEXT_CHARS]

        last_source = joined.rfind("SOURCE ")
        if last_source > 0:
            joined = joined[:last_source]

    if lang == "en":
        prompt = f"""
You are a legal assistant.
Answer ONLY if the question is legal.

Question:
"{question}"

Retrieved legal sources:
{joined}

Write a clean legal answer in English in markdown.
"""
    else:
        prompt = f"""
Ти си правен асистент.
Отговаряй САМО ако въпросът е правен.

Въпрос:
"{question}"

Подадени правни източници:
{joined}

Напиши ясен правен отговор на български в markdown.
"""

    answer = await ask_gemini_summary(prompt.strip())
    return answer.strip()


async def fallback_legal_answer(question: str, lang: str):
    if lang == "en":
        prompt = f"""
You are a legal assistant.
Answer ONLY if the question is legal.

Question:
"{question}"

Write a clean legal answer in markdown.
"""
    else:
        prompt = f"""
Ти си правен асистент.
Отговаряй САМО ако въпросът е правен.

Въпрос:
"{question}"

Напиши ясен правен отговор в markdown.
"""

    answer = await ask_gemini_summary(prompt.strip())
    return answer.strip()


# =========================================================
# MAIN PIPELINE
# =========================================================
async def handle_question(question: str, options: str):
    options = (options or "all").strip().lower()
    lang = output_language(options)
    allowed_cols = allowed_collections_for_options(options)

    # паралелно стартираме vector търсенето и term extraction
    vector_docs_task = asyncio.create_task(
        search_chroma_docs(
            standalone_question=question,
            allowed_collections=allowed_cols,
            n_results=VECTOR_DOC_LIMIT,
        )
    )

    term, matched_indices, failed_terms = await generate_term_with_retries(question, allowed_cols, lang=lang)

    prerequisite_hits = []
    hits = []

    if term and matched_indices:
        if matched_indices and not isinstance(matched_indices[0], str):
            matched_indices = matched_indices[0]

        matched_indices = [i for i in matched_indices if i]
        indices_str = ",".join(matched_indices)
        print(f"🔍 Searching term '{term}' in indices: {indices_str} | options={options}")

        detailed_dsl_task = asyncio.create_task(
            generate_detailed_dsl(question, term, matched_indices, lang=lang, excluded_terms=failed_terms)
        )

        prereq_task = None
        if is_eu_regs_or_directives(matched_indices):
            agreements_idx = agreements_index_for_options(options)
            prereq_task = asyncio.to_thread(search_index_for_term, agreements_idx, question, term, lang)

        detailed_dsl = await detailed_dsl_task

        main_es_task = asyncio.to_thread(es.search, index=indices_str, body=detailed_dsl)

        if prereq_task:
            prereq_res, main_res = await asyncio.gather(prereq_task, main_es_task, return_exceptions=True)
            if not isinstance(prereq_res, Exception):
                prerequisite_hits = prereq_res
            else:
                print("⚠️ prerequisite search failed:", prereq_res)

            if not isinstance(main_res, Exception):
                hits = main_res.get("hits", {}).get("hits", [])
            else:
                print("⚠️ Elasticsearch search failed:", main_res)
        else:
            try:
                main_res = await main_es_task
                hits = main_res.get("hits", {}).get("hits", [])
            except Exception as e:
                print("⚠️ Elasticsearch search failed:", e)

    es_blocks = []
    sources = []

    for hit in prerequisite_hits + hits:
        src = hit.get("_source", {}) or {}
        desc = src.get("description", "") or ""
        title = src.get("title", "Без заглавие")
        hit_index = (hit.get("_index") or "").lower()
        highlighted_fragments = hit.get("highlight", {}).get("description", []) or []

        extracted = extract_highlighted_article_blocks(desc, highlighted_fragments, hit_index)

        for block in extracted:
            es_blocks.append({
                "source_types": ["elastic"],
                "doc_id": normalize_to_str(hit.get("_id")),
                "title": title,
                "collection": hit_index,
                "article_label": extract_article_label(block),
                "content": block,
                "vector_distance": None,
                "local_score": block_score(
                    block,
                    question,
                    term or question,
                    tokenize(term or question),
                    title,
                ),
            })



    vector_docs = await vector_docs_task

    vector_blocks = extract_blocks_from_vector_docs(
        vector_docs=vector_docs,
        standalone_question=question,
        legal_query=term or question,
        keywords=tokenize(term or question),
        per_doc_top_n=PER_VECTOR_DOC_BLOCK_LIMIT,
    )

    all_candidate_blocks = dedupe_blocks(es_blocks + vector_blocks)
    all_candidate_blocks = sorted(
        all_candidate_blocks,
        key=lambda x: x.get("local_score", 0),
        reverse=True
    )

    if all_candidate_blocks:
        answer = await final_answer_from_blocks(question, all_candidate_blocks, lang=lang)
        mode = "rag"
    else:
        answer = await fallback_legal_answer(question, lang=lang)
        mode = "generated"
    # build sources ONLY from blocks (matches)
    sources_map = {}

    for b in all_candidate_blocks:
        key = (b.get("collection"), b.get("title"))

        if key not in sources_map:
            sources_map[key] = {
                "index": b.get("collection"),
                "title": b.get("title"),
            }

    sources = list(sources_map.values())

    return {
        "question": question,
        "term": term,
        "options": options,
        "language": lang,
        "allowed_collections": [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in allowed_cols],
        "indices": matched_indices if term else [],
        "results_count": len(all_candidate_blocks),
        "summary": answer,
        "sources": sources,
        "matches": [b["content"] for b in all_candidate_blocks],
        "matches": [
            {
                "content": b["content"],
                "title": b["title"],
                "index": b["collection"],
                "article": b.get("article_label", ""),
                "doc_id": b.get("doc_id", ""),
                "source_types": b.get("source_types", []),
            }
            for b in all_candidate_blocks
        ],
        "meta": {
            "mode": mode,
            "elastic_blocks": len(es_blocks),
            "vector_docs": len(vector_docs),
            "vector_blocks": len(vector_blocks),
            "final_blocks": len(all_candidate_blocks),
        },
    }


# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {"message": "JusticIA API is running. POST your question to /generate"}


@app.post("/index/es")
def index_es():
    index_mongo_to_es()
    return {"message": "MongoDB -> Elasticsearch indexed successfully."}


@app.post("/index/chroma")
async def index_chroma():
    await index_mongo_to_chroma()
    return {"message": "MongoDB -> Chroma indexed successfully."}


@app.post("/index/all")
async def index_all_data():
    index_mongo_to_es()
    await index_mongo_to_chroma()
    return {"message": "MongoDB -> Elasticsearch + Chroma indexed successfully."}


@app.get("/collections")
def list_collections():
    return {
        "modes": ["old", "all", "bg", "en"],
        "collections": [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in MONGO_COLLECTIONS],
    }


@app.post("/generate")
async def generate(payload: GenerateRequest):
    try:
        return await handle_question(payload.question, payload.options)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print("❌ /generate failed:", e)
        raise HTTPException(status_code=500, detail="Internal server error")