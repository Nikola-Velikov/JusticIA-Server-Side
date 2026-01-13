from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
import json
import re
import ast
from bson import ObjectId
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from groq import Groq
import google.generativeai as genai

print("RUNNING FILE:", __file__)
print("CWD:", os.getcwd())

load_dotenv()

# 🧩 Configure Gemini only for summaries
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ✅ Groq LLaMA for query understanding + DSL generation
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ⚙️ Database configuration
MONGO_URI = "mongodb://mongo:kpBxiANKRFwHbakPiIIiVUgbzCFFsvyr@tramway.proxy.rlwy.net:30965"
MONGO_DB = "legaldb"
ES_URL = "https://elasticsearch-production-e1d2.up.railway.app"

BASE_COLLECTIONS = [
    "constitution",
    "codex",
    "laws",
    "implementableRegulations",
    "regulations",
    "rules",
]

# ✅ FIX: agreements names aligned with the rest of your code (agreements_index_for_options)
EU_COLLECTIONS = [
    "regulations-eu-bg",
    "regulations-eu-en",
    "directives-eu-bg",
    "directives-eu-en",
    "agreement-eu-bg",
    "agreement-eu-en",
]

# ✅ Collections physically indexed in ES
MONGO_COLLECTIONS = BASE_COLLECTIONS + EU_COLLECTIONS

MAX_RETRIES = 5

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
es = Elasticsearch(
    ES_URL,
    verify_certs=False,
    request_timeout=60,
    retry_on_timeout=True,
    max_retries=3,
)

# ⚖️ FastAPI setup
app = FastAPI(
    title="JusticIA API",
    description="Legal AI Assistant API (Groq LLaMA 3.3 + Gemini summaries)",
    version="2.1.0",
)

# ✅ Collection descriptions (for better routing + API clarity)
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
        "Включват: Договор за Европейския съюз (консолидиран текст 2016 г.), "
        "Договор за функционирането на ЕС (2016 г.), Договор за създаване на "
        "Европейската общност за атомна енергия (2016 г.), "
        "Харта на основните права на ЕС (2016 г.)."
    ),
    # ✅ FIX: keep as ONE string (not list)
    "agreement-eu-en": (
        "EU treaties are binding agreements between the parties. Includes: "
        "Treaty on European Union (consolidated 2016), Treaty on the Functioning of the EU (2016), "
        "Euratom Treaty (2016), Charter of Fundamental Rights of the EU (2016)."
    ),
}


# ✅ CLEAN + INDEX FUNCTION
def clean_document(doc):
    clean_doc = {}
    for k, v in doc.items():
        clean_key = re.sub(r"[.$]", "_", k)
        if isinstance(v, dict):
            clean_doc[clean_key] = clean_document(v)
        elif isinstance(v, list):
            clean_doc[clean_key] = [clean_document(i) if isinstance(i, dict) else i for i in v]
        else:
            clean_doc[clean_key] = v
    return clean_doc


def index_mongo_to_es():
    print(EU_COLLECTIONS)
    # Delete old indices first
    for coll_name in MONGO_COLLECTIONS:
        index_name = coll_name.lower()
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
                print(f"🧹 Deleted old index '{index_name}' before reindexing.")
        except Exception as e:
            print(f"⚠️ Could not check/delete index '{index_name}': {e}")

    # Reindex all collections
    for coll_name in MONGO_COLLECTIONS:
        collection = mongo_db[coll_name]
        docs = collection.find()
        actions = []
        error_count = 0
        total_indexed = 0

        print(f"🚀 Indexing collection: {coll_name}")

        for doc in docs:
            try:
                doc_id = str(doc["_id"])
                doc.pop("_id", None)

                for key, value in doc.items():
                    if isinstance(value, ObjectId):
                        doc[key] = str(value)
                    elif hasattr(value, "isoformat"):
                        doc[key] = value.isoformat()

                doc = clean_document(doc)
                actions.append({"_index": coll_name.lower(), "_id": doc_id, "_source": doc})

                if len(actions) >= 100:
                    helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
                    total_indexed += len(actions)
                    actions = []
            except Exception as e:
                error_count += 1
                print(f"⚠️ Skipped one doc in {coll_name}: {e}")

        if actions:
            try:
                helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
                total_indexed += len(actions)
            except BulkIndexError as e:
                print(f"❌ Bulk index error in {coll_name}")
                for err in e.errors[:5]:
                    print(json.dumps(err, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"💥 Unexpected error in {coll_name}: {e}")

        print(f"✅ Indexed {total_indexed} documents from {coll_name}")
        if error_count:
            print(f"⚠️ Skipped {error_count} invalid docs in {coll_name}")

    print("🎯 All MongoDB collections successfully indexed into Elasticsearch!")


def ask_gemini(prompt: str):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Gemini error:", e)
        return None


def ask_llama(prompt: str):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1024,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ LLaMA request failed:", e)
        return None


# ----------------------------
# ✅ OPTIONS-BASED ROUTING
# ----------------------------
def allowed_collections_for_options(options: str):
    """
    options:
      - old -> only the previous BG collections (BASE_COLLECTIONS)
      - all -> BASE_COLLECTIONS + regulations-eu-bg + directives-eu-bg
      - bg  -> regulations-eu-bg + directives-eu-bg
      - en  -> regulations-eu-en + directives-eu-en
    """
    options = (options or "all").strip().lower()

    if options == "old":
        return BASE_COLLECTIONS

    if options == "bg":
        return ["regulations-eu-bg", "directives-eu-bg"]

    if options == "en":
        return ["regulations-eu-en", "directives-eu-en"]

    # default: all
    return BASE_COLLECTIONS + ["regulations-eu-bg", "directives-eu-bg"]


def agreements_index_for_options(options: str):
    options = (options or "all").strip().lower()
    if options == "en":
        return "agreement-eu-en"
    return "agreement-eu-bg"


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


def output_language(options: str) -> str:
    return "en" if (options or "").strip().lower() == "en" else "bg"


# ----------------------------
# ✅ TERM + COLLECTION EXTRACT
# ----------------------------
# ✅ ONLY CHANGE: robust JSON parsing + robust coercion to string/list


def safe_parse_json(output: str):
    """
    Extract the first {...} JSON block from model output and parse it.
    Handles cases where the model returns extra text before/after JSON.
    """
    if not output:
        return None
    m = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def normalize_to_str(x) -> str:
    """
    Coerce LLaMA fields into a string safely.
    Fixes cases where term is returned as dict/list instead of string.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, list):
        return " ".join(normalize_to_str(i) for i in x if i is not None)
    if isinstance(x, dict):
        # common keys models might use
        for k in ("term", "value", "text", "name", "query"):
            if k in x:
                return normalize_to_str(x[k])
        # fallback: join values
        return " ".join(normalize_to_str(v) for v in x.values() if v is not None)
    return str(x)


def normalize_collection_list(x):
    """
    Coerce LLaMA 'collection' field into a list[str] safely.
    Handles list/str/dict cases.
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, dict):
                # if it returns {"name": "..."}
                if "name" in item:
                    out.append(normalize_to_str(item["name"]))
                else:
                    out.append(normalize_to_str(item))
            else:
                out.append(normalize_to_str(item))
        return out
    if isinstance(x, dict):
        if "name" in x:
            return [normalize_to_str(x["name"])]
        return [normalize_to_str(x)]
    if isinstance(x, str):
        # allow "['laws']" style strings too
        try:
            evaluated = ast.literal_eval(x)
            if isinstance(evaluated, list):
                return [normalize_to_str(i) for i in evaluated]
        except Exception:
            pass
        return [x]
    return [normalize_to_str(x)]


def extract_term_and_collection(question: str, allowed_collections: list[str], lang: str):
    allowed_payload = [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in allowed_collections]
    allowed_json = json.dumps(allowed_payload, ensure_ascii=False)

    if lang == "en":
        prompt = f"""
You are a legal assistant. If the question is not about law, say you cannot answer.
If it is about law, extract the main legal term and choose the best single collection from the allowed list.

Allowed collections (choose ONLY from this list):
{allowed_json}

Question: "{question}"

Return ONLY JSON in the format:
{{
  "term": "main legal term",
  "collection": ["one_collection_from_allowed_list"]
}}
"""
    else:
        prompt = f"""
Ти си български правен асистент. Ако въпросът няма нищо общо с правото, отговори, че не можеш да отговориш.
Ако е свързан с правото, извлечи основния правен термин и избери най-подходящата ЕДНА колекция от разрешения списък.

Разрешени колекции (избирай САМО от този списък):
{allowed_json}

Въпрос: "{question}"

Върни само JSON във формат:
{{
  "term": "ключов правен термин",
  "collection": ["one_collection_from_allowed_list"]
}}
"""

    output = ask_llama(prompt)
    try:
        parsed = safe_parse_json(output)
        if not parsed:
            raise ValueError("No valid JSON found in LLaMA output")

        term = normalize_to_str(parsed.get("term", "")).strip()
        collection = normalize_collection_list(parsed.get("collection", []))

        collection = [c.strip().lower() for c in collection if c and normalize_to_str(c).strip()]

        allowed_set = {c.lower() for c in allowed_collections}
        collection = [c for c in collection if c in allowed_set]

        return term, collection
    except Exception as e:
        print("⚠️ Failed to parse LLaMA term response:", e)
        return None, []


def find_matching_indices(term: str, indices: list[str]):
    matched = []
    for idx in indices:
        if not idx:
            continue
        try:
            res = es.search(
                index=idx,
                body={
                    "query": {"multi_match": {"query": term, "fields": ["title^3", "description"]}},
                    "size": 1,
                },
            )
            if res.get("hits", {}).get("total", {}).get("value", 0) > 0:
                matched.append(idx)
        except Exception as e:
            print(f"⚠️ Error searching in index '{idx}': {e}")
    return matched


def generate_detailed_dsl(question: str, term: str, indices: list[str], lang: str, excluded_terms: list[str] = []):
    if indices and not isinstance(indices[0], str):
        indices = indices[0]

    excluded = (
        f" Previous no-result terms: {', '.join(excluded_terms)}."
        if (excluded_terms and lang == "en")
        else (f" Предишни термини без резултат: {', '.join(excluded_terms)}." if excluded_terms else "")
    )

    if lang == "en":
        prompt = f"""
Question: "{question}"
Current term: "{term}"{excluded}
Generate an Elasticsearch DSL query JSON using 'highlight' and 'multi_match' (title^3, description).
Return ONLY JSON.
"""
    else:
        prompt = f"""
Изходен въпрос: "{question}"
Текущ термин: "{term}"{excluded}
Генерирай Elasticsearch DSL заявка (JSON) с 'highlight' и 'multi_match' (title^3, description).
Върни САМО JSON.
"""

    output = ask_llama(prompt)
    try:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        json_text = output[json_start:json_end]
        dsl = json.loads(json_text)

        if "query" not in dsl or "multi_match" not in dsl["query"]:
            raise ValueError("Incomplete DSL")

        if "highlight" not in dsl:
            dsl["highlight"] = {"fields": {"title": {}, "description": {}}}

        dsl["size"] = dsl.get("size", 100)
        return dsl
    except Exception:
        return {
            "query": {"multi_match": {"query": term, "fields": ["title^3", "description"]}},
            "highlight": {"fields": {"title": {}, "description": {}}},
            "size": 100,
        }


# -------------------------------------------------------------------
# ✅ CONTEXT EXTRACTION LOGIC (CHANGED AS YOU REQUESTED)
#   - BG old collections: "Чл."
#   - regulations-eu-bg / directives-eu-bg / agreements-eu-bg: "Член"
#   - regulations-eu-en / directives-eu-en / agreements-eu-en: "Article"
# -------------------------------------------------------------------
def _extract_blocks(description: str, header_regex: str) -> list[str]:
    """
    Generic block extractor:
      captures from a header (e.g., "Член 12" or "Article 12") until next header or end.
    """
    if not description:
        return []
    pattern = rf"({header_regex}.*?)(?={header_regex}|$)"
    return [m.strip() for m in re.findall(pattern, description, flags=re.DOTALL | re.IGNORECASE)]


def extract_context_chl(description: str, term: str) -> list[str]:
    # Bulgarian short form: "Чл."
    blocks = _extract_blocks(description, r"Чл\.\s*\d+[^\n]*")
    return [b for b in blocks if term.lower() in b.lower()]


def extract_context_chlen(description: str, term: str) -> list[str]:
    # Bulgarian EU form: "Член"
    blocks = _extract_blocks(description, r"Член\s*\d+[^\n]*")
    # fallback if dataset sometimes uses "Чл."
    if not blocks:
        blocks = _extract_blocks(description, r"Чл\.\s*\d+[^\n]*")
    return [b for b in blocks if term.lower() in b.lower()]


def extract_context_article(description: str, term: str) -> list[str]:
    # English form: "Article"
    blocks = _extract_blocks(description, r"Article\s*\d+[^\n]*")
    return [b for b in blocks if term.lower() in b.lower()]


def pick_extractor_by_index(index_name: str):
    idx = (index_name or "").lower()

    # EU BG regs/directives + BG agreements => "Член"
    if idx in {"regulations-eu-bg", "directives-eu-bg", "agreement-eu-bg"}:
        return extract_context_chlen

    # EU EN regs/directives + EN agreements => "Article"
    if idx in {"regulations-eu-en", "directives-eu-en", "agreement-eu-en"}:
        return extract_context_article

    # everything else (old BG corpora) => "Чл."
    return extract_context_chl


def generate_term_with_retries(question: str, allowed_cols: list[str], lang: str):
    failed_terms = []
    for _ in range(MAX_RETRIES):
        term, collection = extract_term_and_collection(question, allowed_cols, lang=lang)
        if not isinstance(collection, list):
            collection = [collection]
        if not term or not collection:
            continue

        matched_indices = find_matching_indices(term, collection)
        if matched_indices:
            return term, matched_indices, failed_terms

        failed_terms.append(term)

    return None, [], failed_terms


def summarize_results(question: str, chunks: list[str], lang: str):
    full_text = "\n\n".join(chunks)

    if lang == "en":
        prompt = f"""
User question: "{question}"
Found excerpts:
{full_text}

Write a clean, professional legal summary in ENGLISH using markdown.

Requirements:
- Read like a short legal article, clear and structured.
- Avoid excessive formatting.
- Use **bold** only for key legal concepts.
- Use `##` headings only for major sections.
- No nested lists.
- Neutral, precise legal tone.
"""
    else:
        prompt = f"""
Потребителят пита: "{question}"
Намерени членове:
{full_text}

Обобщи на български в markdown елегантно и професионално, в трето лице.

🪶💼 **Стил и изисквания:**
- Форматирай като кратка юридическа статия с ясен, изчистен markdown.
- Без празни редове между всеки елемент — текстът трябва да изглежда сбит и подреден.
- Използвай **bold** за ключови понятия и не за дъми като първо второ и други подобни.
- Използвай подзаглавия с `##` само за основните раздели.
- Не ползвай вложени списъци — само обикновените точки (`-`) при нужда.
- Без емоджита, без прекомерно форматиране.
- Разделай на параграфи посмислово
- Гласът да бъде **неутрален, ясен и юридически точен**.
- Да звучи като правно резюме, а не като автоматичен превод.

🧾 **Цел:** Резултатът да прилича на добре оформен правен текст, подходящ за уеб публикация.
"""

    output = ask_gemini(prompt)
    return output.strip() if output else ("No answer." if lang == "en" else "Няма отговор.")


def search_index_for_term(index_name: str, question: str, term: str, lang: str):
    dsl = generate_detailed_dsl(question, term, [index_name], lang=lang)
    try:
        res = es.search(index=index_name, body=dsl)
        return res.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"⚠️ Failed searching prerequisite index '{index_name}': {e}")
        return []


def handle_question(question: str, options: str):
    options = (options or "all").strip().lower()
    lang = output_language(options)

    allowed_cols = allowed_collections_for_options(options)

    term, matched_indices, failed_terms = generate_term_with_retries(question, allowed_cols, lang=lang)
    if not term:
        return {"error": "Cannot find a term with results." if lang == "en" else "Не може да се намери термин с резултати."}

    if not matched_indices:
        return {"error": "No indices have results for this term." if lang == "en" else "Няма индекси с резултати за този термин."}

    if matched_indices and not isinstance(matched_indices[0], str):
        matched_indices = matched_indices[0]

    matched_indices = [i for i in matched_indices if i]
    indices_str = ",".join(matched_indices)
    print(f"🔍 Searching term '{term}' in indices: {indices_str} | options={options}")

    # ✅ Agreements-first prerequisite (only for EU regs/directives)
    prerequisite_hits = []
    if is_eu_regs_or_directives(matched_indices):
        agreements_idx = agreements_index_for_options(options)
        prerequisite_hits = search_index_for_term(agreements_idx, question, term, lang=lang)

    # main search
    detailed_dsl = generate_detailed_dsl(question, term, matched_indices, lang=lang, excluded_terms=failed_terms)
    try:
        res = es.search(index=indices_str, body=detailed_dsl)
    except Exception as e:
        return {"error": f"Elasticsearch search failed: {str(e)}" if lang == "en" else f"Неуспешно търсене в Elasticsearch: {str(e)}"}

    hits = res.get("hits", {}).get("hits", [])
    if not hits and not prerequisite_hits:
        return {"message": f"No results for '{term}'." if lang == "en" else f"Няма намерени резултати за '{term}'."}

    all_chunks = []
    sources = []

    # 1) agreements first (if any)
    for hit in prerequisite_hits:
        src = hit.get("_source", {}) or {}
        desc = src.get("description", "") or ""
        title = (src.get("title", "") or "").lower()
        hit_index = (hit.get("_index") or "").lower()

        extractor = pick_extractor_by_index(hit_index)

        if term.lower() in title and desc:
            all_chunks.append(desc)
        else:
            all_chunks.extend(extractor(desc, term))

        sources.append({
            "index": hit_index or "agreement",
            "title": src.get("title", "Untitled" if lang == "en" else "Без заглавие")
        })

    # 2) then matched indices
    for hit in hits:
        src = hit.get("_source", {}) or {}
        title = (src.get("title", "") or "").lower()
        desc = src.get("description", "") or ""
        hit_index = (hit.get("_index") or "").lower()

        extractor = pick_extractor_by_index(hit_index)

        if term.lower() in title and desc:
            all_chunks.append(desc)
            sources.append({
                "index": hit_index,
                "title": src.get("title", "Untitled" if lang == "en" else "Без заглавие")
            })
            continue

        all_chunks.extend(extractor(desc, term))
        sources.append({
            "index": hit_index,
            "title": src.get("title", "Untitled" if lang == "en" else "Без заглавие")
        })

    summary = summarize_results(question, all_chunks, lang=lang) if all_chunks else (
        "No relevant excerpts." if lang == "en" else "Няма релевантни членове."
    )

    return {
        "term": term,
        "options": options,
        "language": lang,
        "allowed_collections": [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in allowed_cols],
        "indices": matched_indices,
        "results_count": len(all_chunks),
        "summary": summary,
        "sources": sources,
        "matches": all_chunks,
    }


# 🧩 FastAPI Routes
class GenerateRequest(BaseModel):
    question: str
    options: str = Field(default="all", pattern="^(all|bg|en|old)$")


@app.get("/")
def home():
    return {"message": "JusticIA API is running. POST your question to /generate"}


@app.post("/index")
def index_all_data():
    index_mongo_to_es()
    return {"message": "Data indexed successfully (base + EU collections)."}


@app.get("/collections")
def list_collections():
    return {
        "modes": ["old", "all", "bg", "en"],
        "collections": [{"name": c, "description": COLLECTION_DESCRIPTIONS.get(c, "")} for c in MONGO_COLLECTIONS],
    }


@app.post("/generate")
def generate(payload: GenerateRequest):
    return handle_question(payload.question, payload.options)
