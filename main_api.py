from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
import json
import re
import ast
import google.generativeai as genai
from bson import ObjectId
from fastapi import FastAPI
from pydantic import BaseModel

# 🧠 Configure Gemini
genai.configure(api_key="AIzaSyAsO2plQSBkJrv0EAYv7LMCbd3XZYnaeng")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ⚙️ Database configuration
MONGO_URI = "mongodb://mongo:kpBxiANKRFwHbakPiIIiVUgbzCFFsvyr@tramway.proxy.rlwy.net:30965"
MONGO_DB = "legaldb"
ES_URL = "https://elasticsearch-production-e1d2.up.railway.app"
MONGO_COLLECTIONS = ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
MAX_RETRIES = 5

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
es = Elasticsearch(
    ES_URL,
    verify_certs=False,
    request_timeout=60,
    retry_on_timeout=True,
    max_retries=3
)

# ⚖️ FastAPI setup
app = FastAPI(title="JusticIA API", description="Legal AI Assistant API", version="1.0.0")


# ✅ CLEAN + INDEX FUNCTION
def clean_document(doc):
    """Recursively sanitize MongoDB document keys for Elasticsearch."""
    clean_doc = {}
    for k, v in doc.items():
        clean_key = re.sub(r'[.$]', '_', k)
        if isinstance(v, dict):
            clean_doc[clean_key] = clean_document(v)
        elif isinstance(v, list):
            clean_doc[clean_key] = [clean_document(i) if isinstance(i, dict) else i for i in v]
        else:
            clean_doc[clean_key] = v
    return clean_doc


def index_mongo_to_es():
    """Indexes all MongoDB collections into Elasticsearch safely and completely."""
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

                actions.append({
                    "_index": coll_name.lower(),
                    "_id": doc_id,
                    "_source": doc
                })

                if len(actions) >= 500:
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


def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Не може да се отговори на въпроса ви:", e)
        return None


def extract_term_and_collection(question):
    prompt = f"""
Ти си български правен асистент. Ако въпросът който е попитал потребителя няма нищо общо с правото му кажи, че не можеш да отговориш на този въпрос, ако е свързан тогава извлечи основния правен термин или го генерирай на базата на въпроса и предполагаемата колекция от следния въпрос:

\"{question}\"

Върни отговор във формат:
{{
  "term": "ключов правен термин",
  "collection": ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
}}

Без обяснения, върни само JSON.
"""
    output = ask_gemini(prompt)

    try:
        json_start = output.find("{")
        json_end = output.find("}", json_start) + 1
        json_str = output[json_start:json_end]
        parsed = json.loads(json_str)

        term = parsed.get("term", "").lower()
        collection = parsed.get("collection", [])

        # ✅ FIX: always make collection a list
        if isinstance(collection, str):
            try:
                evaluated = ast.literal_eval(collection)
                if isinstance(evaluated, list):
                    collection = evaluated
                else:
                    collection = [evaluated]
            except Exception:
                collection = [collection]
        elif not isinstance(collection, list):
            collection = [collection]

        return term, collection
    except Exception as e:
        print("Failed to parse Gemini term response:", e)
        return None, []


def find_matching_indices(term, indices):
    matched = []
    for idx in indices:
        if not idx:
            continue
        try:
            res = es.search(index=idx, body={
                "query": {
                    "multi_match": {
                        "query": term,
                        "fields": ["description"]
                    }
                },
                "size": 1
            })
            if res.get("hits", {}).get("total", {}).get("value", 0) > 0:
                matched.append(idx)
        except Exception as e:
            print(f"⚠️ Error searching in index '{idx}': {e}")
    return matched


def generate_detailed_dsl(question, term, indices, excluded_terms=[]):
    if not isinstance(indices[0], str):
        indices = indices[0]
    excluded = f" Предишни термини без резултат: {', '.join(excluded_terms)}." if excluded_terms else ""
    prompt = f"""
Изходен въпрос: \"{question}\"
Текущ термин: \"{term}\".{excluded}
Генерирай детайлна Elasticsearch DSL заявка с 'highlight', търсеща в поле 'description'. Върни само JSON. НЕ включвай 'indices' в JSON заявката.
"""
    output = ask_gemini(prompt)

    try:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        json_text = output[json_start:json_end]
        return json.loads(json_text)
    except Exception as e:
        print("DSL parse error in detailed_dsl:", e)
        return {
            "query": {
                "match": {
                    "description": term
                }
            },
            "highlight": {
                "fields": {
                    "description": {}
                }
            }
        }


def extract_article_context(description, term):
    pattern = r"(Чл\..*?)(?=Чл\.|$)"
    matches = re.findall(pattern, description, flags=re.DOTALL)
    return [m.strip() for m in matches if term.lower() in m.lower()]


def summarize_results(question, chunks):
    full_text = "\n\n".join(chunks)
    prompt = f"""
Потребителят пита: \"{question}\"
Намерени са следните членове:
{full_text}

Обобщи ги на български, като говориш в трето лицe. Формата трябва да е markdown и не се обръщай към потребителя. Ако въпросът който е попитал потребителя няма нищо общо с правото му кажи, че не можеш да отговориш на този въпрос. 
"""
    output = ask_gemini(prompt)
    return output.strip()


def generate_term_with_retries(question):
    for attempt in range(MAX_RETRIES):
        term, collection = extract_term_and_collection(question)
        if not isinstance(collection, list):
            collection = [collection]
        if not term or not collection:
            continue
        matched_indices = find_matching_indices(term, collection)
        if matched_indices:
            return term, matched_indices, []
    return None, [], []


def handle_question(question):
    term, matched_indices, failed_terms = generate_term_with_retries(question)

    if not term:
        return {"error": "Не може да се намери термин с резултати."}
    if not matched_indices:
        return {"error": "Няма индекси с резултати за този термин."}

    if not isinstance(matched_indices[0], str):
        matched_indices = matched_indices[0]
    matched_indices = [i for i in matched_indices if i]
    indices_str = ",".join(matched_indices)

    print(f"🔍 Searching for term '{term}' in indices: {indices_str}")

    detailed_dsl = generate_detailed_dsl(question, term, matched_indices)

    try:
        res = es.search(index=indices_str, body=detailed_dsl)
    except Exception as e:
        print(f"💥 Elasticsearch search error: {e}")
        return {"error": f"Неуспешно търсене в Elasticsearch: {str(e)}"}

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        return {"message": f"Няма намерени резултати за '{term}'."}

    all_hits = []
    sources = []

    for hit in hits:
        desc = hit["_source"].get("description", "")
        chlen_matches = extract_article_context(desc, term)
        all_hits.extend(chlen_matches)
        sources.append({
            "index": hit["_index"],
            "title": hit["_source"].get("title", "Без заглавие")
        })

    summary = summarize_results(question, all_hits) if all_hits else "Няма релевантни членове."

    return {
        "term": term,
        "indices": matched_indices,
        "results_count": len(all_hits),
        "summary": summary,
        "sources": sources,
        "matches": all_hits
    }


# 🧩 FastAPI Routes
class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "JusticIA API is running. POST your question to /generate"}


@app.post("/index")
def index_all_data():
    index_mongo_to_es()
    return {"message": "Data indexed successfully."}


@app.post("/generate")
def generate(payload: Question):
    return handle_question(payload.question)
