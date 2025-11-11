from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
import json
import re
import google.generativeai as genai
from bson import ObjectId
from fastapi import FastAPI
from pydantic import BaseModel

# 🧠 Configure Gemini
genai.configure(api_key="AIzaSyDFIm6r7BW-SFdngtGUd_76zUV2cVKXOl4")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ⚙️ Configuration
MONGO_URI = "mongodb://mongo:kpBxiANKRFwHbakPiIIiVUgbzCFFsvyr@tramway.proxy.rlwy.net:30965"
MONGO_DB = "legaldb"
ES_URL = "https://elasticsearch-production-e1d2.up.railway.app"
MONGO_COLLECTIONS = ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
MAX_RETRIES = 5

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]

# 👇 Force compatible header version 8
es = Elasticsearch(
    ES_URL,
    verify_certs=False,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8",
             "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8"},
    request_timeout=60,
    retry_on_timeout=True,
    max_retries=3
)

# ⚖️ FastAPI
app = FastAPI(title="JusticIA API", description="Legal AI Assistant API", version="1.0.0")


def clean_document(doc):
    """Recursively sanitize MongoDB document keys."""
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


def index_missing_collections():
    """Ensures every Mongo collection is indexed in Elasticsearch."""
    try:
        existing_indices = set(es.indices.get_alias(name="*").keys())
    except Exception:
        existing_indices = set()

    for coll_name in MONGO_COLLECTIONS:
        if coll_name in existing_indices:
            continue  # skip already indexed
        print(f"⚙️ Creating missing index: {coll_name}")
        collection = mongo_db[coll_name]
        docs = collection.find()
        actions = []

        for doc in docs:
            try:
                doc_id = str(doc["_id"])
                doc.pop("_id", None)
                for key, value in doc.items():
                    if isinstance(value, ObjectId):
                        doc[key] = str(value)
                    elif hasattr(value, "isoformat"):
                        doc[key] = value.isoformat()
                actions.append({
                    "_index": coll_name.lower(),
                    "_id": doc_id,
                    "_source": clean_document(doc)
                })
            except Exception as e:
                print(f"⚠️ Skipped a doc from {coll_name}: {e}")

        if actions:
            try:
                helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
                print(f"✅ Indexed {len(actions)} docs into '{coll_name}'")
            except Exception as e:
                print(f"❌ Error indexing {coll_name}: {e}")


def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        print("Term:", response.text)
        return response.text
    except Exception as e:
        print("⚠️ Gemini error:", e)
        return None


def extract_term_and_collection(question):
    prompt = f"""
Ти си интелигентен български правен асистент.
Определи основния правен термин (само съществително) и колекциите, в които вероятно се намира.

Формат само в JSON:
{{
  "term": "<ключов правен термин>",
  "collection": ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
}}

Въпрос: "{question}"
"""
    output = ask_gemini(prompt)
    try:
        json_start = output.find("{")
        json_end = output.rfind("}") + 1
        parsed = json.loads(output[json_start:json_end])
        term = parsed.get("term", "").lower()
        collection = parsed.get("collection", [])
        if isinstance(collection, str):
            collection = [collection]
        return term, collection
    except Exception as e:
        print("⚠️ Failed to parse Gemini output:", e)
        return None, []


def find_matching_indices(term, indices):
    """Only search in existing indices."""
    matched = []
    try:
        all_indices = set(es.indices.get_alias(name="*").keys())
    except Exception as e:
        print(f"⚠️ Could not list ES indices: {e}")
        all_indices = set()

    for idx in indices:
        if not idx:
            continue
        if idx not in all_indices:
            print(f"⚠️ Skipping missing index '{idx}' — not found in Elasticsearch.")
            continue
        try:
            res = es.search(index=idx, body={
                "query": {"multi_match": {"query": term, "fields": ["description"]}},
                "size": 1
            })
            if res.get("hits", {}).get("total", {}).get("value", 0) > 0:
                matched.append(idx)
        except Exception as e:
            print(f"⚠️ Error searching in index '{idx}': {e}")
    return matched


def generate_detailed_dsl(question, term):
    return {
        "query": {"match": {"description": term}},
        "highlight": {"fields": {"description": {}}}
    }


def extract_article_context(description, term):
    pattern = r"(Чл\..*?)(?=Чл\.|$)"
    return [m.strip() for m in re.findall(pattern, description, flags=re.DOTALL)
            if term.lower() in m.lower()]


def summarize_results(question, chunks):
    text = "\n\n".join(chunks)
    prompt = f"""
Потребителят пита: "{question}"
Намерени членове:
{text}

Обобщи резултата в markdown, кратко и юридически точно.
"""
    return ask_gemini(prompt)


def handle_question(question):
    term, collections = extract_term_and_collection(question)
    if not term or not collections:
        return {"error": "Не може да се извлече термин."}

    print(f"🔍 Term: {term}, Collections: {collections}")

    matched_indices = find_matching_indices(term, collections)
    if not matched_indices:
        index_missing_collections()  # 👈 Auto-index if missing
        matched_indices = find_matching_indices(term, collections)
        if not matched_indices:
            return {"error": f"Няма намерени индекси за {term}."}

    res = es.search(index=matched_indices, body=generate_detailed_dsl(question, term))
    hits = res.get("hits", {}).get("hits", [])

    if not hits:
        return {"message": f"Няма намерени резултати за '{term}'."}

    articles, sources = [], []
    for hit in hits:
        desc = hit["_source"].get("description", "")
        articles.extend(extract_article_context(desc, term))
        sources.append({"index": hit["_index"], "title": hit["_source"].get("title", "Без заглавие")})

    summary = summarize_results(question, articles)
    return {
        "term": term,
        "indices": matched_indices,
        "results_count": len(articles),
        "summary": summary,
        "sources": sources,
        "matches": articles
    }


# 🚀 FastAPI Routes
class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "JusticIA API is running."}


@app.post("/index")
def index_all_data():
    index_missing_collections()
    return {"message": "All missing collections indexed successfully."}


@app.post("/generate")
def generate(payload: Question):
    return handle_question(payload.question)
