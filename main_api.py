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
genai.configure(api_key="AIzaSyALQ_87RpTUQjK1X-24JY8oc_SCNPEoTDE")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ⚙️ Database configuration
MONGO_URI = "mongodb://mongo:kpBxiANKRFwHbakPiIIiVUgbzCFFsvyr@tramway.proxy.rlwy.net:30965"
MONGO_DB = "legaldb"
ES_URL = "http://localhost:9200"
MONGO_COLLECTIONS = ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
MAX_RETRIES = 5

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
es = Elasticsearch(ES_URL)

# ⚖️ FastAPI setup
app = FastAPI(title="JusticIA API", description="Legal AI Assistant API", version="1.0.0")


def index_mongo_to_es():
    for coll_name in MONGO_COLLECTIONS:
        collection = mongo_db[coll_name]
        docs = collection.find()
        actions = []
        error_count = 0

        for doc in docs:
            try:
                doc_id = str(doc["_id"])
                doc.pop("_id", None)

                for key, value in doc.items():
                    if isinstance(value, ObjectId):
                        doc[key] = str(value)
                    elif hasattr(value, 'isoformat'):
                        doc[key] = value.isoformat()

                action = {
                    "_index": coll_name,
                    "_id": doc_id,
                    "_source": doc
                }
                actions.append(action)
            except Exception as e:
                error_count += 1
                print(f"Error preparing doc: {e}")

        if actions:
            try:
                helpers.bulk(es, actions, raise_on_error=False)
                print(f"Indexed {len(actions)} documents from {coll_name}")
            except BulkIndexError as e:
                print(f"Bulk index error in {coll_name}")
                for err in e.errors[:5]:
                    print(json.dumps(err, indent=2, ensure_ascii=False))
                print(f"...and {len(e.errors)} more errors.\n")

        if error_count:
            print(f"Skipped {error_count} documents in {coll_name} due to errors.")


def ask_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Не може да се отговори на въпроса ви:", e)
        return None


def extract_term_and_collection(question):
    prompt = f"""
Ти си български правен асистент. Извлечи основния правен термин и предполагаемата колекция от следния въпрос:

\"{question}\"

Върни отговор във формат:
{{
  "term": "ключов правен термин",
  "collection":"constitution"- конституция, "codex", "laws"- закони, "implementableRegulations" - правилници по прилагане , "regulations"- правилници, "rules" - наредби, може и няколко
}}

Без обяснения, върни само JSON.
"""
    output = ask_gemini(prompt)

    try:
        json_start = output.find("{")
        json_end = output.find("}", json_start) + 1
        json_str = output[json_start:json_end]
        parsed = json.loads(json_str)
        return parsed.get("term", "").lower(), parsed.get("collection", "")
    except Exception as e:
        print("Failed to parse Gemini term response:", e)
        return None, None


def find_matching_indices(term, indices):
    matched = []
    for idx in indices:
        res = es.search(index=idx, body={
            "query": {
                "multi_match": {
                    "query": term,
                    "fields": ["description"]
                }
            },
            "size": 1
        })
        if res["hits"]["total"]["value"] > 0:
            matched.append(idx)
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
        # fallback на базово търсене по термина
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
    list_collection = []
    for attempt in range(MAX_RETRIES):
        term, collection = extract_term_and_collection(question)
        if not isinstance(collection, list):
            list_collection.append(collection)
            collection = list_collection
        if not term or not collection:
            continue
        matched_indices = find_matching_indices(term, [collection])
        if matched_indices:
            return term, matched_indices, []
    return None, [], []


def handle_question(question):
    term, matched_indices, failed_terms = generate_term_with_retries(question)
    if not term:
        return {"error": "Не може да се намери термин с резултати."}

    if not isinstance(matched_indices[0], str):
        matched_indices = matched_indices[0]

    detailed_dsl = generate_detailed_dsl(question, term, matched_indices)

    all_hits = []
    sources = []

    indices_str = ",".join(matched_indices)
    res = es.search(index=indices_str, body=detailed_dsl)
    hits = res["hits"]["hits"]

    for hit in hits:
        desc = hit["_source"].get("description", "")
        chlen_matches = extract_article_context(desc, term)
        all_hits.extend(chlen_matches)
        sources.append({
            "index": hit["_index"],
            "title": hit["_source"].get("title", "Без заглавие")
        })

    summary = summarize_results(question, all_hits)

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


@app.post("/generate")
def generate(payload: Question):
    """Generate a summarized legal answer."""
    return handle_question(payload.question)
