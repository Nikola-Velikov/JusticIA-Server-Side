from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError
import json
import re
import ast
from bson import ObjectId
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq   # ✅ NEW: import Groq client
import google.generativeai as genai

# 🧩 Configure Gemini only for summaries
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # ✅ NEW: configure Groq client

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
app = FastAPI(title="JusticIA API", description="Legal AI Assistant API (Groq LLaMA 3.3)", version="1.0.0")


# ✅ CLEAN + INDEX FUNCTION
def clean_document(doc):
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
    for coll_name in MONGO_COLLECTIONS:
        index_name = coll_name.lower()
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name)
                print(f"🧹 Deleted old index '{index_name}' before reindexing.")
        except Exception as e:
            print(f"⚠️ Could not check/delete index '{index_name}': {e}")

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


def ask_gemini(prompt):

    try:

        response = gemini_model.generate_content(prompt)

        return response.text

    except Exception as e:

        print("Не може да се отговори на въпроса ви:", e)

        return None
def ask_llama(prompt):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1024
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ LLaMA request failed:", e)
        return None


# --- All logic below now calls ask_llama() instead of ask_gemini() ---

def extract_term_and_collection(question):
    prompt = f"""
Ти си български правен асистент. Ако въпросът няма нищо общо с правото, отговори, че не можеш да отговориш.
Ако е свързан с правото, извлечи основния правен термин и най-подходящата колекция:

Въпрос: "{question}"

Върни само JSON във формат:
{{
  "term": "ключов правен термин",
  "collection": ["constitution", "codex", "laws", "implementableRegulations", "regulations", "rules"]
}}
"""
    output = ask_llama(prompt)
    try:
        json_start = output.find("{")
        json_end = output.find("}", json_start) + 1
        json_str = output[json_start:json_end]
        parsed = json.loads(json_str)
        term = parsed.get("term", "").lower()
        collection = parsed.get("collection", [])
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
        collection = [c.lower() for c in collection]
        return term, collection
    except Exception as e:
        print("⚠️ Failed to parse LLaMA term response:", e)
        return None, []


def find_matching_indices(term, indices):
    matched = []
    for idx in indices:
        if not idx:
            continue
        try:
            res = es.search(index=idx, body={
                "query": {"multi_match": {"query": term, "fields": ["title^3", "description"]}},
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
Изходен въпрос: "{question}"
Текущ термин: "{term}"{excluded}
Генерирай Elasticsearch DSL заявка (JSON) с 'highlight' и 'multi_match' (title^3, description).
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
            "size": 100
        }


def extract_article_context(description, term):
    pattern = r"(Чл\..*?)(?=Чл\.|$)"
    matches = re.findall(pattern, description, flags=re.DOTALL)
    return [m.strip() for m in matches if term.lower() in m.lower()]


def summarize_results(question, chunks):
    full_text = "\n\n".join(chunks)
    prompt = f"""
Потребителят пита: "{question}"
Намерени членове:
{full_text}

Обобщи на български в markdown елегантно и професионално, в трето лице.

🪶💼 **Стил и изисквания:**
- Форматирай като кратка юридическа статия с ясен, изчистен markdown.
- Без празни редове между всеки елемент — текстът трябва да изглежда сбит и подреден.
- Използвай **bold** за ключови понятия и заглавия и не за разделители на изреченията и параграфите.
- Използвай подзаглавия с `##` само за основните раздели.
- Не ползвай вложени списъци — само обикновените точки (`-`) при нужда.
- Без емоджита, без прекомерно форматиране.
- Гласът да бъде **неутрален, ясен и юридически точен**.
- Да звучи като правно резюме, а не като автоматичен превод.

🧾 **Цел:** Резултатът да прилича на добре оформен правен текст, подходящ за уеб публикация.
"""

    output = ask_gemini(prompt)
    return output.strip() if output else "Няма отговор."


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
        return {"error": f"Неуспешно търсене в Elasticsearch: {str(e)}"}

    hits = res.get("hits", {}).get("hits", [])
    if not hits:
        return {"message": f"Няма намерени резултати за '{term}'."}

    all_hits = []
    sources = []
    for hit in hits:
        source = hit["_source"]
        title = source.get("title", "").lower()
        desc = hit["_source"].get("description", "")
        if term.lower() in title:
            all_hits.append(desc)
            sources.append({"index": hit["_index"], "title": source.get("title", "Без заглавие")})
            continue
        chlen_matches = extract_article_context(desc, term)
        all_hits.extend(chlen_matches)
        sources.append({"index": hit["_index"], "title": hit["_source"].get("title", "Без заглавие")})
    summary = summarize_results(question, all_hits) if all_hits else "Няма релевантни членове."
    return {"term": term, "indices": matched_indices, "results_count": len(all_hits),
            "summary": summary, "sources": sources, "matches": all_hits}


# 🧩 FastAPI Routes
class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "JusticIA API (Groq LLaMA 3.3) is running. POST your question to /generate"}


@app.post("/index")
def index_all_data():
    index_mongo_to_es()
    return {"message": "Data indexed successfully."}


@app.post("/generate")
def generate(payload: Question):
    return handle_question(payload.question)
