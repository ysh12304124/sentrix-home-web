import os
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import MemoryAgent
from .db import MemoryStore, make_id
from .model_clients import ClipAdapter, FaceAdapter, FunASRClient, GammaClient, parse_json_response
from .pipeline import IngestionPipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("SENTRIX_DATA_DIR", ROOT / "data"))
MEDIA_DIR = DATA_DIR / "media"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

store = MemoryStore(os.getenv("SENTRIX_DB_PATH", str(DATA_DIR / "sentrix.db")))
gamma = GammaClient()
pipeline = IngestionPipeline(store, gamma=gamma, asr=FunASRClient(), face=FaceAdapter(), clip=ClipAdapter())
agent = MemoryAgent(store, gamma=gamma, clip=pipeline.clip)

app = FastAPI(title="Sentrix Home Memory API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchRequest(BaseModel):
    query: str
    spaceId: str = "home-default"


class ImportRequest(BaseModel):
    fileName: str = "unknown"
    mediaType: str = "text"


def process_asset(asset_id):
    pipeline.process(asset_id)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": "sentrix-local-backend",
        "models": {
            "gamma4_12B": {"name": gamma.model, "endpoint": gamma.base_url},
            "asr": {"name": pipeline.asr.model_name, "vad": pipeline.asr.vad_model, "punc": pipeline.asr.punc_model, "ready": pipeline.asr.error is None, "error": pipeline.asr.error},
            "face": {"enabled": pipeline.face.enabled, "ready": pipeline.face.error is None, "error": pipeline.face.error},
            "clip": {"enabled": pipeline.clip.enabled, "model": pipeline.clip.model_name, "ready": pipeline.clip.error is None, "error": pipeline.clip.error},
        },
        "memory": {"mode": "sentrix-native", "vectorSpaces": ["episodic", "semantic", "visual"]},
        "videoExtraction": "reserved",
        "database": str(DATA_DIR / "sentrix.db"),
    }


@app.get("/api/dashboard")
def dashboard():
    all_facts = store.list_facts(1000)
    return {
        "stats": {
            "assets": store.count("assets"),
            "observations": store.count("observations"),
            "events": store.count("events"),
            "facts": store.count("facts"),
            "persons": store.count("persons"),
            "entities": store.count("entities"),
            "faceClusters": store.count("face_clusters"),
            "relationships": store.count("relationships"),
            "vectors": store.count("memory_vectors"),
        },
        "pendingFacts": len([item for item in all_facts if item["status"] == "pending"]),
        "events": store.list_events(8),
        "observations": store.list_observations(8),
        "facts": store.list_facts(8),
    }


@app.get("/api/events")
def events():
    return {"events": store.list_events(100)}


@app.get("/api/events/{event_id}")
def event_detail(event_id: str):
    value = store.get_event_detail(event_id)
    if not value:
        raise HTTPException(status_code=404, detail="event not found")
    return value


@app.post("/api/events")
def create_event(payload: dict):
    return store.create_event(payload)


@app.patch("/api/events/{event_id}")
def update_event(event_id: str, payload: dict):
    value = store.update_event(event_id, payload)
    if not value:
        raise HTTPException(status_code=404, detail="event not found")
    return value


@app.get("/api/persons")
def persons():
    rows = store.list_persons()
    return {"persons": [{**row, "display_name": row["name"], "confirmed": row["status"] == "confirmed"} for row in rows]}


@app.get("/api/persons/{person_id}")
def person_detail(person_id: str):
    value = store.get_person(person_id)
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.get("/api/entities")
def entities(status: str | None = None):
    return {"entities": store.list_entities(status)}


@app.get("/api/entities/{entity_id}")
def entity_detail(entity_id: str):
    value = store.get_entity_detail(entity_id)
    if not value:
        raise HTTPException(status_code=404, detail="entity not found")
    return value


@app.get("/api/face-clusters")
def face_clusters(status: str | None = None):
    return {"clusters": store.list_face_clusters(status)}


@app.post("/api/face-clusters/{cluster_id}/confirm")
def confirm_face_cluster(cluster_id: str, payload: dict):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="entity name is required")
    value = store.confirm_face_cluster(cluster_id, name, str(payload.get("family_role") or "").strip() or None)
    if not value:
        raise HTTPException(status_code=404, detail="face cluster not found")
    return value


@app.post("/api/face-clusters/{cluster_id}/reject")
def reject_face_cluster(cluster_id: str):
    value = store.reject_face_cluster(cluster_id)
    if not value:
        raise HTTPException(status_code=404, detail="face cluster not found")
    return value


@app.get("/api/relationships")
def relationships():
    entities = store.list_entities()
    values = store.list_relationships()
    nodes = [{"id": entity["id"], "label": entity["canonical_name"], "status": entity["status"], "entity_type": entity["entity_type"]} for entity in entities]
    edges = [{"source": item["subject_entity_id"], "target": item["object_entity_id"], "label": item["predicate"], "status": item["status"], "id": item["id"]} for item in values]
    return {"nodes": nodes, "edges": edges, "relationships": values}


@app.post("/api/relationships")
def create_relationship(payload: dict):
    required = (payload.get("subject_entity_id"), payload.get("predicate"), payload.get("object_entity_id"))
    if not all(required):
        raise HTTPException(status_code=400, detail="subject_entity_id, predicate and object_entity_id are required")
    value = store.create_relationship(*required, payload.get("evidence_ids") or [], float(payload.get("confidence", 0.5) or 0.5), payload.get("status", "pending"))
    return value


@app.post("/api/relationships/{relationship_id}/confirm")
def confirm_relationship(relationship_id: str):
    value = store.confirm_relationship(relationship_id)
    if not value:
        raise HTTPException(status_code=404, detail="relationship not found")
    return value


@app.post("/api/persons/{person_id}/confirm")
def confirm_person(person_id: str, payload: dict | None = None):
    name = (payload or {}).get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="person name is required")
    value = store.update_person(person_id, name, "confirmed")
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.post("/api/persons/{person_id}/reject")
def reject_person(person_id: str):
    value = store.update_person(person_id, status="rejected")
    if not value:
        raise HTTPException(status_code=404, detail="person not found")
    return value


@app.get("/api/observations")
def observations(assetId: str | None = None, limit: int = 200):
    values = store.list_observations(max(1, min(limit, 1000)))
    if assetId:
        values = [item for item in values if item["asset_id"] == assetId]
    return {"observations": values}


@app.get("/api/observations/{observation_id}")
def observation(observation_id: str):
    value = store.get_observation(observation_id)
    if not value:
        raise HTTPException(status_code=404, detail="observation not found")
    value["asset"] = store.get_asset(value["asset_id"])
    return value


@app.get("/api/assets/{asset_id}")
def asset(asset_id: str):
    value = store.get_asset(asset_id)
    if not value:
        raise HTTPException(status_code=404, detail="asset not found")
    return value


@app.get("/api/assets")
def assets(mediaType: str | None = None, status: str | None = None, limit: int = 200):
    return {"assets": store.list_assets(mediaType, status, max(1, min(limit, 1000)))}


@app.get("/api/assets/{asset_id}/file")
def asset_file(asset_id: str):
    value = store.get_asset(asset_id)
    if not value or not Path(value["path"]).is_file():
        raise HTTPException(status_code=404, detail="asset file not found")
    return FileResponse(value["path"], media_type=value.get("mime_type") or "application/octet-stream", filename=value.get("file_name"))


@app.get("/api/facts")
def facts(status: str | None = None):
    values = store.list_facts(1000)
    return {"facts": [item for item in values if not status or item["status"] == status]}


@app.get("/api/stories")
def stories():
    return {"stories": store.list_stories()}


@app.post("/api/stories")
def create_story(payload: dict):
    event_ids = payload.get("event_ids") or []
    if event_ids and not payload.get("content"):
        evidence = []
        for event_id in event_ids:
            detail = store.get_event_detail(event_id)
            if detail:
                evidence.append({"event": detail["event"], "observations": [{"id": item["id"], "caption": item.get("caption"), "transcript": item.get("transcript"), "asset_id": item.get("asset_id")} for item in detail["observations"]]})
        if evidence:
            prompt = """根据下面的真实家庭事件和证据生成故事初稿。不要补造人物、地点或时间，只能使用证据。严格返回 JSON：title、content、outline（数组）。使用中文，content 300 字以内。证据：""" + str(evidence)
            try:
                generated = parse_json_response(gamma.chat(prompt))
                payload = {**payload, "title": payload.get("title") or generated.get("title"), "content": generated.get("content", ""), "outline": generated.get("outline", [])}
            except Exception:
                pass
    return store.create_story(payload)


@app.patch("/api/stories/{story_id}")
def update_story(story_id: str, payload: dict):
    value = store.update_story(story_id, payload)
    if not value:
        raise HTTPException(status_code=404, detail="story not found")
    return value


@app.delete("/api/stories/{story_id}")
def delete_story(story_id: str):
    value = store.delete_story(story_id)
    if not value:
        raise HTTPException(status_code=404, detail="story not found")
    return value


@app.post("/api/invites")
def create_invite(payload: dict):
    invite = store.create_invite(payload.get("label", "家庭成员"))
    return {**invite, "invite_url": f"sentrix://join/{invite['token']}"}


@app.post("/api/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    result = agent.answer(request.query.strip())
    result["retrievalTrace"] = result.get("retrieval_trace", [])
    return result


@app.post("/api/ingest", status_code=202)
async def ingest(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    safe_name = Path(file.filename or "upload.bin").name
    asset_id = make_id("asset")
    destination = MEDIA_DIR / f"{asset_id}_{safe_name}"
    with destination.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    media_type = (file.content_type or "application/octet-stream").split("/", 1)[0]
    if media_type not in {"image", "audio", "video", "text"}:
        media_type = "text"
    created = store.create_asset(asset_id, safe_name, media_type, str(destination), file.content_type, destination.stat().st_size)
    background_tasks.add_task(process_asset, asset_id)
    return {"accepted": True, "assetId": created["id"], "fileName": safe_name, "status": "queued", "mediaType": media_type}


@app.post("/api/import", status_code=202)
def import_placeholder(request: ImportRequest):
    return {"accepted": True, "assetId": make_id("asset"), "fileName": request.fileName, "status": "queued", "mediaType": request.mediaType}


@app.post("/api/maintenance/recheck")
def recheck(background_tasks: BackgroundTasks):
    assets = [store.get_asset(row["id"]) for row in store._rows("SELECT id FROM assets WHERE status IN ('queued', 'failed') ORDER BY created_at")]
    for item in assets:
        background_tasks.add_task(process_asset, item["id"])
    return {"accepted": len(assets), "status": "recheck-queued"}


@app.post("/api/facts/{fact_id}/confirm")
def confirm_fact(fact_id: str):
    try:
        return store.confirm_fact(fact_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="fact not found")


@app.post("/api/facts/{fact_id}/reject")
def reject_fact(fact_id: str):
    if not store.get_fact(fact_id):
        raise HTTPException(status_code=404, detail="fact not found")
    return store.reject_fact(fact_id)
