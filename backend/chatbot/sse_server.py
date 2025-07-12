import hashlib
import time
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import queue
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

crawl_event_queue = queue.Queue()
event_history = {}

from urllib.parse import urlparse

def make_event_id(event_type, data, source):
    if event_type == "search_started":
        base_data = {
            "event_type": event_type,
            "source": source,
            "query": data.get("query"),
        }
    elif event_type == "urls_found":
        base_data = {
            "event_type": event_type,
            "source": source,
            "urls": tuple(sorted(data.get("urls", []))),
        }
    elif event_type in ("scraping_url", "url_scraped"):
        url = data.get("url")
        domain = data.get("domain")
        # Nếu domain không có, lấy từ url
        if not domain and url:
            try:
                domain = urlparse(url).netloc
            except Exception:
                domain = ""
        base_data = {
            "event_type": event_type,
            "source": source,
            "url": url,
            "domain": domain or "",
        }
    elif event_type == "scraping_complete":
        base_data = {
            "event_type": event_type,
            "source": source,
        }
    else:
        base_data = {
            "event_type": event_type,
            "source": source,
        }
    base = json.dumps(base_data, sort_keys=True)
    return hashlib.md5(base.encode()).hexdigest()

def add_crawl_event(event_type: str, data: dict, source: str = "unknown") -> None:
    print(f"CALL add_crawl_event: {event_type} | data={data} | source={source}")
    try:
        event_id = make_event_id(event_type, data, source)
        if event_id in event_history:
            return
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
            "id": event_id,
            "source": source
        }
        event_history[event_id] = event
        crawl_event_queue.put(event, block=False)
        print(f"ADD EVENT: {event_type} | {data} | {source} | id={event_id}")
    except Exception as e:
        print(f"Error in add_crawl_event: {e}")

@app.post("/publish-event")
async def publish_event(event: dict):
    event_type = event.get("event_type")
    data = event.get("data", {})
    source = event.get("source", "unknown")
    add_crawl_event(event_type, data, source)
    return {"success": True}

async def crawl_event_generator(request: Request):
    while True:
        if await request.is_disconnected():
            break
        try:
            event = crawl_event_queue.get_nowait()
            event_data = json.dumps(event)
            yield f"event: {event['event_type']}\ndata: {event_data}\n\n"
        except queue.Empty:
            yield f": keep-alive\n\n"
            await asyncio.sleep(0.1)

@app.get("/api/crawl-events")
async def get_crawl_events(request: Request):
    return StreamingResponse(
        crawl_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )