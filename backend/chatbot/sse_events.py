import hashlib
import time
import json
import queue
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse

crawl_event_queue = queue.Queue()
event_history = {}

def make_event_id(event_type, data, source):
    base = json.dumps({
        "event_type": event_type,
        "data": data,
        "source": source
    }, sort_keys=True)
    return hashlib.md5(base.encode()).hexdigest()

def add_crawl_event(event_type: str, data: dict, source: str = "unknown") -> None:
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