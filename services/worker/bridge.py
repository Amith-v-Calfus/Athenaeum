"""Bridge: Go's raw Redis list  ->  Celery task.
 
The Go gateway keeps things simple: it RPUSHes a JSON job onto a plain Redis
list ("ingestion_jobs"). Celery cannot consume that list directly because it
expects its own message envelope. This bridge is the tiny adapter in between.
 
It blocks on the list (BLPOP), validates that each item is well-formed JSON
matching the ingestion-job contract, and hands it to the Celery task with
.delay(). Keeping this separate means Go never needs to know anything about
Celery, and Celery never needs to know anything about Go.
 
Run it as its own process alongside the Celery worker:
    python bridge.py
"""

import json
import logging
import os
import redis

from tasks import ingest_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(message)s")
log=logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "ingestion_jobs")

# The fields the Go gateway promises to send (see shared/schemas/ingestion_job.schema.json).
REQUIRED_FIELDS = {
    "job_id",
    "original_filename",
    "storage_path",
    "content_type",
    "size_bytes",
    "uploaded_at",
    "source",
}

def is_valid_job(job:dict)->bool:
    """Small structural check before the job matches the contract before the dispatch."""
    missing=REQUIRED_FIELDS-job.keys()
    if missing:
        log.warning("dropping job: missing fields %s",sorted(missing))
        return False
    return True

def main() -> None:
    client = redis.Redis.from_url(REDIS_URL)
    log.info("bridge started; watching Redis list %r", QUEUE_NAME)

    while True:
        # BLPOP blocks server-side until an item is available. The redis-py
        # client also has its own socket read timeout, which fires and raises
        # if the wait is long enough -- that is a client-library quirk, not a
        # real failure, so we catch it here and just loop back to waiting.
        try:
            item = client.blpop(QUEUE_NAME, timeout=30)
        except redis.exceptions.TimeoutError:
            continue
        except redis.exceptions.ConnectionError:
            log.warning("redis connection dropped, retrying...")
            continue

        if item is None:
            # blpop's own timeout elapsed with nothing arriving -- normal, keep waiting.
            continue

        _key, raw = item
        try:
            job = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("dropping job: not valid JSON: %r", raw[:200])
            continue

        if not isinstance(job, dict) or not is_valid_job(job):
            continue

        # Hand off to Celery. The bridge does no ingestion work itself.
        ingest_document.delay(job)
        log.info("dispatched job %s (%s)", job["job_id"], job["original_filename"])
        
if __name__ == "__main__":
    main()