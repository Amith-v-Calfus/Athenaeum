import json
import logging
import os
import redis
from dotenv import load_dotenv
from tasks import ingest_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [bridge] %(message)s")
log=logging.getLogger(__name__)
load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "ingestion_jobs")

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
        try:
            item = client.blpop(QUEUE_NAME, timeout=30)
        except redis.exceptions.TimeoutError:
            continue
        except redis.exceptions.ConnectionError:
            log.warning("redis connection dropped, retrying...")
            continue

        if item is None:
            continue

        _key, raw = item
        try:
            job = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("dropping job: not valid JSON: %r", raw[:200])
            continue

        if not isinstance(job, dict) or not is_valid_job(job):
            continue

        ingest_document.delay(job)
        log.info("dispatched job %s (%s)", job["job_id"], job["original_filename"])
        
if __name__ == "__main__":
    main()