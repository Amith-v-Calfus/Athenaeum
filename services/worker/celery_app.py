"""Celery application for Athenaeum ingestion workers.

The Go gateway pushes raw JSON jobs onto a plain Redis list. It does NOT speak
Celery's protocol. A small bridge (bridge.py) reads that list and calls the
Celery task defined in tasks.py, which is where the real ingestion work runs.

So the flow is:
    Go gateway  --RPUSH-->  Redis list "ingestion_jobs"
    bridge.py   --BLPOP-->  reads job  --.delay()-->  Celery
    Celery worker           runs tasks.ingest_document
"""

import os

from celery import Celery

# Celery uses Redis both as the broker (task queue) and the result backend.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "athenaeum",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Acknowledge a task only after it finishes, so a worker crash mid-ingestion
    # returns the job to the queue instead of silently losing the document.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)