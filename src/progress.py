import json, time, pathlib

PROGRESS_PATH = pathlib.Path("progress/lesson_progress.json")
RECEIPT_PATH = pathlib.Path("progress/receipt.json")


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {
        "quiz": {},
        "metrics": {},
        "last_updated": int(time.time()),
    }


def save_progress(obj):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    obj["last_updated"] = int(time.time())
    PROGRESS_PATH.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_receipt(status, metrics):
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": int(time.time()), "status": status, "metrics": metrics}
    RECEIPT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload
