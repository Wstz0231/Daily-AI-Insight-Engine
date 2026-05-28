import json
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, date


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_env_simple() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        if k and k not in os.environ:
            os.environ[k.strip()] = v.strip()


def get_openai_client():
    _read_env_simple()
    from openai import OpenAI
    return OpenAI()


def parse_yyyy_mm_dd(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def sentiment_label(v: Optional[int]) -> str:
    if v == 1:
        return "正面"
    if v == -1:
        return "负面"
    return "中性"
