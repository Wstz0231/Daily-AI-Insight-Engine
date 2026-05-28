import json
import os
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, date


def load_json(path: Path) -> Any:
    """读取JSON"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """写入JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_env_simple() -> None:
    """读取环境变量"""
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
    """返回OPENAI client"""
    _read_env_simple()
    from openai import OpenAI
    return OpenAI()


def parse_yyyy_mm_dd(s: Optional[str]) -> Optional[date]:
    # 处理时间
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def sentiment_label(v: Optional[int]) -> str:
    # 情感标签
    if v == 1:
        return "正面"
    if v == -1:
        return "负面"
    return "中性"
