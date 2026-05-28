import json
import re
import os
from pathlib import Path
from typing import Any, List, Dict


def load_json(path: Path) -> Any:
    """Load JSON from a file path (UTF-8)."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    """Save JSON to a file path (UTF-8, pretty). Creates parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_env_simple() -> None:
    """Minimal .env loader from project root; silent if missing."""
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


def get_openai_client():
    """Initialize OpenAI client after a simple .env load (no strict checks)."""
    _read_env_simple()
    from openai import OpenAI
    return OpenAI()


def safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def json_call(messages: List[Dict[str, str]], client, model: str = "gpt-4o-mini", temperature: float = 0.2) -> Dict[str, Any]:
    """Call Chat Completions expecting JSON object in the message content."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    content = (resp.choices[0].message.content or "").strip()
    return safe_parse_json(content)


# Constants for schema extraction
CATEGORIES: List[str] = ["技术", "产品", "政策", "资本", "产业"]
TOPIC_SET: List[str] = [
    "大模型/LLM", "多模态", "开源", "芯片/算力", "云服务", "办公助手/生产力",
    "合规/政策", "医疗", "语音/对话", "搜索", "机器人", "边缘/端侧",
]


# Stubs for model-powered computations
def compute_category_and_topics_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """Use the LLM to assign a single category and 0-6 topic tags from fixed sets."""
    title = title or ""
    summary = summary or ""

    allowed_categories = ", ".join(f'"{c}"' for c in CATEGORIES)
    topics_list = ", ".join(f'"{t}"' for t in TOPIC_SET)

    messages = [
        {
            "role": "system",
            "content": (
                "你是新闻分类与主题标注助手。严格按照要求输出JSON，不要任何说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                "根据标题和摘要，完成两件事：\n"
                "1) 在以下分类中选择且仅选择一个：[" + allowed_categories + "]。\n"
                "2) 在以下主题集中选择0-6个最相关标签（可少于3个）：[" + topics_list + "]。\n"
                "请仅返回JSON：{\n"
                "  \"category\": <string>,\n"
                "  \"topic_tags\": <array of strings>,\n"
                "  \"confidence\": <number 0..1>\n"
                "}\n\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)

    # Extract and validate category
    cat = res.get("category") if isinstance(res, dict) else None
    if not isinstance(cat, str) or cat not in CATEGORIES:
        # Small normalization for common variants
        norm = (cat or "").strip()
        mapping = {
            "政策法规": "政策",
            "产业链": "产业",
            "行业": "产业",
            "产品发布": "产品",
            "技术研究": "技术",
            "资本市场": "资本",
        }
        cat = mapping.get(norm, "技术")

    # Extract and validate topics
    topics = res.get("topic_tags") if isinstance(res, dict) else []
    if not isinstance(topics, list):
        topics = []
    # Keep only known topics, preserve order, limit to 6
    seen = set()
    filtered: List[str] = []
    for t in topics:
        if isinstance(t, str) and t in TOPIC_SET and t not in seen:
            filtered.append(t)
            seen.add(t)
        if len(filtered) >= 6:
            break

    conf = res.get("confidence") if isinstance(res, dict) else 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    return {"category": cat, "topic_tags": filtered, "confidence": conf}


def extract_key_entities_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """Use the LLM to extract 3-5 key entities (companies/products/models/persons)."""
    title = title or ""
    summary = summary or ""

    messages = [
        {
            "role": "system",
            "content": (
                "你是信息抽取助手。只输出JSON对象，不要任何额外说明。"
            ),
        },
        {
            "role": "user",
            "content": (
                "从标题和摘要中提取3-5个关键实体（公司/机构/产品/模型/人物），"
                "使用常见中文或官方英文简称，不要附加描述或标点。\n"
                "仅返回JSON：{\n  \"key_entities\": [string,...],\n  \"confidence\": 0..1\n}\n\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)
    ents = res.get("key_entities") if isinstance(res, dict) else []
    if not isinstance(ents, list):
        ents = []
    # Clean, dedupe, and cap to 5
    seen = set()
    out_ents: List[str] = []
    for e in ents:
        if not isinstance(e, str):
            continue
        norm = e.strip()
        if not norm or norm in seen:
            continue
        out_ents.append(norm)
        seen.add(norm)
        if len(out_ents) >= 5:
            break

    conf = res.get("confidence") if isinstance(res, dict) else 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0

    return {"key_entities": out_ents, "confidence": conf}


def compute_impact_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """Use the LLM to generate a 1–2 sentence Chinese impact statement and a 1–3 level."""
    title = title or ""
    summary = summary or ""

    messages = [
        {
            "role": "system",
            "content": "你是科技新闻分析员。只输出JSON对象，不要任何额外说明。",
        },
        {
            "role": "user",
            "content": (
                "基于标题和摘要，用1-2句中文概述该事件对行业/企业/用户的影响，"
                "并给出影响强度1(弱)/2(中)/3(强)与置信度(0..1)。\n"
                "仅返回JSON：{\\n  \\\"impact\\\": string,\\n  \\\"impact_level\\\": 1|2|3,\\n  \\\"confidence\\\": 0..1\\n}\\n\\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)
    impact_txt = res.get("impact") if isinstance(res, dict) else None
    if not isinstance(impact_txt, str) or not impact_txt.strip():
        impact_txt = "影响待评估。"
    impact_txt = impact_txt.strip()

    # Limit to 1-2 sentences (rough split on common punctuation)
    parts = [p for p in re.split(r"[。！？.!?]", impact_txt) if p.strip()]
    if len(parts) > 2:
        impact_txt = "。".join(parts[:2]) + "。"

    level = res.get("impact_level") if isinstance(res, dict) else 2
    try:
        level = int(level)
    except Exception:
        level = 2
    if level < 1:
        level = 1
    if level > 3:
        level = 3

    conf = res.get("confidence") if isinstance(res, dict) else 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0

    return {"impact": impact_txt, "impact_level": level, "confidence": conf}


def compute_sentiment_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """Use the LLM to assign sentiment: -1 (negative), 0 (neutral), 1 (positive)."""
    title = title or ""
    summary = summary or ""

    messages = [
        {
            "role": "system",
            "content": "你是舆情分析助手。只输出JSON对象，不要任何额外说明。",
        },
        {
            "role": "user",
            "content": (
                "判断该新闻的舆情倾向：-1=负面, 0=中性, 1=正面，并给出置信度(0..1)。\n"
                "仅返回JSON：{\\n  \\\"sentiment\\\": -1|0|1,\\n  \\\"confidence\\\": 0..1\\n}\\n\\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)
    val = res.get("sentiment") if isinstance(res, dict) else 0
    if isinstance(val, (int, float)):
        try:
            val = int(round(val))
        except Exception:
            val = 0
    else:
        val = 0
    if val < -1:
        val = -1
    if val > 1:
        val = 1

    conf = res.get("confidence") if isinstance(res, dict) else 0.0
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    if conf < 0:
        conf = 0.0
    if conf > 1:
        conf = 1.0

    return {"sentiment": val, "confidence": conf}


def transform_record(rec: Dict[str, Any], client) -> Dict[str, Any]:
    """Compose the target schema for a single cleaned record."""
    title = rec.get("title", "")
    summary = rec.get("summary", "")

    cat_topics = compute_category_and_topics_lm(title, summary, client)
    ents = extract_key_entities_lm(title, summary, client)
    impact = compute_impact_lm(title, summary, client)
    senti = compute_sentiment_lm(title, summary, client)

    out: Dict[str, Any] = {
        "id": rec.get("id"),
        "title": title,
        "source": rec.get("source", ""),
        "url": rec.get("url", ""),
        "published_at": rec.get("published_at", ""),
        "category": cat_topics.get("category", "技术"),
        "summary": summary,
        "key_entities": ents.get("key_entities", []),
        "impact": impact.get("impact", "影响待评估。"),
        "impact_level": impact.get("impact_level", 2),
        "sentiment": senti.get("sentiment", 0),
        "topic_tags": cat_topics.get("topic_tags", []),
    }
    return out


def run(input_path: Path = Path("data/cleaned_data.json"), output_path: Path = Path("data/structured_data.json")) -> None:
    client = get_openai_client()
    items = load_json(input_path)
    structured: List[Dict[str, Any]] = []
    for rec in items:
        structured.append(transform_record(rec, client))
    # Assign sequential ids for any missing ones
    for i, r in enumerate(structured, start=1):
        if r.get("id") in (None, ""):
            r["id"] = i
    save_json(structured, output_path)


if __name__ == "__main__":
    run()
