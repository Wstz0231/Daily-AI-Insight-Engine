import json
import re
from pathlib import Path
from typing import Any, List, Dict
from util import get_openai_client, load_json, save_json


def safe_parse_json(text: str) -> Dict[str, Any]:
    """JSON处理"""
    try:
        return json.loads(text)
    except Exception:
        return {}


def json_call(messages: List[Dict[str, str]], client, model: str = "gpt-4o-mini", temperature: float = 0.2) -> Dict[str, Any]:
    """语言模型JSON包装器"""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    content = (resp.choices[0].message.content or "").strip()
    return safe_parse_json(content)


# 类别
TOPIC_SET: List[str] = [
    "大模型",
    "多模态",
    "自然语言处理",
    "语音",
    "对话机器人",
    "计算机视觉",
    "生成式图像/视频",
    "生成式音频",
    "Agent",
    "RAG",
    "提示工程",
    "安全/对齐",
    "数据管理",
    "开源项目",
    "模型压缩/蒸馏",
    "推理优化/加速",
    "芯片",
    "云平台",
    "边缘/端侧AI",
    "机器人/机械臂",
    "自动驾驶",
    "医疗AI",
    "金融AI",
    "教育AI",
    "工业AI",
    "办公/生产力",
    "搜索/推荐",
    "合规/政策",
]

def compute_category_and_topics_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """标注一个类别，若无则生成"""
    title = title or ""
    summary = summary or ""

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
                "根据标题和摘要，从以下主题集中选择且仅选择1个最相关标签：[" + topics_list + "]。\n"
                "若没有任何合适主题，请自行生成1个简洁中文主题标签（风格与给定主题一致，2-10字）。\n"
                "仅返回JSON：{\n"
                "  \"topic_tags\": [<string: exactly one>]\n"
                "}\n\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)

    topics = res.get("topic_tags") if isinstance(res, dict) else []
    topic: str = ""
    if isinstance(topics, list) and topics:
        cand = topics[0]
        if isinstance(cand, str):
            topic = cand.strip()
    elif isinstance(topics, str):
        topic = topics.strip()

    if not topic:
        topic = "大模型"

    filtered: List[str] = [topic]

    return {"topic_tags": filtered}


def extract_key_entities_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """3-5个实体提取"""
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
                "仅返回JSON：{\n  \"key_entities\": [string,...]\n}\n\n"
                f"标题：{title}\n"
                f"摘要：{summary}"
            ),
        },
    ]

    res = json_call(messages, client)
    ents = res.get("key_entities") if isinstance(res, dict) else []
    if not isinstance(ents, list):
        ents = []
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

    return {"key_entities": out_ents}


def compute_impact_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """生成新闻影响和影响力分数"""
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
                "并给出从1到5的影响强度："
                "1=边缘信息，对市场几乎无影响（如活动预告、人事变动）；"
                "2=细分领域局部影响（如小功能更新、小型融资）；"
                "3=单一领域明显影响（如新模型发布、重要数据集、中型融资）；"
                "4=多领域或行业级影响（如主流平台重大升级、大型收购、重要监管政策）；"
                "5=对整个AI行业有颠覆性影响（如突破性架构、GPT级发布、重大法规出台）。\n"
                "仅返回JSON：{\"impact\": string, \"impact_level\": 1|2|3|4|5}\n\n"
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

    # 限制长度
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
    if level > 5:
        level = 5

    return {"impact": impact_txt, "impact_level": level}


def compute_sentiment_lm(title: str, summary: str, client) -> Dict[str, Any]:
    """生成情感分数1为正面，0为中性，-1为负面"""
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
                "仅返回JSON：{\\n  \\\"sentiment\\\": -1|0|1\\n}\\n\\n"
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

    return {"sentiment": val}


def transform_record(rec: Dict[str, Any], client) -> Dict[str, Any]:
    """合并成为条目"""
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
    for i, r in enumerate(structured, start=1):
        if r.get("id") in (None, ""):
            r["id"] = i
    save_json(structured, output_path)


if __name__ == "__main__":
    run()
