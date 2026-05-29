# Daily-AI-Insight-Engine
同花顺笔试AI舆情分析系统

## 1.项目总览
利用Claude Code等AI工具完成AI舆情分析日报。给定爬虫或手动整理舆情资料，对资料进行结构化整理并且形成可视化报告。

## 2.项目目录结构
```text
Daily-AI-Insight-Engine/
├── data/
│   ├── raw_data.json
│   ├── cleaned_data.json
│   └── structured_data.json
│
├── output/
│   ├── report_YYYY-MM-DD.md
│   └── figures/
│       ├── topics_YYYY-MM-DD.png
│       └── sentiment_YYYY-MM-DD.png
│
├── scripts/
│   ├── clean_data.py
│   ├── schema.py
│   ├── report.py
│   ├── visualization.py
│   ├── util.py
│   └── main.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 3.数据源说明
本项目使用 AI 行业相关新闻作为原始数据，主要采集网站为机器之心和Techcrunch作为混合语言语料。相比其他综合平台，机器之心和TechcrunchAI板块文章资源更加聚焦人工智能，以文章的形式提供AI方面最新舆情。相比于微博，知乎等可信度和曝光度更高，更加能够反映客观舆情。但是相比之下没有曝光度数据（点赞，收藏等）因此影响力更难计算，并且机器之心无法获得发布的精准时间，导致计算影响力的时间权重难以计算。当前 MVP 阶段主要采用人工整理后的静态 JSON 数据。

## 4.系统设计思路
原始数据(JSON)
    ↓ clean_data.py
数据清理与总结(JSON)
    ↓ schema.py
结构化提取(JSON)
    ↓ report.py， visualization.py
生成报告与可视化结果(md, png)
### 4.1 链路文字描述
原始数据经过对文本清理，日期标准化，表题与摘要的生成和翻译。标准化数据进而被赋予情感分类，影响力值，具体影响以及主题和关键词，形成结构化数据。通过加权发布时间，情感分类以及影响力值判断重要程度，最后利用结构化数据生成报告和视图。
### 4.2 结构化数据模型
| 字段 | 作用 |
|---|---|
| `id` | 新闻唯一编号 |
| `title` | 新闻标题 |
| `source` | 新闻来源，用于追踪数据出处 |
| `url` | 原文链接 |
| `published_at` | 发布时间，用于计算日报日期和时间权重 |
| `summary` | 新闻核心内容摘要 |
| `key_entities` | 抽取公司、机构、产品、模型、人物等核心实体 |
| `impact` | 对行业、企业或用户影响的自然语言分析 |
| `impact_level` | 影响力等级，1 到 3 |
| `sentiment` | 舆情倾向，-1 负面，0 中性，1 正面 |
| `topic_tags` | 新闻主题标签，用于趋势统计和可视化 |
|
字段设计逻辑在于用更小的数据量进行新闻分类（标签，实体，情感），总结（影响，摘要）以及对最后影响力进行加权计算（情感，影响力，时间）。
## 5 AI 使用方式
本项目使用大模型完成以下任务：
### 5.1 数据清洗阶段
- 英文标题翻译为中文
- 无标题新闻的标题生成
- 新闻摘要生成
### 5.2 结构化抽取阶段
- 主题标签选择
- 关键实体抽取
- 行业影响分析
- 影响力等级判断
- 舆情倾向判断
### 5.3 报告生成阶段
- 基于 Top 5 结构化事件生成趋势判断
### 5.4 提示词设计
使用Claude code以规定人物，输出内容，语义限制生成系统提示词，以及以规定输出格式，输出内容以及数值范围等生成用户提示词。在此基础上对提示词进行微调。例如在标签选择中将可选择标签注入提示词，并且在无法匹配时允许生成。错误处理以检测并且返回默认值为主，对模型JSON返回错误则以返回空值为主。


## 6. 项目运行

本项目基于 Python 实现，建议使用 Python 3.10 及以上版本。

### 6.1 安装依赖

```
pip install -r requirements.txt
```

### 6.2 配置 API Key

在项目根目录创建 `.env` 文件，并写入：

```
OPENAI_API_KEY=your_api_key_here
```

### 6.3 运行完整流程

```
python scripts/main.py
```

执行后会依次完成：

* 数据清洗 → `data/cleaned_data.json`
* 结构化抽取 → `data/structured_data.json`
* 日报生成 → `output/report_YYYY-MM-DD.md`
* 可视化图表 → `output/figures/`

### 6.4 分步骤运行

#### 6.4.1 数据清洗

```
python scripts/clean_data.py
```

输出：

```
data/cleaned_data.json
```

#### 6.4.2 结构化抽取

```
python scripts/schema.py
```

输出：

```
data/structured_data.json
```

#### 6.4.3 日报生成

```
python scripts/report.py
```

输出：

```
output/report_YYYY-MM-DD.md
```

#### 6.4.4 可视化图表

```
python scripts/visualization.py
```

输出：

```
output/figures/topics_YYYY-MM-DD.png
output/figures/sentiment_YYYY-MM-DD.png
```