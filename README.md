# 三高临床诊疗指南 RAG 问答系统

## 项目概述

在 Azure Databricks 平台构建的三高（高血压、高血脂、糖尿病）临床诊疗指南问答系统。基于 RAG（Retrieval-Augmented Generation）架构，从 9 份权威临床指南 PDF 中检索相关内容，由大语言模型生成专业回答。目标群体为医疗工作者和相关医疗人员。

---

## 整体架构

```
PDF (Volume)
  → PyMuPDF / ai_parse_document 解析
    → chunks 表 (Delta, CDF enabled, 1095行)
      → Delta Sync Index (自动 embedding by qwen-0.6b, 1024维)
        |
      Query → embedding → 向量检索 Top-5 chunks
        → LLM (qwen-80b) 生成回答
          → ChatModel 包装为标准 ChatCompletion 格式
            → Serving Endpoint (agents.deploy)
              → Playground / API 调用
```

评估分支：
```
评估数据集 (8个问题 + 参考答案)
  → Agent 逐条生成回答 (create_react_agent + VectorSearch)
    → LLM Judge 对每条打分 (correctness/safety/groundedness)
      → MLflow 记录 metrics
```

---

## 核心概念说明

### 1. Delta Sync Index + Managed Embeddings

你不需要手动写代码生成 embedding。创建 index 时指定 embedding 模型和源表列，Databricks 在 sync 时自动调用 qwen embedding 模型对 content 列生成向量。当源表数据更新后，触发 sync 即可增量更新向量。

### 2. LLM-as-a-Judge 评估范式

评估不是用 embedding 做向量相似度对比。`mlflow.evaluate(model_type="databricks-agent")` 调用一个 LLM Judge（评判模型），它读取问题 + Agent回答 + 参考答案，像人类评审员一样给出 correctness/relevance/groundedness 的评分。

### 3. log_model vs register_model vs deploy

| 操作 | 位置 | 含义 |
| --- | --- | --- |
| `mlflow.pyfunc.log_model()` | MLflow Experiment | 保存模型 artifact，开发阶段，可有多个版本 |
| `mlflow.register_model()` | Unity Catalog | 正式注册，有版本号和别名（Champion），可部署 |
| `agents.deploy()` | Serving Endpoint | 创建在线服务，提供 API 接口 |

### 4. ChatModel 封装的作用

`create_react_agent` 返回的是 langgraph 内部 state 格式 (`{"messages": [...]}`)，而 Playground/API 期望标准 OpenAI ChatCompletion 格式 (`{"choices": [{"message": {...}}]}`)。`ChatModel.predict()` 是这个转换桥梁。

### 5. Service Principal 权限模型

Notebook 中你用自己的身份操作（owner 权限），但 Serving Endpoint 运行在自动创建的 Service Principal 下。通过 `resources` 声明告诉 `agents.deploy()` 需要访问哪些资源，框架自动为 SP 授权。

---

## 流程详解

### Step 1: 数据准备 (Notebook 01 + 02)

| Notebook | 处理方式 | 输入 | 输出 |
| --- | --- | --- | --- |
| 01_pdf_parsing_and_chunking | PyMuPDF (文本型PDF) | 6份PDF | 875 chunks |
| 02_ocr_pdf_parsing | ai_parse_document v2.0 (扫描PDF/OCR) | 3份PDF | 220 chunks |

- 原始 PDF 存储位置: `/Volumes/medical/medical_knowledge/raw_pdfs/`
- 分块策略: 300-800字，标题层级识别（4级），表格完整保留，参考文献排除
- 最终输出: `medical.medical_knowledge.chunks` (1,095 行, PK: chunk_id, CDF enabled)

### Step 2: 向量索引创建 (Notebook 03)

- **Vector Search Endpoint**: `medical-rag-endpoint` (STANDARD 类型)
- **Delta Sync Index**: `medical.medical_knowledge.chunks_index`
  - Pipeline: TRIGGERED (手动触发同步)
  - Embedding: `databricks-qwen3-embedding-0-6b` (1024维, 多语言/中文优化)
  - Managed Embeddings 在 `content` 列上
- 前置条件: 源表启用 CDF (`delta.enableChangeDataFeed = true`) + Primary Key

### Step 3: RAG Agent 构建 (Notebook 04)

- 检索工具: `VectorSearchRetrieverTool` (Top-5 检索, cosine similarity)
- LLM: `databricks-qwen3-next-80b-a3b-instruct`
- Agent 框架: `langgraph.prebuilt.create_react_agent`
- System Prompt: 强制基于检索内容回答、中文输出、标注来源
- MLflow 日志: `mlflow.pyfunc.log_model()` + `resources` 声明 + `pip_requirements`
- 模型代码: `agent_code.py` (ChatModel + ChatCompletionResponse 封装)

### Step 4: Agent 评估 (Notebook 05)

- 评估数据集: 8 个典型三高临床问题 + 参考答案
- 方法: `mlflow.evaluate(model_type="databricks-agent")`
- 评分维度: correctness, safety, groundedness, chunk_relevance
- 同一 Experiment 下记录评估 run，与模型 run 对比

### Step 5: 注册与部署 (Notebook 06)

1. `mlflow.pyfunc.log_model()` — ChatModel 代码 + 依赖 + 资源声明记录到 MLflow
2. `mlflow.register_model()` — 注册到 Unity Catalog (`medical.medical_knowledge.rag_agent`)
3. `client.set_registered_model_alias("Champion")` — 设置生产别名
4. `agents.deploy()` — 部署为 Serving Endpoint (自动处理容器、权限、SP 授权)

---

## 部署踩坑记录

### 问题 1: ModuleNotFoundError: databricks_langchain

- **原因**: `log_model()` 未声明 `pip_requirements`，MLflow 自动依赖检测失败
- **修复**: 显式添加 `pip_requirements=["databricks-langchain", "langchain", "langgraph", ...]`

### 问题 2: PermissionDenied: chunks_index

- **原因**: Serving Endpoint 运行在 Service Principal 下，不是你的身份
- **修复**: `log_model()` 中添加 `resources` 参数声明 `DatabricksVectorSearchIndex` + `DatabricksServingEndpoint`

### 问题 3: Playground 返回 schema 不兼容

- **原因**: `create_react_agent` 返回 langgraph state 格式，缺少标准 `choices`/`role` 字段
- **修复**: 用 `mlflow.pyfunc.ChatModel` 封装，`predict()` 返回 `ChatCompletionResponse`

---

## 关键资源清单

| 类别 | 名称 | 说明 |
| --- | --- | --- |
| Catalog/Schema | `medical.medical_knowledge` | 项目命名空间 |
| 源表 | `medical.medical_knowledge.chunks` | 1,095行, 9份指南分块 |
| 向量索引 | `medical.medical_knowledge.chunks_index` | Delta Sync, 1024维 |
| VS Endpoint | `medical-rag-endpoint` | STANDARD, 持续计费 |
| UC 模型 | `medical.medical_knowledge.rag_agent` | Champion 别名 |
| Serving Endpoint | `medical-rag-agent-endpoint` | scale_to_zero=True |
| LLM | `databricks-qwen3-next-80b-a3b-instruct` | 生成模型 |
| Embedding | `databricks-qwen3-embedding-0-6b` | 向量化模型 |
| Agent 代码 | `agent_code.py` | ChatModel 实现 |
| 原始数据 | `/Volumes/medical/medical_knowledge/raw_pdfs/` | 9份PDF |

---

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 平台 | Azure Databricks |
| 存储 | Delta Lake + Unity Catalog |
| 向量检索 | Databricks Vector Search (Delta Sync + Managed Embeddings) |
| Agent 框架 | LangGraph (`create_react_agent`) |
| LLM 接口 | `databricks-langchain` (`ChatDatabricks`) |
| 模型管理 | MLflow + Unity Catalog Model Registry |
| 部署 | `databricks-agents` (`agents.deploy()`) |
| 评估 | MLflow Evaluate (LLM-as-a-Judge) |
| PDF 解析 | PyMuPDF (文本型) + `ai_parse_document` (扫描型) |

---

## 费用结构

| 组件 | 计费方式 | 空闲时 |
| --- | --- | --- |
| Vector Search Endpoint | 持续计费 (~$0.12-$0.37/天) | 仍然收费 |
| Model Serving Endpoint | 按请求计费 | scale_to_zero, 不收费 |
| LLM 推理 | Pay-per-token | 不收费 |
| Embedding | Pay-per-token (sync时) | 不收费 |
| Delta 存储 | 按存储量 | 极少 |

---

## Notebook 索引

| 编号 | 名称 | 职责 |
| --- | --- | --- |
| 01 | pdf_parsing_and_chunking | 文本型PDF解析+分块 (PyMuPDF) |
| 02 | ocr_pdf_parsing | 扫描PDF OCR解析+分块 (ai_parse_document) |
| 03 | vector_search_index | VS Endpoint + Delta Sync Index 创建 |
| 04 | rag_agent_chain | Agent 构建 + 测试 + MLflow log_model |
| 05 | agent_evaluation | LLM-as-a-Judge 评估 |
| 06 | register_and_deploy | UC注册 + Endpoint部署 |

---

## 文件结构

```
medical/
├── README.md                    # 本文件
├── agent_code.py                # ChatModel 实现（部署用）
├── 01_pdf_parsing_and_chunking  # PDF解析
├── 02_ocr_pdf_parsing           # OCR解析
├── 03_vector_search_index       # 向量索引
├── 04_rag_agent_chain           # Agent构建+测试
├── 05_agent_evaluation          # 评估
├── 06_register_and_deploy       # 注册部署
└── medical-rag-agent/           # MLflow Experiment
```
