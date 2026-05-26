# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # RAG Agent Chain - 三高临床诊疗指南问答系统
# MAGIC
# MAGIC **项目**: 医疗RAG知识库  
# MAGIC **功能**: 基于 Vector Search 检索 + LLM 生成，构建临床诊疗问答Agent  
# MAGIC **架构**: VectorSearchRetrieverTool + langgraph create_react_agent + MLflow日志  
# MAGIC **LLM**: `databricks-qwen3-next-80b-a3b-instruct`（原生中文）  
# MAGIC **检索**: `medical.medical_knowledge.chunks_index`
# MAGIC
# MAGIC ### 流程
# MAGIC ```
# MAGIC 用户问题 → Vector Search检索相关chunks → LLM基于检索内容生成回答 → 返回结构化答案(含来源引用)
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Step 1: 安装依赖
# MAGIC %pip install databricks-langchain langchain langchain-core langgraph mlflow --upgrade -q
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Step 2: 导入和配置
import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
)
from langgraph.prebuilt import create_react_agent

# === 配置 ===
LLM_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
VS_INDEX = "medical.medical_knowledge.chunks_index"
VS_ENDPOINT = "medical-rag-endpoint"

print(f"LLM: {LLM_ENDPOINT}")
print(f"Vector Index: {VS_INDEX}")
print(f"VS Endpoint: {VS_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Step 3: 创建检索工具
# 创建 Vector Search 检索工具
vs_tool = VectorSearchRetrieverTool(
    index_name=VS_INDEX,
    # 返回的字段
    columns=[
        "chunk_id", 
        "guideline_name", 
        "section", 
        "content", 
        "content_type",
        "char_count"
    ],
    # 检索参数
    num_results=5,
    # 工具描述（Agent用来决定何时调用）
    tool_name="search_medical_guidelines",
    tool_description=(
        "搜索临床诊疗指南知识库。"
        "包含糖尿病、高血压、血脂异常等三高相关的"
        "中国临床实践指南和专家共识。9份权威文献，"
        "覆盖诊断标准、治疗方案、药物选择、分级管理等。"
        "当用户提问与糖尿病、高血压、血脂相关的临床问题时使用此工具。"
    ),
)

print(f"✅ 检索工具已创建: {vs_tool.tool_name}")
print(f"   返回字段: {vs_tool.columns}")
print(f"   Top-K: {vs_tool.num_results}")

# COMMAND ----------

# DBTITLE 1,Step 4: 创建  Agent
# 创建 LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

# 系统提示词 - 定义Agent行为
SYSTEM_PROMPT = """你是一个专业的临床诊疗助手，专注于“三高”（高血压、高血脂、糖尿病）的临床指南问答。

## 工作原则
1. **必须基于检索到的指南内容回答**，不要使用检索结果之外的知识。
2. 每次回答必须调用 search_medical_guidelines 工具检索相关内容。
3. 回答格式要求：
   - 用清晰的结构化格式（分点、表格）
   - 在回答末尾标注来源：【来源：XX指南，第X章】
   - 如果检索结果不足以回答，如实告知用户。
4. 语言：始终用中文回答。
5. 重要：临床建议仅供参考，不能替代医生诊断。

## 知识库范围
- 中国糖尿病防治指南（2024版）
- 中国高血压临床实践指南（2024版）
- 国家基层高血压防治管理指南（2025版）
- 糖尿病患者血脂管理中国专家共识（2024版）
- 中国血脂管理指南（基层版2024年）
- 基层血脂管理适宜技术与质量控制中国专家建议（2025年）
- 2型糖尿病患者泛血管疾病风险评估与管理中国专家共识（2022版）
- 糖尿病分型诊断中国专家共识临床实践应用
- 内分泌性高血压筛查专家共识（2025版）
"""

# 创建 Agent（使用 langgraph create_react_agent）
agent = create_react_agent(llm, tools=[vs_tool], prompt=SYSTEM_PROMPT)

print("\u2705 RAG Agent 已创建")
print(f"   LLM: {LLM_ENDPOINT}")
print(f"   Tools: [search_medical_guidelines]")
print(f"   Framework: langgraph create_react_agent")

# COMMAND ----------

# DBTITLE 1,Step 5: 封装推理函数
# 封装为可部署的函数
def predict(messages: list) -> str:
    """
    RAG Agent 推理入口。
    输入: [{"role": "user", "content": "问题"}]
    输出: 结构化回答字符串
    """
    from langchain_core.messages import HumanMessage, AIMessage

    input_msgs = []
    for msg in messages:
        if msg["role"] == "user":
            input_msgs.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            input_msgs.append(AIMessage(content=msg["content"]))

    response = agent.invoke({"messages": input_msgs})
    # 提取最后一条AI消息
    for m in reversed(response["messages"]):
        if hasattr(m, 'type') and m.type == 'ai' and m.content:
            return m.content
    return "抱歉，无法生成回答。"

print("\u2705 predict() 函数已定义")

# COMMAND ----------

# DBTITLE 1,Step 6: 测试RAG Agent
# 测试 RAG Agent - 临床问诊场景
test_questions = [
    "2型糖尿病的诊断标准是什么？空腹血糖和OGTT分别是多少？",
    "高血压患者合并糖尿病，降压目标值应该是多少？推荐哪类降压药？",
    "糖尿病患者的LDL-C控制目标是什么？如何分层管理？",
]

print("=" * 70)
print("RAG Agent 测试 - 三高临床问诊")
print("=" * 70)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*70}")
    print(f"👨‍⚕️ 问题 {i}: {question}")
    print("-" * 70)
    
    messages = [{"role": "user", "content": question}]
    answer = predict(messages)
    
    print(f"\n🤖 回答:\n{answer}")
    print(f"\n{'='*70}")

# COMMAND ----------

# DBTITLE 1,Step 7: MLflow日志记录
import mlflow
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

# 设置实验
experiment_name = "/Users/cici@caoxx1018gmail.onmicrosoft.com/medical/medical-rag-agent"
mlflow.set_experiment(experiment_name)

# agent_code.py 已实现为 ChatModel + ChatCompletionResponse
# （正确的可部署版本，与 06 一致）
agent_code_path = "/Workspace/Users/cici@caoxx1018gmail.onmicrosoft.com/medical/agent_code.py"
print(f"\u2705 Agent代码路径: {agent_code_path}")

# 声明依赖资源（Serving Endpoint 的 Service Principal 需要这些权限）
resources = [
    DatabricksVectorSearchIndex(index_name="medical.medical_knowledge.chunks_index"),
    DatabricksServingEndpoint(endpoint_name="databricks-qwen3-next-80b-a3b-instruct"),
]

input_example = {
    "messages": [
        {"role": "user", "content": "2型糖尿病的诊断标准是什么？"}
    ]
}

# 使用 pyfunc.log_model（与06一致的方式）
with mlflow.start_run(run_name="medical-rag-agent-v1"):
    mlflow.log_params({
        "llm_endpoint": LLM_ENDPOINT,
        "vs_index": VS_INDEX,
        "vs_endpoint": VS_ENDPOINT,
        "embedding_model": "databricks-qwen3-embedding-0-6b",
        "num_results": 5,
        "source_table": "medical.medical_knowledge.chunks",
        "total_chunks": 1095,
    })

    logged_model = mlflow.pyfunc.log_model(
        python_model=agent_code_path,
        name="model",
        input_example=input_example,
        pip_requirements=[
            "databricks-langchain",
            "langchain",
            "langchain-core",
            "langgraph",
            "mlflow",
        ],
        resources=resources,
    )

    print(f"\u2705 Agent 已记录到 MLflow (pyfunc + ChatModel)")
    print(f"   Experiment: {experiment_name}")
    print(f"   Model URI: {logged_model.model_uri}")
    print(f"\n下一步:")
    print(f"   1. 运行 05_agent_evaluation 评估质量")
    print(f"   2. 运行 06_register_and_deploy 注册部署")

# COMMAND ----------

# DBTITLE 1,完成说明
# MAGIC %md
# MAGIC ## 完成
# MAGIC
# MAGIC RAG Agent Chain 已构建完成：
# MAGIC
# MAGIC | 组件 | 配置 |
# MAGIC |------|------|
# MAGIC | 检索 | VectorSearchRetrieverTool → `chunks_index` (Top-5) |
# MAGIC | LLM | `databricks-qwen3-next-80b-a3b-instruct` |
# MAGIC | Agent框架 | `langgraph.prebuilt.create_react_agent` |
# MAGIC | 日志 | MLflow pyfunc (models-from-code, ChatModel) |
# MAGIC | 资源声明 | DatabricksVectorSearchIndex + DatabricksServingEndpoint |
# MAGIC
# MAGIC ### 架构图
# MAGIC ```
# MAGIC 用户问题
# MAGIC     ↓
# MAGIC create_react_agent
# MAGIC     ↓ (决定调用工具)
# MAGIC VectorSearchRetrieverTool
# MAGIC     ↓ (query_text → cosine similarity)
# MAGIC chunks_index → Top-5 chunks
# MAGIC     ↓
# MAGIC LLM 基于 chunks 生成回答
# MAGIC     ↓
# MAGIC 结构化中文回答 (含来源标注)
# MAGIC ```
# MAGIC
# MAGIC ### 下一步
# MAGIC - `05_agent_evaluation`: 评估 Agent 回答质量
# MAGIC - `06_register_and_deploy`: 注册到 UC 并部署为 Serving Endpoint