# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # Agent Evaluation - RAG回答质量评估
# MAGIC
# MAGIC **项目**: 医疗RAG知识库 - 三高临床诊疗指南问答系统  
# MAGIC **功能**: 使用 `mlflow.evaluate()` 评估 Agent 回答的准确性、相关性、接地性  
# MAGIC **评估维度**:
# MAGIC - **Correctness** - 回答与参考答案是否一致
# MAGIC - **Relevance** - 回答是否与问题相关
# MAGIC - **Groundedness** - 回答是否基于检索到的上下文（无幻觉）
# MAGIC - **Chunk Relevance** - 检索到的chunks是否与问题相关

# COMMAND ----------

# DBTITLE 1,Step 1: 安装依赖
# MAGIC %pip install databricks-langchain langchain langchain-core langgraph mlflow databricks-agents --upgrade -q
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Step 2: 配置
import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent

# 配置
LLM_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
VS_INDEX = "medical.medical_knowledge.chunks_index"

mlflow.set_registry_uri("databricks-uc")
experiment_name = "/Users/cici@caoxx1018gmail.onmicrosoft.com/medical/medical-rag-agent"
mlflow.set_experiment(experiment_name)

print("\u2705 配置完成")

# COMMAND ----------

# DBTITLE 1,Step 3: 构建评估数据集
# 构建评估数据集 - 覆盖三高各领域的典型临床问题
eval_data = pd.DataFrame([
    {
        "request": "2型糖尿病的诊断标准是什么？空腹血糖和OGTT分别是多少？",
        "expected_response": "空腹静脉血浆葡萄糖≥7.0 mmol/L，OGTT 2h静脉血浆葡萄糖≥11.1 mmol/L，HbA1c≥6.5%。有典型症状满足任一即可诊断，无典型症状需不同时间重复确认。",
    },
    {
        "request": "高血压患者合并糖尿病，降压目标值应该是多少？",
        "expected_response": "一般糖尿病患者合并高血压降压目标为<130/80 mmHg。妊娠女性建议110~135/85 mmHg。老年患者可适当放宽。",
    },
    {
        "request": "高血压患者合并糖尿病推荐哪类降压药？",
        "expected_response": "首选ACEI或ARB，可联合钙通道阻滞剂CCB、利尿剂。SGLT2i和GLP-1RA具有心肾保护作用，也有一定降压效果。",
    },
    {
        "request": "糖尿病患者的LDL-C控制目标是什么？",
        "expected_response": "超高危<1.4 mmol/L且较基线降低≥50%；极高危<1.8 mmol/L且较基线降低≥50%；高危<2.6 mmol/L。非HDL-C目标为LDL-C目标+0.8 mmol/L。",
    },
    {
        "request": "什么是内分泌性高血压？常见病因有哪些？",
        "expected_response": "内分泌性高血压是由内分泌疾病引起的继发性高血压。常见病因包括原发性醉固酮增多症、嘲铬细胞瘤、库欣综合征、甲状旁腺功能亢进等。",
    },
    {
        "request": "基层血脂管理的质控指标有哪些？",
        "expected_response": "基层血脂管理质控指标包括血脂检测率、血脂达标率、他汀类药物使用率、随访率等。应定期评估和反馈。",
    },
    {
        "request": "HbA1c的控制目标是多少？不同患者群体有什么差异？",
        "expected_response": "一般T2DM患者HbA1c目标<7.0%。年轻、病程短、无严重并发症者可<6.5%。老年、病程长、有严重低血糖史者可放宽至<8.0%或<8.5%。",
    },
    {
        "request": "糖尿病患者的血压测量有什么特殊要求？",
        "expected_response": "糖尿病患者应常规测量血压，推荐家庭血压监测和24h动态血压监测，关注夜间血压和晨峰血压。",
    },
])

print(f"✅ 评估数据集已创建: {len(eval_data)} 个问题")
print(f"   覆盖: 糖尿病诊断/治疗、高血压管理、血脂分层、内分泌性高血压、基层质控")
display(eval_data)

# COMMAND ----------

# DBTITLE 1,Step 4: 重建 Agent
# 重建 Agent（与 04 notebook 和 agent_code.py 一致）
vs_tool = VectorSearchRetrieverTool(
    index_name=VS_INDEX,
    columns=["chunk_id", "guideline_name", "section", "content", "content_type", "char_count"],
    num_results=5,
    tool_name="search_medical_guidelines",
    tool_description=(
        "搜索临床诊疗指南知识库。"
        "包含糖尿病、高血压、血脂异常等三高相关的"
        "中国临床实践指南和专家共识。9份权威文献，"
        "覆盖诊断标准、治疗方案、药物选择、分级管理等。"
        "当用户提问与糖尿病、高血压、血脂相关的临床问题时使用此工具。"
    ),
)

llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

SYSTEM_PROMPT = """你是一个专业的临床诊疗助手，专注于“三高”（高血压、高血脂、糖尿病）的临床指南问答。
必须基于检索到的指南内容回答，不要使用检索结果之外的知识。
每次回答必须调用 search_medical_guidelines 工具检索相关内容。
回答用中文，并在末尾标注来源指南和章节。
临床建议仅供参考，不能替代医生诊断。
"""

# 使用 langgraph create_react_agent（与 agent_code.py 一致）
agent = create_react_agent(llm, tools=[vs_tool], prompt=SYSTEM_PROMPT)

print("\u2705 Agent 已重建（langgraph create_react_agent），准备评估")

# COMMAND ----------

# DBTITLE 1,Step 5: 定义评估函数
# 定义模型调用函数（databricks-agent 模式）
# mlflow.evaluate 会传入单条 dict: {"messages": [{"role": "user", "content": "..."}]}
def predict_fn(input):
    """
    评估用的模型调用函数。
    输入: dict with 'messages' key
    输出: dict with 'content' key (string response)
    """
    from langchain_core.messages import HumanMessage, AIMessage

    messages = input.get("messages", [])
    if not messages:
        return {"content": "无法生成回答: 没有输入消息"}

    try:
        # 转换为 langchain message 格式
        input_msgs = []
        for msg in messages:
            if msg["role"] == "user":
                input_msgs.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                input_msgs.append(AIMessage(content=msg["content"]))

        response = agent.invoke({"messages": input_msgs})
        # 提取最后一条 AI 消息
        answer = ""
        for m in reversed(response["messages"]):
            if hasattr(m, 'type') and m.type == 'ai' and m.content:
                answer = m.content
                break
        if not answer:
            answer = "无法生成回答"
    except Exception as e:
        answer = f"错误: {str(e)}"

    return {"content": answer}

print("\u2705 predict_fn 已定义（langgraph + databricks-agent 格式）")

# COMMAND ----------

# DBTITLE 1,Step 6: 运行评估
# 运行 MLflow Agent Evaluation
with mlflow.start_run(run_name="medical-rag-eval-v1"):
    eval_results = mlflow.evaluate(
        model=predict_fn,
        data=eval_data,
        model_type="databricks-agent",
    )

print("✅ 评估完成!")
print(f"\n=== 整体指标 ===")
for metric, value in sorted(eval_results.metrics.items()):
    if isinstance(value, float):
        print(f"   {metric}: {value:.4f}")
    else:
        print(f"   {metric}: {value}")

# COMMAND ----------

# DBTITLE 1,Step 7: 查看详细结果
# 查看整体指标
print("=== 整体评估指标 ===")
for metric, value in sorted(eval_results.metrics.items()):
    if isinstance(value, float):
        print(f"   {metric}: {value:.4f}")
    else:
        print(f"   {metric}: {value}")

# 查看详细评估表
print("\n\n=== 完整评估结果表 ===")
eval_table = eval_results.tables["eval_results"]

# 将混合类型列转为字符串，避免 Arrow 转换错误
for col in eval_table.columns:
    if eval_table[col].apply(type).nunique() > 1:
        eval_table[col] = eval_table[col].astype(str)

display(eval_table)

# COMMAND ----------

# DBTITLE 1,评估维度说明
# MAGIC %md
# MAGIC ## 评估维度说明
# MAGIC
# MAGIC | 指标 | 含义 | 期望 |
# MAGIC |------|------|------|
# MAGIC | **correctness** | 回答与参考答案的一致性 | ≥ 4/5 |
# MAGIC | **relevance** | 回答与问题的相关性 | ≥ 4/5 |
# MAGIC | **groundedness** | 回答是否基于检索内容（无幻觉） | ≥ 4/5 |
# MAGIC | **chunk_relevance** | 检索到的chunks与问题的相关性 | ≥ 3/5 |
# MAGIC
# MAGIC ### 下一步
# MAGIC - 如果评分低，分析具体哪个问题表现差，调整 prompt 或增加 num_results
# MAGIC - 确认通过后，进入下一步：注册到 UC 并部署为 Serving Endpoint