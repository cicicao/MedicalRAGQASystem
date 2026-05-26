# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # Vector Search 向量索引创建
# MAGIC
# MAGIC **项目**: 医疗RAG知识库 - 三高临床诊疗指南问答系统  
# MAGIC **功能**: 创建 Vector Search Endpoint + Delta Sync Index（自动Embedding）  
# MAGIC **源表**: `medical.medical_knowledge.chunks` (1095 chunks)  
# MAGIC **Embedding模型**: `databricks-qwen3-embedding-0-6b`（多语言/中文）  
# MAGIC **索引**: `medical.medical_knowledge.chunks_index`
# MAGIC
# MAGIC ### 步骤
# MAGIC 1. 源表准备（启用CDF + Primary Key）
# MAGIC 2. 创建 Vector Search Endpoint
# MAGIC 3. 创建 Delta Sync Index（Managed Embeddings）
# MAGIC 4. 等待同步完成
# MAGIC 5. 测试查询

# COMMAND ----------

# DBTITLE 1,Step 1: 源表准备 - 启用CDF和Primary Key
# MAGIC %sql
# MAGIC -- Step 1a: 启用 Change Data Feed（Delta Sync 必需）
# MAGIC ALTER TABLE medical.medical_knowledge.chunks 
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC -- Step 1b: 声明 Primary Key 约束（Vector Search 必需）
# MAGIC ALTER TABLE medical.medical_knowledge.chunks 
# MAGIC ADD CONSTRAINT chunks_pk PRIMARY KEY (chunk_id);

# COMMAND ----------

# DBTITLE 1,Step 2: 配置参数
from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()

# 配置
ENDPOINT_NAME = "medical-rag-endpoint"
INDEX_NAME = "medical.medical_knowledge.chunks_index"
SOURCE_TABLE = "medical.medical_knowledge.chunks"
EMBEDDING_MODEL = "databricks-qwen3-embedding-0-6b"

print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Index: {INDEX_NAME}")
print(f"Source: {SOURCE_TABLE}")
print(f"Embedding: {EMBEDDING_MODEL}")

# COMMAND ----------

# DBTITLE 1,Step 3: 创建 Vector Search Endpoint
# 创建 Standard Endpoint（低延迟，适合问答场景）
# 如果已存在则跳过
try:
    ep = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    print(f"✅ Endpoint已存在: {ENDPOINT_NAME}")
    print(f"   状态: {ep.status.state}")
except Exception:
    print(f"🚀 创建 Endpoint: {ENDPOINT_NAME} ...")
    w.vector_search_endpoints.create_endpoint(
        name=ENDPOINT_NAME,
        endpoint_type="STANDARD"
    )
    print("   创建请求已提交，等待就绪...")

# 等待 Endpoint 就绪
while True:
    ep = w.vector_search_endpoints.get_endpoint(ENDPOINT_NAME)
    state = ep.status.state.value if hasattr(ep.status.state, 'value') else str(ep.status.state)
    if state == "ONLINE":
        print(f"✅ Endpoint 就绪: {ENDPOINT_NAME}")
        break
    elif state in ("PROVISIONING_FAILED", "FAILED"):
        raise Exception(f"Endpoint创建失败: {ep.status.message}")
    else:
        print(f"   状态: {state}，等待30s...")
        time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Step 4: 创建 Delta Sync Index
# 创建 Delta Sync Index（Managed Embeddings）
# Databricks 自动对 content 列调用 qwen3 生成 embedding
try:
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    print(f"✅ Index已存在: {INDEX_NAME}")
    print(f"   状态: {idx.status.status}")
except Exception:
    print(f"🚀 创建 Index: {INDEX_NAME} ...")
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key="chunk_id",
        index_type="DELTA_SYNC",
        delta_sync_index_spec={
            "source_table": SOURCE_TABLE,
            "embedding_source_columns": [
                {
                    "name": "content",
                    "embedding_model_endpoint_name": EMBEDDING_MODEL
                }
            ],
            "pipeline_type": "TRIGGERED",
            "columns_to_sync": [
                "chunk_id", "guideline_name", "section", 
                "content", "content_type", "page_start", "page_end", "char_count"
            ]
        }
    )
    print("   Index创建请求已提交")
    print("   Databricks将自动对content列生成embedding并建立索引")

# COMMAND ----------

# DBTITLE 1,Step 5: 等待索引同步完成
# 等待 Index 同步完成
import time

print("⏳ 等待索引同步...")
print("   (首次同步需要为1095个chunks生成embedding，预计3-10分钟)")
print()

while True:
    idx = w.vector_search_indexes.get_index(INDEX_NAME)
    status = idx.status
    state = status.status.value if hasattr(status.status, 'value') else str(status.status)
    
    # 检查是否有详细信息
    msg = getattr(status, 'message', '') or ''
    ready = getattr(status, 'indexed_row_count', None)
    
    if state == "ONLINE":
        print(f"\n✅ Index 同步完成!")
        print(f"   状态: ONLINE")
        if ready:
            print(f"   已索引行数: {ready}")
        break
    elif "FAILED" in state.upper():
        print(f"\n❌ 同步失败: {msg}")
        break
    else:
        print(f"   [{time.strftime('%H:%M:%S')}] 状态: {state} {f'- {msg}' if msg else ''}")
        time.sleep(30)

# COMMAND ----------

# DBTITLE 1,Step 6: 测试向量搜索
# 测试向量搜索 - 模拟临床问诊场景
test_queries = [
    "2型糖尿病的诊断标准是什么？",
    "高血压患者的降压目标值",
    "血脂异常的药物治疗方案",
]

print("=" * 70)
print("向量搜索测试 - 临床问诊模拟")
print("=" * 70)

for query in test_queries:
    print(f"\n🔍 问题: {query}")
    print("-" * 50)
    
    results = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        columns=["chunk_id", "guideline_name", "section", "content", "content_type"],
        query_text=query,
        num_results=3
    )
    
    for i, row in enumerate(results.result.data_array, 1):
        chunk_id, guideline, section, content, ctype = row[:-1]
        score = row[-1]
        print(f"\n  Top {i} (相似度: {score:.4f})")
        print(f"  来源: {guideline}")
        print(f"  章节: {section}")
        print(f"  类型: {ctype}")
        print(f"  内容: {content[:150]}...")

print(f"\n{'=' * 70}")
print("✅ 向量搜索测试完成!")

# COMMAND ----------

# DBTITLE 1,完成说明
# MAGIC %md
# MAGIC ## 完成
# MAGIC
# MAGIC 向量索引已创建完成：
# MAGIC - **Endpoint**: `medical-rag-endpoint`（Standard，低延迟）
# MAGIC - **Index**: `medical.medical_knowledge.chunks_index`（Delta Sync，自动Embedding）
# MAGIC - **Embedding**: `databricks-qwen3-embedding-0-6b`（中文优化）
# MAGIC - **同步模式**: TRIGGERED（手动触发，如需自动同步可改为CONTINUOUS）
# MAGIC
# MAGIC ### 下一步
# MAGIC 1. 构建 RAG Agent Chain（LangGraph / OpenAI Agent）
# MAGIC 2. 添加 VectorSearchRetrieverTool
# MAGIC 3. 部署为 Model Serving Endpoint
# MAGIC 4. 连接 AI Playground 测试