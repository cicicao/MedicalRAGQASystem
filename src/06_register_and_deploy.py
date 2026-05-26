# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # Register and Deploy Medical RAG Agent
# MAGIC
# MAGIC **项目**: 三高临床诊疗指南问答系统  
# MAGIC **目标**: 将 RAG Agent 注册到 Unity Catalog，部署为 Model Serving Endpoint  
# MAGIC **UC 模型**: `medical.medical_knowledge.rag_agent`  
# MAGIC **Serving Endpoint**: `medical-rag-agent-endpoint`
# MAGIC
# MAGIC ### 步骤
# MAGIC 1. 安装依赖
# MAGIC 2. 配置参数
# MAGIC 3. 检查 UC 模型版本状态
# MAGIC 4. 设置别名
# MAGIC 5. 创建 Serving Endpoint
# MAGIC 6. 验证部署状态

# COMMAND ----------

# DBTITLE 1,Step 1: 安装依赖
# MAGIC %pip install "mlflow[databricks]" databricks-agents databricks-langchain langchain langgraph langchain-core --upgrade -q
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Step 2: 配置参数
import mlflow
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

REGISTERED_MODEL_NAME = "medical.medical_knowledge.rag_agent"
MODEL_ALIAS = "Champion"
ENDPOINT_NAME = "medical-rag-agent-endpoint"

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
w = WorkspaceClient()

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Alias: {MODEL_ALIAS}")
print(f"Serving endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# DBTITLE 1,Step 3: 注册到 Unity Catalog
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

EXPERIMENT_NAME = "/Users/cici@caoxx1018gmail.onmicrosoft.com/medical/medical-rag-agent"
mlflow.set_experiment(EXPERIMENT_NAME)

# 删除旧版本
print("🗑️ 清理旧模型版本...")
try:
    old_versions = list(client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'"))
    for v in old_versions:
        client.delete_model_version(name=REGISTERED_MODEL_NAME, version=v.version)
    client.delete_registered_model(name=REGISTERED_MODEL_NAME)
    print(f"   已删除: {REGISTERED_MODEL_NAME}")
except Exception as e:
    print(f"   跳过: {e}")

# agent_code.py 已修复为 ChatModel + ChatCompletionResponse 实现
agent_code_path = "/Workspace/Users/cici@caoxx1018gmail.onmicrosoft.com/medical/agent_code.py"
print(f"✅ Agent 代码: {agent_code_path}")

# 声明依赖资源
resources = [
    DatabricksVectorSearchIndex(index_name="medical.medical_knowledge.chunks_index"),
    DatabricksServingEndpoint(endpoint_name="databricks-qwen3-next-80b-a3b-instruct"),
]

input_example = {
    "messages": [{"role": "user", "content": "2型糖尿病的诊断标准是什么？"}]
}

print("🚀 记录模型 (ChatModel / pyfunc)...")
with mlflow.start_run(run_name="medical-rag-agent-deploy-v7"):
    model_info = mlflow.pyfunc.log_model(
        python_model=agent_code_path,
        artifact_path="model",
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
    print(f"   Model URI: {model_info.model_uri}")
    print(f"   Signature: {model_info.signature}")

# 注册到 UC
print(f"🚀 注册到 UC: {REGISTERED_MODEL_NAME}...")
registered = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=REGISTERED_MODEL_NAME,
)
version = int(registered.version)
print(f"✅ 注册成功: {REGISTERED_MODEL_NAME} v{version}")

# 等待 READY
for _ in range(60):
    mv = client.get_model_version(name=REGISTERED_MODEL_NAME, version=str(version))
    if str(mv.status) == 'READY':
        print(f"✅ 模型状态: READY")
        break
    time.sleep(5)
else:
    print(f"⚠️ 超时，当前状态: {mv.status}")

# 设置别名
client.set_registered_model_alias(name=REGISTERED_MODEL_NAME, alias=MODEL_ALIAS, version=version)
print(f"✅ 别名: {MODEL_ALIAS} -> v{version}")

# COMMAND ----------

# DBTITLE 1,Step 4: 设置模型别名
# 上一步已完成别名设置，跳过
print(f"✅ 别名已在 Step 3 中设置: {MODEL_ALIAS} -> v{version}")

# COMMAND ----------

# DBTITLE 1,Step 5: 创建或更新 Serving Endpoint
from databricks import agents

# 先清理残留的失败 endpoint
try:
    w.serving_endpoints.delete(ENDPOINT_NAME)
    print(f"🗑️ 已删除残留 endpoint: {ENDPOINT_NAME}")
    import time; time.sleep(10)
except Exception:
    pass

# 使用 databricks.agents.deploy 部署
# 它会自动处理依赖、容器配置、权限和服务端点创建
print(f"🚀 使用 databricks.agents.deploy() 部署...")
print(f"   模型: {REGISTERED_MODEL_NAME} v{version}")
print(f"   Endpoint: {ENDPOINT_NAME}")

deployment = agents.deploy(
    model_name=REGISTERED_MODEL_NAME,
    model_version=version,
    endpoint_name=ENDPOINT_NAME,
    scale_to_zero=True,
)

print(f"\n✅ 部署成功!")
print(f"   Endpoint: {deployment.endpoint_name}")
print(f"   Query endpoint: {deployment.query_endpoint}")

# COMMAND ----------

# DBTITLE 1,Step 6: 验证部署状态
# 检查 Endpoint 状态
endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
print("=== Endpoint 状态 ===")
print(f"名称: {endpoint.name}")
print(f"状态: {endpoint.state.ready if endpoint.state else 'UNKNOWN'}")
print(f"配置更新时间: {endpoint.config_update if hasattr(endpoint, 'config_update') else 'N/A'}")

if endpoint.state and hasattr(endpoint.state, 'config_update'):
    print(f"配置状态: {endpoint.state.config_update}")

print("\n=== Served Entity ===")
if endpoint.config and endpoint.config.served_entities:
    for e in endpoint.config.served_entities:
        print(f"实体名: {e.name}")
        print(f"模型: {getattr(e, 'entity_name', 'N/A')} v{getattr(e, 'entity_version', 'N/A')}")
        print(f"Scale to zero: {getattr(e, 'scale_to_zero_enabled', 'N/A')}")
        print(f"Workload: {getattr(e, 'workload_size', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,完成说明
# MAGIC %md
# MAGIC ## 完成
# MAGIC
# MAGIC 注册部署完成：
# MAGIC
# MAGIC * Unity Catalog 模型: `medical.medical_knowledge.rag_agent` v1 (READY)
# MAGIC * 模型别名: `Champion`
# MAGIC * Serving Endpoint: `medical-rag-agent-endpoint`
# MAGIC
# MAGIC ### 下一步
# MAGIC * 进入 Review App，收集人工反馈
# MAGIC * 用 SDK / REST API 调用 Serving Endpoint
# MAGIC * 继续优化 prompt、检索参数和 reranker