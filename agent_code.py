import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatCompletionResponse, ChatMessage, ChatChoice


class MedicalRAGAgent(ChatModel):
    def load_context(self, context):
        from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
        from langgraph.prebuilt import create_react_agent

        LLM_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
        VS_INDEX = "medical.medical_knowledge.chunks_index"

        vs_tool = VectorSearchRetrieverTool(
            index_name=VS_INDEX,
            columns=["chunk_id", "guideline_name", "section", "content", "content_type", "char_count"],
            num_results=5,
            tool_name="search_medical_guidelines",
            tool_description="搜索三高临床诊疗指南知识库（糖尿病、高血压、血脂异常）。"
        )

        llm = ChatDatabricks(endpoint=LLM_ENDPOINT)

        SYSTEM_PROMPT = """你是一个专业的临床诊疗助手，专注于三高相关的临床指南问答。
必须基于检索到的指南内容回答，使用中文，并标注来源。
临床建议仅供参考，不能替代医生诊断。
"""
        self.agent = create_react_agent(llm, tools=[vs_tool], prompt=SYSTEM_PROMPT)

    def predict(self, context, messages, params):
        from langchain_core.messages import HumanMessage, AIMessage

        input_msgs = []
        for msg in messages:
            if msg.role == "user":
                input_msgs.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                input_msgs.append(AIMessage(content=msg.content))

        result = self.agent.invoke({"messages": input_msgs})

        last_ai_content = ""
        for m in reversed(result["messages"]):
            if hasattr(m, "type") and m.type == "ai" and m.content:
                last_ai_content = m.content
                break

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=last_ai_content),
                )
            ]
        )


mlflow.models.set_model(MedicalRAGAgent())
