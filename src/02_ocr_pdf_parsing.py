# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # 扫描版PDF解析 - ai_parse_document()
# MAGIC
# MAGIC **项目**: 医疗RAG知识库  
# MAGIC **功能**: 使用Databricks AI函数解析3份无法直接提取文本的PDF  
# MAGIC **输入**: `/Volumes/medical/medical_knowledge/raw_pdfs/`  
# MAGIC **输出**: 追加到 `medical.medical_knowledge.chunks`
# MAGIC
# MAGIC ### 待处理PDF
# MAGIC | # | 文档 | 页数 | 问题 |
# MAGIC |---|------|------|------|
# MAGIC | 1 | 《糖尿病分型诊断中国专家共识》临床实践应用 | 8页 | 扫描版，无法复制文字 |
# MAGIC | 2 | 中国高血压临床实践指南（2024版） | 48页 | 扫描版，16.4MB |
# MAGIC | 3 | 内分泌性高血压筛查专家共识（2025版） | 14页 | OCR质量差，乱码多 |
# MAGIC
# MAGIC ### 技术方案
# MAGIC 使用 `ai_parse_document()` v2.0 提取文档结构（段落、表格、标题、图片描述），然后进行分块处理，最终追加到现有chunks表。

# COMMAND ----------

# DBTITLE 1,Step 1 说明
# MAGIC %md
# MAGIC ## Step 1: 解析PDF为结构化VARIANT
# MAGIC
# MAGIC 使用 `ai_parse_document()` 函数读取二进制PDF文件，返回包含页面、元素（文本、表格、标题、图片）的结构化数据。

# COMMAND ----------

# DBTITLE 1,解析扫描版PDF
# MAGIC %sql
# MAGIC -- 解析3份扫描版PDF（含图片描述生成）
# MAGIC CREATE OR REPLACE TABLE medical.medical_knowledge.parsed_docs AS
# MAGIC SELECT
# MAGIC   path,
# MAGIC   regexp_extract(path, '([^/]+)\\.pdf$', 1) AS filename,
# MAGIC   ai_parse_document(
# MAGIC     content,
# MAGIC     MAP('version', '2.0', 'descriptionElementTypes', '*')
# MAGIC   ) AS parsed
# MAGIC FROM READ_FILES(
# MAGIC   '/Volumes/medical/medical_knowledge/raw_pdfs/',
# MAGIC   format => 'binaryFile'
# MAGIC )
# MAGIC WHERE path LIKE '%糖尿病分型诊断%'
# MAGIC    OR path LIKE '%中国高血压临床实践指南%'
# MAGIC    OR path LIKE '%内分泌性高血压筛查%';

# COMMAND ----------

# DBTITLE 1,Step 2 说明
# MAGIC %md
# MAGIC ## Step 2: 检查解析结果
# MAGIC
# MAGIC 验证解析是否成功，查看元素类型分布。

# COMMAND ----------

# DBTITLE 1,检查解析状态
# MAGIC %sql
# MAGIC -- 检查解析状态
# MAGIC SELECT
# MAGIC   filename,
# MAGIC   try_cast(parsed:error_status AS STRING) AS error_status,
# MAGIC   try_cast(parsed:metadata:page_count AS INT) AS page_count,
# MAGIC   size(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS element_count
# MAGIC FROM medical.medical_knowledge.parsed_docs;

# COMMAND ----------

# DBTITLE 1,元素类型分布
# MAGIC %sql
# MAGIC -- 查看元素类型分布
# MAGIC SELECT
# MAGIC   filename,
# MAGIC   try_cast(element:type AS STRING) AS element_type,
# MAGIC   COUNT(*) AS cnt
# MAGIC FROM medical.medical_knowledge.parsed_docs
# MAGIC LATERAL VIEW EXPLODE(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS element
# MAGIC GROUP BY filename, try_cast(element:type AS STRING)
# MAGIC ORDER BY filename, cnt DESC;

# COMMAND ----------

# DBTITLE 1,Step 3 说明
# MAGIC %md
# MAGIC ## Step 3: 展开元素为行
# MAGIC
# MAGIC 将每个文档的elements数组展开为独立行，保留元素类型、页码、内容。

# COMMAND ----------

# DBTITLE 1,展开元素为行
# MAGIC %sql
# MAGIC -- 展开元素为行
# MAGIC CREATE OR REPLACE TEMP VIEW parsed_elements AS
# MAGIC SELECT
# MAGIC   filename,
# MAGIC   posexplode(try_cast(parsed:document:elements AS ARRAY<VARIANT>)) AS (element_idx, element),
# MAGIC   try_cast(element:type AS STRING) AS element_type,
# MAGIC   try_cast(element:content AS STRING) AS element_content,
# MAGIC   try_cast(element:page AS INT) AS page_num
# MAGIC FROM medical.medical_knowledge.parsed_docs
# MAGIC WHERE try_cast(parsed:error_status AS STRING) IS NULL;
# MAGIC
# MAGIC SELECT element_type, COUNT(*) as cnt, ROUND(AVG(LENGTH(element_content))) as avg_len
# MAGIC FROM parsed_elements
# MAGIC GROUP BY element_type
# MAGIC ORDER BY cnt DESC;

# COMMAND ----------

# DBTITLE 1,Step 4 说明
# MAGIC %md
# MAGIC ## Step 4: 分块逻辑
# MAGIC
# MAGIC 将解析出的元素按以下规则组合为chunks：
# MAGIC - **title/section_header**: 作为当前section标记
# MAGIC - **text**: 主要内容，按800字限制分块
# MAGIC - **table**: 单独成chunk（完整保留HTML内容转为文本）
# MAGIC - **figure**: 保留AI描述作为占位
# MAGIC - **page_header/page_footer/page_number/footnote**: 跳过

# COMMAND ----------

# DBTITLE 1,分块函数定义
import re
from typing import List, Dict

def chunk_parsed_elements(elements: list, guideline_name: str, max_size: int = 800) -> List[Dict]:
    """
    将ai_parse_document解析出的elements列表转为chunks。
    
    Args:
        elements: list of dicts with keys: element_type, element_content, page_num, element_idx
        guideline_name: 指南名称
        max_size: 单chunk最大字符数
    
    Returns:
        List of chunk dicts compatible with catalog1.medical_knowledge.chunks schema
    """
    chunks = []
    chunk_id = 0
    current_section = "摘要/前言"
    current_buf = []
    current_buf_size = 0
    current_start_page = 1
    
    # 跳过的元素类型
    skip_types = {'page_header', 'page_footer', 'page_number', 'footnote'}
    
    for elem in elements:
        etype = elem['element_type']
        content = elem['element_content'] or ''
        page = elem['page_num'] or 1
        
        if etype in skip_types:
            continue
        
        # 标题/章节标题 → 更新当前section
        if etype in ('title', 'section_header'):
            # 先保存之前的buffer
            if current_buf:
                chunk_id += 1
                chunk_text = '\n'.join(current_buf)
                chunks.append({
                    "chunk_id": chunk_id,
                    "guideline_name": guideline_name,
                    "section": current_section,
                    "content": chunk_text,
                    "content_type": "paragraph",
                    "page_start": current_start_page,
                    "page_end": page,
                    "char_count": len(chunk_text),
                })
                current_buf = []
                current_buf_size = 0
            current_section = content.strip() if content.strip() else current_section
            current_start_page = page
            continue
        
        # 表格 → 单独成chunk
        if etype == 'table':
            # 先保存buffer
            if current_buf:
                chunk_id += 1
                chunk_text = '\n'.join(current_buf)
                chunks.append({
                    "chunk_id": chunk_id,
                    "guideline_name": guideline_name,
                    "section": current_section,
                    "content": chunk_text,
                    "content_type": "paragraph",
                    "page_start": current_start_page,
                    "page_end": page,
                    "char_count": len(chunk_text),
                })
                current_buf = []
                current_buf_size = 0
            
            # 清除HTML标签，保留表格文本
            table_text = re.sub(r'<[^>]+>', ' ', content)
            table_text = re.sub(r'\s+', ' ', table_text).strip()
            if len(table_text) > 30:
                chunk_id += 1
                chunks.append({
                    "chunk_id": chunk_id,
                    "guideline_name": guideline_name,
                    "section": current_section,
                    "content": table_text,
                    "content_type": "table",
                    "page_start": page,
                    "page_end": page,
                    "char_count": len(table_text),
                })
            current_start_page = page
            continue
        
        # 图片 → 占位标记
        if etype == 'figure':
            desc = content.strip() if content else '无描述'
            current_buf.append(f"[图片: {desc}]")
            current_buf_size += len(desc) + 10
            continue
        
        # caption → 附加到buffer
        if etype == 'caption':
            current_buf.append(content.strip())
            current_buf_size += len(content)
            continue
        
        # text → 主要内容
        if etype == 'text':
            text = content.strip()
            if not text:
                continue
            
            # 检查是否超过max_size
            if current_buf_size + len(text) > max_size and current_buf:
                chunk_id += 1
                chunk_text = '\n'.join(current_buf)
                chunks.append({
                    "chunk_id": chunk_id,
                    "guideline_name": guideline_name,
                    "section": current_section,
                    "content": chunk_text,
                    "content_type": "paragraph",
                    "page_start": current_start_page,
                    "page_end": page,
                    "char_count": len(chunk_text),
                })
                current_buf = [text]
                current_buf_size = len(text)
                current_start_page = page
            else:
                current_buf.append(text)
                current_buf_size += len(text)
    
    # 保存最后的buffer
    if current_buf:
        chunk_id += 1
        chunk_text = '\n'.join(current_buf)
        if len(chunk_text.strip()) > 30:
            chunks.append({
                "chunk_id": chunk_id,
                "guideline_name": guideline_name,
                "section": current_section,
                "content": chunk_text,
                "content_type": "paragraph",
                "page_start": current_start_page,
                "page_end": len(elements),
                "char_count": len(chunk_text),
            })
    
    return chunks

print("✅ chunk_parsed_elements() 已定义")

# COMMAND ----------

# DBTITLE 1,Step 5 说明
# MAGIC %md
# MAGIC ## Step 5: 从parsed_docs提取元素并分块

# COMMAND ----------

# DBTITLE 1,提取元素并分块
# 从parsed_docs表读取解析结果，转为chunks
from pyspark.sql import SparkSession
from collections import defaultdict

spark = SparkSession.builder.getOrCreate()

# 读取解析结果
parsed_df = spark.sql("""
    SELECT
        filename,
        element_idx,
        element_type,
        element_content,
        page_num
    FROM parsed_elements
    ORDER BY filename, element_idx
""").collect()

# 按文件分组
files_elements = defaultdict(list)
for row in parsed_df:
    files_elements[row['filename']].append({
        'element_type': row['element_type'],
        'element_content': row['element_content'],
        'page_num': row['page_num'],
        'element_idx': row['element_idx'],
    })

# 文件名到指南名称的映射
filename_to_guideline = {
    "《糖尿病分型诊断中国专家共识》在糖尿病分型诊断中的临床实践应用": "糖尿病分型诊断中国专家共识临床实践应用",
    "中国高血压临床实践指南（2024版）": "中国高血压临床实践指南（2024版）",
    "内分泌性高血压筛查专家共识2025版-": "内分泌性高血压筛查专家共识（2025版）",
}

# 分块处理
all_ocr_chunks = []
for fname, elements in files_elements.items():
    guideline_name = filename_to_guideline.get(fname, fname)
    chunks = chunk_parsed_elements(elements, guideline_name)
    all_ocr_chunks.extend(chunks)
    
    table_count = sum(1 for c in chunks if c['content_type'] == 'table')
    print(f"📄 {guideline_name}")
    print(f"   chunks: {len(chunks)} | 表格: {table_count} | 平均字符: {sum(c['char_count'] for c in chunks)//max(len(chunks),1)}")

print(f"\n✅ OCR PDF总计: {len(all_ocr_chunks)} chunks")

# COMMAND ----------

# DBTITLE 1,Step 6 说明
# MAGIC %md
# MAGIC ## Step 6: 追加到chunks表
# MAGIC
# MAGIC 将OCR解析的chunks追加到现有的 `catalog1.medical_knowledge.chunks` 表中。

# COMMAND ----------

# DBTITLE 1,追加到chunks表
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 获取现有最大chunk_id
max_id = spark.sql("SELECT MAX(chunk_id) as max_id FROM medical.medical_knowledge.chunks").collect()[0]['max_id']
print(f"现有最大chunk_id: {max_id}")

# 重新编号（接续现有ID）
for i, c in enumerate(all_ocr_chunks):
    c['chunk_id'] = max_id + i + 1

# 写入（追加模式）
schema = StructType([
    StructField("chunk_id", IntegerType(), False),
    StructField("guideline_name", StringType(), False),
    StructField("section", StringType(), False),
    StructField("content", StringType(), False),
    StructField("content_type", StringType(), False),
    StructField("page_start", IntegerType(), False),
    StructField("page_end", IntegerType(), False),
    StructField("char_count", IntegerType(), False),
])

df = spark.createDataFrame(all_ocr_chunks, schema=schema)
df.write.format("delta").mode("append").saveAsTable("medical.medical_knowledge.chunks")

new_total = spark.sql("SELECT COUNT(*) as cnt FROM medical.medical_knowledge.chunks").collect()[0]['cnt']
print(f"\n✅ 追加成功!")
print(f"   新增: {len(all_ocr_chunks)} chunks")
print(f"   总计: {new_total} chunks")

# COMMAND ----------

# DBTITLE 1,Step 7 说明
# MAGIC %md
# MAGIC ## Step 7: 最终验证

# COMMAND ----------

# DBTITLE 1,全部指南统计
# MAGIC %sql
# MAGIC -- 验证所有指南的chunks统计
# MAGIC SELECT 
# MAGIC     guideline_name,
# MAGIC     content_type,
# MAGIC     COUNT(*) as chunk_count,
# MAGIC     ROUND(AVG(char_count)) as avg_chars
# MAGIC FROM medical.medical_knowledge.chunks
# MAGIC GROUP BY guideline_name, content_type
# MAGIC ORDER BY guideline_name, content_type;

# COMMAND ----------

# DBTITLE 1,OCR chunks质量检查
# MAGIC %sql
# MAGIC -- 查看OCR解析的chunks质量
# MAGIC SELECT 
# MAGIC     chunk_id, guideline_name, section, content_type, char_count,
# MAGIC     LEFT(content, 200) as preview
# MAGIC FROM medical.medical_knowledge.chunks
# MAGIC WHERE guideline_name IN (
# MAGIC     '糖尿病分型诊断中国专家共识临床实践应用',
# MAGIC     '中国高血压临床实践指南（2024版）',
# MAGIC     '内分泌性高血压筛查专家共识（2025版）'
# MAGIC )
# MAGIC ORDER BY guideline_name, chunk_id
# MAGIC LIMIT 15;

# COMMAND ----------

# DBTITLE 1,完成说明
# MAGIC %md
# MAGIC ## 完成
# MAGIC
# MAGIC 所有9份PDF均已处理完成：
# MAGIC - **6份文本PDF**: 01_pdf_parsing_and_chunking(PyMuPDF直接解析)
# MAGIC - **3份扫描/OCR PDF**: 02_ocr_pdf_parsing (ai_parse_document)
# MAGIC
# MAGIC 数据已统一存储在 `medical.medical_knowledge.chunks` 表中，可用于后续的：
# MAGIC 1. Embedding生成
# MAGIC 2. Vector Search Index创建
# MAGIC 3. RAG Chain构建