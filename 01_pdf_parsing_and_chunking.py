# Databricks notebook source
# DBTITLE 1,项目说明
# MAGIC %md
# MAGIC # 医疗指南PDF解析与分块Pipeline
# MAGIC
# MAGIC **项目**: 医疗RAG知识库 - 临床诊疗指南问答系统  
# MAGIC **功能**: 解析6份文本PDF → 智能分块 → 写入Delta Table  
# MAGIC **输入**: `/Volumes/medical/medical_knowledge/raw_pdfs/`  
# MAGIC **输出**: `medical.medical_knowledge.chunks`
# MAGIC
# MAGIC ### 处理策略
# MAGIC - 标题层级识别（第X章 > 一、二、 > （一）（二） > 1.1）
# MAGIC - 表格完整保留（含续表合并）
# MAGIC - 图片占位标记
# MAGIC - 参考文献排除
# MAGIC - 分块大小控制（目标300-800字）
# MAGIC
# MAGIC ### PDF分类
# MAGIC | 类型 | 数量 | 处理方式 |
# MAGIC |------|------|----------|
# MAGIC | 文本可提取 | 6份 | PyMuPDF直接解析（本notebook） |
# MAGIC | 扫描/OCR质量差 | 3份 | 后续用ai_parse_document()处理 |

# COMMAND ----------

# DBTITLE 1,安装依赖
# MAGIC %pip install pymupdf -q
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,导入和配置
import fitz
import os
import re
import unicodedata
from typing import List, Dict, Tuple

# 配置
volume_path = "/Volumes/medical/medical_knowledge/raw_pdfs"

# 6份文本可直接解析的PDF（排除3份需要OCR的）
clean_pdfs = {
    "中国糖尿病防治指南（2024版）.pdf": "中国糖尿病防治指南（2024版）",
    "糖尿病患者血脂管理中国专家共识（2024-版）.pdf": "糖尿病患者血脂管理中国专家共识（2024版）",
    "2-型糖尿病患者泛血管疾病风险评估与管理中国专家共识（2022-版）.pdf": "2型糖尿病患者泛血管疾病风险评估与管理中国专家共识（2022版）",
    "国家基层高血压防治管理指南（2025版）.pdf": "国家基层高血压防治管理指南（2025版）",
    "中国血脂管理指南（基层版-2024-年）.pdf": "中国血脂管理指南（基层版2024年）",
    "基层血脂管理适宜技术与质量控制中国专家建议（2025-年）.pdf": "基层血脂管理适宜技术与质量控制中国专家建议（2025年）",
}

print(f"Volume路径: {volume_path}")
print(f"待处理PDF: {len(clean_pdfs)}份")
print(f"\n文件列表:")
for fn, gn in clean_pdfs.items():
    print(f"  • {gn}")

# COMMAND ----------

# DBTITLE 1,解析函数说明
# MAGIC %md
# MAGIC ## 核心解析函数
# MAGIC
# MAGIC 以下函数实现PDF结构识别：
# MAGIC - **标题层级检测**: 4级层次（章 > 节 > 小节 > 条目）
# MAGIC - **表格识别**: 表N标题 + 续表合并 + 智能退出
# MAGIC - **噪声过滤**: 页眉页脚、目录、参考文献
# MAGIC - **图片标记**: 保留图片标题作为占位符

# COMMAND ----------

# DBTITLE 1,解析函数定义
# ============================================================
# 标题层级检测
# ============================================================
def detect_heading_level(line: str) -> Tuple[int, str]:
    """
    识别标题层级。
    Level 1: 第X章, 附录N
    Level 2: 一、二、三、 或 "N  标题"(双空格)
    Level 3: （一）（二）或 N.N 标题
    Level 4: N.N.N 标题
    返回 (level, title) 或 (0, "") 表示非标题。
    """
    line = line.strip()
    if len(line) > 60 or len(line) < 3:
        return (0, "")
    # 排除含医学单位的行（这些是数据不是标题）
    if re.search(r'(mmol/L|mg/d|kg/m|mL/min|mm\s*Hg|片\s*qd|片\s*bid|片\s*tid)', line):
        return (0, "")

    # Level 1: 第X章
    if re.match(r'^第[一二三四五六七八九十百]+章\s', line):
        return (1, line)
    # Level 1: 附录N
    if re.match(r'^附录\s*\d+', line):
        return (1, line)

    # Level 2: 一、二、三、
    if re.match(r'^[一二三四五六七八九十]+、', line) and len(line) < 50:
        return (2, line)
    # Level 2: "N  标题" (数字后双空格，无小数点)
    m = re.match(r'^(\d{1,2})\s{2,}(\S.{2,})', line)
    if m and not re.search(r'\d\.\d', line):
        return (2, line)

    # Level 3: （一）（二）
    if re.match(r'^[（(][一二三四五六七八九十]+[）)]', line) and len(line) < 50:
        return (3, line)
    # Level 3: N.N 标题
    m = re.match(r'^(\d{1,2}\.\d{1,2})\s{1,}(\S.{2,})', line)
    if m:
        return (3, line)

    # Level 4: N.N.N 标题
    if re.match(r'^(\d{1,2}\.\d{1,2}\.\d{1,2})\s', line):
        return (4, line)

    return (0, "")


# ============================================================
# 噪声检测
# ============================================================
def is_noise(line: str) -> bool:
    """检测页眉、页脚、期刊名等噪声行。"""
    ls = line.strip()
    if re.match(r'^[·•]\s*\d+\s*[·•]$', ls):
        return True
    if '中华糖尿病杂志' in ls and ('年' in ls or 'Vol' in ls):
        return True
    if 'Chin J Diabetes' in ls:
        return True
    if re.match(r'^中国循环杂志\s*\d{4}', ls):
        return True
    if 'Chinese Circulation Journal' in ls:
        return True
    if re.match(r'^\d{4}年\d+月\s+第\d+卷', ls):
        return True
    if re.match(r'^https?://', ls):
        return True
    return False


# ============================================================
# 目录行检测
# ============================================================
def is_toc(line: str) -> bool:
    """检测目录行（含连续省略号的行）。"""
    return bool(re.search(r'…{3,}|\.{6,}', line))


# ============================================================
# 参考文献检测
# ============================================================
def is_ref_start(line: str) -> bool:
    """检测参考文献章节的开始。"""
    return bool(re.match(r'^参\s*考\s*文\s*献\s*$', line.strip()))


# ============================================================
# 表格标题检测
# ============================================================
def is_table_title(line: str) -> bool:
    """
    检测表格标题行。
    匹配: "表N 标题内容"(>8字符) 或 "表N"单独一行
    不匹配: 正文引用如"见表5）"、"表1），该类患者"
    """
    ls = line.strip()
    # "表N 标题" 格式（表号后有空格和标题文字）
    if re.match(r'^表\s*\d+[\s\u3000]+\S', ls) and len(ls) > 8:
        return True
    # "表N" 单独一行
    if re.match(r'^表\s*\d+\s*$', ls):
        return True
    return False


# ============================================================
# 续表检测
# ============================================================
def is_cont_table(line: str) -> bool:
    """检测续表标记：续表N, （续表N）"""
    return bool(re.match(r'^[（(]?续表\s*\d+[）)]?\s*$', line.strip()))


# ============================================================
# 图片标题检测
# ============================================================
def is_fig_title(line: str) -> bool:
    """检测图片标题：图N 标题"""
    ls = line.strip()
    if re.match(r'^图\s*\d+[\s\u3000]+\S', ls) and len(ls) > 5:
        return True
    return False


# ============================================================
# 文本清洗
# ============================================================
def clean(text: str) -> str:
    """清除私有Unicode字符（装饰符号），整理空行。"""
    cleaned = ''.join(
        c for c in text
        if not (0xF000 <= ord(c) <= 0xFFFF or ord(c) > 0x10000)
    )
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


print("✅ 解析函数已定义")

# COMMAND ----------

# DBTITLE 1,主解析函数
def parse_pdf(filepath: str, guideline_name: str, max_size: int = 800) -> List[Dict]:
    """
    解析单个PDF文件为结构化chunks。
    
    策略:
    1. 跳过目录页和参考文献
    2. 按标题层级切分section
    3. 表格（含续表）合并为单独chunk
    4. 图片标题保留为占位符
    5. 超大section按段落边界拆分
    
    Args:
        filepath: PDF文件路径
        guideline_name: 指南显示名称
        max_size: 单chunk最大字符数
    
    Returns:
        List of chunk dicts with keys:
        chunk_id, guideline_name, section, content, content_type, page_start, page_end, char_count
    """
    doc = fitz.open(filepath)
    
    # Pass 1: 识别目录页（前5页中含>=5个目录行的页面）
    toc_pages = set()
    for p in range(min(5, len(doc))):
        if sum(1 for l in doc[p].get_text().split('\n') if is_toc(l)) >= 5:
            toc_pages.add(p)
    
    # Pass 2: 逐页逐行提取，识别结构
    sections = []  # [{"t": type, "c": content, "p": (start, end), "h": titles}]
    titles = {1: "", 2: "", 3: "", 4: ""}  # 当前标题层级
    buf = []           # 当前section的内容行
    start_p = 1        # 当前section起始页
    ctype = "paragraph"  # 当前section类型
    in_refs = False    # 是否进入参考文献区域
    tbl_lines = 0      # 表格模式中的行计数
    
    for pn in range(len(doc)):
        if pn in toc_pages:
            continue
        
        for line in doc[pn].get_text().split('\n'):
            ls = line.strip()
            if not ls or is_noise(ls) or is_toc(ls):
                continue
            
            # 参考文献检测 → 之后的内容全部跳过
            if is_ref_start(ls):
                in_refs = True
                if buf:
                    sections.append({"t": ctype, "c": '\n'.join(buf),
                                    "p": (start_p, pn + 1), "h": dict(titles)})
                    buf = []
                continue
            if in_refs:
                continue
            
            # 图片标题 → 添加占位符
            if is_fig_title(ls):
                buf.append(f"[图片: {ls}]")
                continue
            
            # 续表 → 追加到当前表格
            if is_cont_table(ls):
                if ctype == "table":
                    buf.append(f"--- {ls} ---")
                continue
            
            # 表格标题 → 开始新的table section
            if is_table_title(ls):
                if buf:
                    sections.append({"t": ctype, "c": '\n'.join(buf),
                                    "p": (start_p, pn + 1), "h": dict(titles)})
                buf, start_p, ctype, tbl_lines = [ls], pn + 1, "table", 0
                continue
            
            # 标题检测 → 更新层级，开始新paragraph section
            lv, tt = detect_heading_level(ls)
            if lv > 0:
                if buf:
                    sections.append({"t": ctype, "c": '\n'.join(buf),
                                    "p": (start_p, pn + 1), "h": dict(titles)})
                titles[lv] = tt
                for l in range(lv + 1, 5):
                    titles[l] = ""
                buf, start_p, ctype = [ls], pn + 1, "paragraph"
                continue
            
            # 表格退出逻辑：遇到长段落行时结束表格
            if ctype == "table":
                tbl_lines += 1
                chinese_chars = sum(1 for c in ls if '\u4e00' <= c <= '\u9fff')
                if len(ls) > 40 and chinese_chars > 20 and tbl_lines > 3:
                    # 长中文段落出现，说明表格已结束
                    sections.append({"t": "table", "c": '\n'.join(buf),
                                    "p": (start_p, pn + 1), "h": dict(titles)})
                    buf, start_p, ctype = [ls], pn + 1, "paragraph"
                else:
                    buf.append(ls)
            else:
                buf.append(ls)
    
    # 保存最后一段
    if buf and not in_refs:
        sections.append({"t": ctype, "c": '\n'.join(buf),
                        "p": (start_p, len(doc)), "h": dict(titles)})
    doc.close()
    
    # Pass 3: 构建chunks（大小控制）
    chunks = []
    cid = 0
    
    for s in sections:
        content = clean(s["c"])
        if len(content) < 30:
            continue
        
        # 构建section路径（面包屑）
        sp = " > ".join([t for t in [s["h"].get(i, "") for i in range(1, 5)] if t])
        if not sp:
            sp = "摘要/前言"
        
        if len(content) <= max_size:
            cid += 1
            chunks.append({
                "chunk_id": cid,
                "guideline_name": guideline_name,
                "section": sp,
                "content": content,
                "content_type": s["t"],
                "page_start": s["p"][0],
                "page_end": s["p"][1],
                "char_count": len(content),
            })
        else:
            # 按段落边界拆分（换行后紧跟非空格字符处）
            parts = re.split(r'\n(?=[^\s])', content)
            b, bs = [], 0
            for part in parts:
                if bs + len(part) > max_size and b:
                    cid += 1
                    ct = '\n'.join(b)
                    chunks.append({
                        "chunk_id": cid,
                        "guideline_name": guideline_name,
                        "section": sp,
                        "content": ct,
                        "content_type": s["t"],
                        "page_start": s["p"][0],
                        "page_end": s["p"][1],
                        "char_count": len(ct),
                    })
                    b, bs = [part], len(part)
                else:
                    b.append(part)
                    bs += len(part)
            if b:
                ct = '\n'.join(b)
                if len(ct.strip()) > 30:
                    cid += 1
                    chunks.append({
                        "chunk_id": cid,
                        "guideline_name": guideline_name,
                        "section": sp,
                        "content": ct,
                        "content_type": s["t"],
                        "page_start": s["p"][0],
                        "page_end": s["p"][1],
                        "char_count": len(ct),
                    })
    
    return chunks


print("✅ parse_pdf() 主解析函数已定义")

# COMMAND ----------

# DBTITLE 1,执行解析
# MAGIC %md
# MAGIC ## 执行解析
# MAGIC
# MAGIC 处理6份文本PDF，生成结构化chunks。

# COMMAND ----------

# DBTITLE 1,执行解析所有PDF
# 解析所有6份文本PDF
all_chunks = []

print("=" * 70)
print("PDF解析分块结果")
print("=" * 70)

for filename, guideline_name in clean_pdfs.items():
    filepath = os.path.join(volume_path, filename)
    chunks = parse_pdf(filepath, guideline_name)
    all_chunks.extend(chunks)
    
    table_count = sum(1 for c in chunks if c['content_type'] == 'table')
    fig_chunks = sum(1 for c in chunks if '[图片:' in c['content'])
    avg_chars = sum(c['char_count'] for c in chunks) // max(len(chunks), 1)
    
    print(f"\n📄 {guideline_name}")
    print(f"   chunks: {len(chunks)} | 表格: {table_count} | 含图片标记: {fig_chunks} | 平均字符: {avg_chars}")

# 全局重新编号
for i, c in enumerate(all_chunks):
    c['chunk_id'] = i + 1

print(f"\n{'=' * 70}")
print(f"✅ 总计: {len(all_chunks)} chunks")
char_counts = [c['char_count'] for c in all_chunks]
print(f"   字符: min={min(char_counts)}, max={max(char_counts)}, avg={sum(char_counts)//len(char_counts)}")
print(f"   表格chunks: {sum(1 for c in all_chunks if c['content_type'] == 'table')}")
print(f"   段落chunks: {sum(1 for c in all_chunks if c['content_type'] == 'paragraph')}")

# COMMAND ----------

# DBTITLE 1,质量检查说明
# MAGIC %md
# MAGIC ## 质量检查
# MAGIC
# MAGIC 验证分块结果的质量：字符分布、表格完整性、内容抽样。

# COMMAND ----------

# DBTITLE 1,质量检查
import statistics

print("=== 分块质量检查 ===\n")

# 1. 字符分布
char_counts = [c['char_count'] for c in all_chunks]
print("1. 字符分布:")
print(f"   中位数: {statistics.median(char_counts):.0f}")
print(f"   <100字:   {sum(1 for x in char_counts if x < 100):4d} chunks")
print(f"   100-300字: {sum(1 for x in char_counts if 100 <= x < 300):4d} chunks")
print(f"   300-600字: {sum(1 for x in char_counts if 300 <= x < 600):4d} chunks")
print(f"   600-800字: {sum(1 for x in char_counts if 600 <= x < 800):4d} chunks")
print(f"   >800字:   {sum(1 for x in char_counts if x >= 800):4d} chunks")

# 2. 表格chunk验证
print(f"\n2. 表格chunk质量:")
table_chunks = [c for c in all_chunks if c['content_type'] == 'table']
real_tables = sum(1 for c in table_chunks if c['content'].split('\n')[0].startswith('表'))
print(f"   总表格chunks: {len(table_chunks)}")
print(f"   以'表N'开头: {real_tables} (正确识别的表格标题)")
print(f"   其他表格内容: {len(table_chunks) - real_tables} (表格后续数据)")

# 3. Section路径分布
print(f"\n3. Section路径分布:")
no_section = sum(1 for c in all_chunks if c['section'] == '摘要/前言')
has_l1 = sum(1 for c in all_chunks if '第' in c['section'] and '章' in c['section'])
print(f"   有章节路径: {len(all_chunks) - no_section}")
print(f"   摘要/前言: {no_section}")
print(f"   含'第X章': {has_l1}")

# 4. 抽样展示
print(f"\n4. 内容抽样:")
# 药物治疗相关
drug = [c for c in all_chunks if '药物' in c['section'] and c['char_count'] > 200]
if drug:
    c = drug[0]
    print(f"\n   【段落示例】")
    print(f"   Section: {c['section']}")
    print(f"   Chars: {c['char_count']} | Pages: {c['page_start']}-{c['page_end']}")
    print(f"   Content: {c['content'][:150]}...")

# 表格示例
if table_chunks:
    c = [t for t in table_chunks if t['content'].startswith('表')][0]
    print(f"\n   【表格示例】")
    print(f"   Section: {c['section']}")
    print(f"   Chars: {c['char_count']} | Pages: {c['page_start']}-{c['page_end']}")
    print(f"   Content: {c['content'][:150]}...")

# COMMAND ----------

# DBTITLE 1,写入Delta说明
# MAGIC %md
# MAGIC ## 写入Delta Table
# MAGIC
# MAGIC 将解析结果写入 `catalog1.medical_knowledge.chunks`，作为后续Vector Search索引的数据源。

# COMMAND ----------

# DBTITLE 1,写入Delta Table
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 定义Schema
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

# 创建DataFrame并写入
df = spark.createDataFrame(all_chunks, schema=schema)
df.write.format("delta").mode("overwrite").saveAsTable("medical.medical_knowledge.chunks")

# 验证
count = spark.sql("SELECT COUNT(*) as cnt FROM medical.medical_knowledge.chunks").collect()[0]['cnt']
print(f"✅ 写入成功: medical.medical_knowledge.chunks")
print(f"   总记录数: {count}")

# COMMAND ----------

# DBTITLE 1,验证说明
# MAGIC %md
# MAGIC ## 验证
# MAGIC
# MAGIC 通过SQL查询验证写入结果。

# COMMAND ----------

# DBTITLE 1,统计验证
# MAGIC %sql
# MAGIC -- 按指南和内容类型统计
# MAGIC SELECT 
# MAGIC     guideline_name, 
# MAGIC     content_type, 
# MAGIC     COUNT(*) as chunk_count, 
# MAGIC     ROUND(AVG(char_count)) as avg_chars,
# MAGIC     MIN(char_count) as min_chars,
# MAGIC     MAX(char_count) as max_chars
# MAGIC FROM medical.medical_knowledge.chunks
# MAGIC GROUP BY guideline_name, content_type
# MAGIC ORDER BY guideline_name, content_type

# COMMAND ----------

# DBTITLE 1,表格chunks抽样
# MAGIC %sql
# MAGIC -- 抽样查看表格chunks
# MAGIC SELECT 
# MAGIC     chunk_id, 
# MAGIC     guideline_name, 
# MAGIC     section, 
# MAGIC     content_type, 
# MAGIC     char_count, 
# MAGIC     LEFT(content, 200) as preview
# MAGIC FROM medical.medical_knowledge.chunks
# MAGIC WHERE content_type = 'table'
# MAGIC ORDER BY guideline_name, page_start
# MAGIC LIMIT 10

# COMMAND ----------

# DBTITLE 1,待处理PDF说明
# MAGIC %md
# MAGIC ## 待处理: 需要OCR/多模态的PDF (3份)
# MAGIC
# MAGIC 以下PDF由于是扫描版或OCR质量差，无法用PyMuPDF直接提取有效文本，需要后续使用多模态AI处理：
# MAGIC
# MAGIC | # | 文档 | 页数 | 问题 | 建议方案 |
# MAGIC |---|------|------|------|----------|
# MAGIC | 1 | 《糖尿病分型诊断中国专家共识》临床实践应用 | 8页 | 扫描版，无法复制文字 | ai_parse_document() |
# MAGIC | 2 | 中国高血压临床实践指南（2024版） | 48页 | 扫描版，16.4MB | Azure Document Intelligence |
# MAGIC | 3 | 内分泌性高血压筛查专家共识（2025版） | 14页 | OCR质量差，乱码多 | ai_parse_document() |
# MAGIC
# MAGIC 处理完成后，将结果追加到 `medical.medical_knowledge.chunks` 表中（使用 `mode("append")`）。