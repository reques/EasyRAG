# 🚀 EasyRAG

> 一个轻量级、高性能的检索增强生成（RAG）系统，融合多策略检索优化与可扩展 Agent 能力，面向真实应用场景设计。

---

## 📖 项目简介

EasyRAG 是一个面向工程落地的 RAG 框架，聚焦于：

- 高精度检索  
- 可扩展 Agent 能力  
- 工程可用性与可部署性  

系统构建了从 **数据处理 → 检索增强 → 大模型生成** 的完整闭环，适用于知识问答、智能助手等场景。

---

## 🧱 系统架构
Data Processing → Vector Indexing → Retrieval Enhancement → LLM Generation

- 支持端到端 RAG 流水线  
- 模块化设计，便于扩展与二次开发  

---

## 🔧 核心功能

### 🔍 高精度检索
- 基于 Milvus 向量数据库 + BGE 嵌入模型  
- 支持语义分块（chunking）与检索策略优化  

### 🧠 Agent 扩展能力
- 支持 Function Call 工具调用机制  
- 可扩展为多步任务执行（Agent Workflow）  

### 🧩 结构化检索结果
- 返回 chunk / doc / metadata 等结构化信息  
- 提升可解释性与调试能力  

### ⚙️ 多策略检索机制
- Top-K 检索  
- 条件过滤  
- 多策略融合，提升复杂查询鲁棒性  

### 📡 工程化能力
- 基于 FastAPI 的服务封装（支持流式输出）  
- 支持系统集成与前后端对接  

### 📄 多源数据接入
- 支持 OCR 文本解析  
- 支持 PDF / Markdown / 多模态数据  

---

## ⚙️ 核心能力总结

- **检索优化能力**：语义向量 + 检索策略融合  
- **Agent 能力**：Function Call 工具调用机制  
- **数据接入能力**：多格式文档解析与知识库构建  
- **工程部署能力**：API 服务 + Docker 部署  
- **可观测性**：支持中间结果追踪与调试  

---

## 📈 性能表现

- 通过分块与检索策略优化，显著提升 **Precision@K**  
- 在复杂查询场景中，回答相关性与稳定性明显增强  
- 支持持续优化（检索策略 / Prompt / 工具调用）  

---

## 🛠 技术栈

- **LLM**：OpenAI / DeepSeek / 兼容 API  
- **Embedding**：BGE（BAAI）  
- **向量数据库**：Milvus  
- **后端框架**：FastAPI  
- **RAG框架**：LangChain / LlamaIndex（可选）  
- **部署方式**：Docker  

---

## 📦 项目结构
EasyRAG/
├── app/ # 核心逻辑
├── models/ # 模型文件（不包含，需自行下载）
├── api/ # FastAPI 服务（开发中）
├── scripts/ # 工具脚本（开发中）
├── README.md


---

## 📥 模型与数据

> ⚠️ 本仓库不包含模型文件

请从以下位置下载模型：

- HuggingFace（推荐）
- 其他网盘 / 对象存储（待补充）

下载后放置到：
/models/

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/reques/EasyRAG.git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Docker（Milvus 等服务）
docker compose up -d

# 4. 启动项目
python run.py

🔮 后续规划
多智能体工作流（LangGraph）
混合检索与重排序（Hybrid / Rerank）
知识图谱增强（Graph RAG）
自动化评估体系（Evaluation Pipeline）

⭐ 支持项目

如果这个项目对你有帮助，欢迎点个 Star ⭐！

也欢迎提交 Issue 或 PR 一起完善项目 🙌
