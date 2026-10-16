# CarePilot 后端

面向客服场景的多 Agent 智能系统，把以下能力串成闭环：

- 细粒度意图识别
- 路由驱动的多 Agent 编排
- 意图驱动 RAG 检索
- Redis + ChromaDB 分层记忆
- 动态 Skills 注入
- 在线监控与路由降权
- LLM-as-Judge 端到端评测

## 快速开始（Docker）

```bash
cp .env.example .env       # 填入 ANTHROPIC_API_KEY
docker compose up -d --build
```

访问：API `http://localhost:8000` · Swagger `http://localhost:8000/docs`

## 快速开始（本地）

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
python -m uvicorn api.main:app --reload --port 8000
```

需先启动 Redis 与 ChromaDB（可只用 `docker compose up -d redis chromadb` 起这两个）。

## 环境变量

```env
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic   # 可选：兼容第三方模型服务
ANTHROPIC_MODEL=deepseek-chat                            # 可选
REDIS_PASSWORD=carepilot123
```

## 核心接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 对话主链路：记忆 → 意图 → 检索 → 路由 → 生成 → 写回 |
| POST | `/search` | 知识库检索 |
| POST | `/knowledge/add` · `/knowledge/upload` | 导入 / 上传文档 |
| GET | `/knowledge/stats` | 知识库统计 |
| GET | `/skills` · POST `/skills/reload` | Skills 列出 / 热加载 |
| GET | `/monitor` · `/metrics` | 监控摘要 / Prometheus 指标 |
| POST | `/eval/run` | 运行离线评测 |
| GET | `/health` | 健康检查 |

## 项目结构

```text
api/main.py                   FastAPI 入口与路由
agents/agent_orchestrator.py  多 Agent 编排
agents/tools.py               共享 RAG 工具
core/intent_recognizer.py     三路融合意图识别
core/skill_loader.py          动态 Skills 加载
memory/conversation_memory.py Redis + ChromaDB 分层记忆
mcp/knowledge_base.py         ChromaDB 知识库
mcp/tool_manager.py           工具管理（改写 / 重排 / 缓存 / 熔断）
monitor/performance_monitor.py 在线监控与告警
evaluation/                   离线评测套件
skills/                       动态 SOP 业务规则
tests/                        单元测试
```

## 运行时架构

```text
用户请求
  -> /chat
  -> MemoryManager 读取工作记忆、情景记忆、用户画像
  -> IntentRecognizer 输出 intent / intent_group / urgency / entities
  -> 按意图决定是否检索知识库
  -> AgentOrchestrator 路由到 General / Technical / Billing / Escalation
  -> Skills 注入、工具调用、回复生成
  -> 写回 Redis 与 ChromaDB
  -> Monitor 采集在线指标 / Evaluator 离线评测
```

## 主要端口

| 服务 | 端口 |
|---|---:|
| CarePilot API | 8000 |
| ChromaDB | 8001 |
| Redis | 6379 |
| Prometheus | 9090 |
| Nginx | 80 |

## 测试

```bash
python -m pytest tests/ -v
```
