# CarePilot · 多 Agent 智能客服系统

CarePilot 是一个面向客服场景的多 Agent 智能系统。它不是简单的「聊天机器人」，
而是把意图识别、Agent 路由、意图驱动 RAG、分层记忆、动态 Skills 注入、
在线监控与离线评测串成一个可观测、可评测、可降级的闭环。

## 核心能力

- **细粒度意图识别**：LLM + Embedding + 关键词三路融合打分，覆盖 19 类意图。
- **路由驱动的多 Agent 编排**：按意图路由到 General / Technical / Billing / Escalation，支持主备 Agent 协同。
- **意图驱动 RAG**：按意图决定是否检索；检索前查询改写、检索后 LLM 重排。
- **分层记忆**：Redis 工作记忆 + ChromaDB 情景记忆与用户画像。
- **动态 Skills**：以 Markdown SOP 形式热加载业务规则，无需改代码。
- **工具层治理**：TTL 缓存、熔断、失败降级。
- **可观测 + 可评测**：Prometheus 指标、在线监控告警、LLM-as-Judge 离线评测。

## 技术栈

| 层 | 组件 |
|---|---|
| 后端 | Python · FastAPI · Anthropic SDK |
| 存储 | Redis（工作记忆）· ChromaDB（向量 / 情景记忆）|
| 前端 | Vue 3 · Vite（单页调试工作台）|
| 监控 | Prometheus |
| 部署 | Docker · Docker Compose · Nginx |

## 快速开始

### Docker 全栈一键

```bash
export ANTHROPIC_API_KEY=your_key
docker compose -f frontend/docker-compose.yml up -d --build
```

启动后：
- 前端工作台：http://localhost
- API：http://localhost:8000
- Swagger：http://localhost:8000/docs

### 本地开发

```bash
# 后端（需先起 Redis / ChromaDB）
cd backend && python -m venv .venv \
  && .venv/Scripts/python -m pip install -r requirements.txt \
  && .venv/Scripts/python -m uvicorn api.main:app --reload

# 前端
cd frontend && npm install && npm run dev   # http://localhost:5173
```

## 目录结构

```text
CarePilot/
├── backend/                 FastAPI 后端
│   ├── api/main.py          应用入口与 REST 路由
│   ├── agents/              多 Agent 编排与工具定义
│   ├── core/                意图识别 / Skill 加载 / LLM 工具
│   ├── memory/              Redis + ChromaDB 分层记忆
│   ├── mcp/                 知识库与工具管理（改写/重排/熔断）
│   ├── monitor/             Prometheus 指标与告警
│   ├── evaluation/          离线评测套件
│   ├── skills/              动态 SOP 技能文档
│   └── tests/               单元测试
└── frontend/                Vue 3 调试工作台
    └── src/                 App.vue / lib/backends.js
```

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/chat` | 对话主链路 |
| POST | `/search` | 知识库检索 |
| POST | `/knowledge/add` · `/knowledge/upload` | 导入 / 上传文档 |
| GET | `/knowledge/stats` | 知识库统计 |
| GET | `/skills` · POST `/skills/reload` | Skills 列出 / 热加载 |
| GET | `/monitor` | 监控摘要 |
| GET | `/metrics` | Prometheus 指标 |
| POST | `/eval/run` | 运行离线评测 |
| GET | `/health` | 健康检查 |
