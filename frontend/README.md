# CarePilot 前端

CarePilot 的 Vue 3 调试工作台，用于联调后端：对话、健康检查、监控摘要、
知识库检索与导入、离线评测。

## 功能

- 对话调试：查看意图、路由、耗时、是否用了知识库
- 健康检查、监控摘要
- 知识库检索、文档导入、文件上传
- 一键运行离线评测

## 默认后端地址

| 项目 | 值 |
|------|----|
| 后端 | `http://localhost:8000` |
| 开发代理 | `/api/python` → `http://localhost:8000` |

Docker 模式下，Nginx 通过运行时注入的 `/api/python` 访问后端。

## 本地运行

```bash
npm install
npm run dev        # http://localhost:5173
```

覆盖后端地址：

```bash
VITE_PYTHON_API_URL=http://localhost:8000 npm run dev
```

## 构建与预览

```bash
npm run build
npm run preview
```

## Docker 部署

```bash
docker compose up -d --build
```

前提：父目录下有 `../backend`（后端源码）。访问 http://localhost ；停止 `docker compose down`。
