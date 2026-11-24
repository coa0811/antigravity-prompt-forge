# Prompt Forge

轻量原生 Antigravity 提示词优化与细节补全插件：通过 `++` 前缀或自然语言技能触发，将简短/模糊的原始指令结合当前仓库真实上下文、场景模板与跨项目个人偏好改写为可直接执行的严谨提示词。

## 触发语法

### 1. 前缀快捷触发（`PreInvocation` Hook）

平时对话无需任何后缀或配置，零打扰、零延迟；仅当消息以 `++` 或全角 `＋＋` 开头时激活（代码片段如 `++i`、`C++` 不会误触发）：

| 写法 | 模式 | 场景 | 说明 |
| :--- | :--- | :--- | :--- |
| `++ 帮我修一下登录报错` | 默认（`confirm`） | 自动识别 | 先展示优化稿并询问确认，再执行 |
| `＋＋ 帮我写个部署脚本` | 默认（`confirm`） | 自动识别 | 支持全角 `＋＋` |
| `++only 帮我整理接口文档` | `only`（别名 `copy` / `draft`） | 自动识别 | 只输出优化后的提示词，不执行任务 |
| `++go 补全测试并跑一遍` | `execute`（别名 `run`） | 自动识别 | 展示要点后直接按优化稿执行 |
| `++code 重构缓存模块` | `confirm` | `code` | 指定场景（支持 `code`/`plan`/`research`/`doc`/`general`） |
| `++only:research 对比两个靶点` | `only` | `research` | 组合模式 + 场景 |
| `++:<custom_id> 原始指令` | `confirm` | 自定义模板 | 使用个人保存的自定义模板 |
| `++only:<custom_id> 原始指令` | `only` | 自定义模板 | 仅输出按自定义模板优化的提示词 |
| `++prefs` / `++prefs 新增约束：...` | `prefs` | — | 查看或修改个人偏好与模板库 |

### 2. 自然语言触发（Skill）

直接用自然语言提问即可触发 `prompt-forge` 技能，例如：
- 「帮我优化这段提示词再执行：……」
- 「查看并管理我的 Prompt Forge 个人偏好和自定义模板」

## 核心能力

- **仓库上下文感知（只读有界）**：自动抓取项目目录树（深度 ≤ 2）、Git 分支/状态/最近提交、根目录 `AGENTS.md`/`GEMINI.md`/`CLAUDE.md` 核心约束摘要、以及原话中提及的真实相对路径（严格过滤 `.git`、`.venv`、`node_modules` 及 `.env`/`*.key`/`*secret*` 等敏感文件）。
- **智能反问补齐**：仅当存在影响方向或不可逆决策的关键未知项时，先提出 1–3 个高价值选择题；意图明确则直接生成优化稿。
- **多场景框架**：内置 `code`、`plan`、`research`、`doc`、`general` 五套结构化模板，支持自动匹配或显式指定，并可自定义模板覆盖或扩展。
- **跨项目个人偏好**：通过纯 Python 原子 JSON 存储于 `~/.gemini/antigravity/plugin_data/prompt-forge/`（或 `PLUGIN_DATA`），跨仓库持久化输出语言、默认模式、常用约束、避坑项与专业术语表。

## 目录结构

```text
prompt-forge/
├── plugin.json
├── hooks.json
├── LICENSE
├── README.md
├── rules/
│   └── AGENTS.md
├── skills/
│   └── prompt-forge/
│       └── SKILL.md
├── templates/
│   └── defaults.json
├── scripts/
│   ├── prompt_forge_cli.py
│   ├── prompt_forge_core.py
│   ├── prompt_forge_hook.py
│   └── run_python.sh
└── tests/
    └── test_prompt_forge.py
```

## 安装、验证与启用方式

```bash
# 0. 获取仓库并进入插件根目录
git clone https://github.com/coa0811/antigravity-prompt-forge.git
cd antigravity-prompt-forge

# 1. 运行单元测试
./scripts/run_python.sh -m unittest discover -s ./tests -v

# 2. 校验插件清单与目录结构
agy plugin validate . --strict

# 3. 安装或软链到 Antigravity 全局插件目录
agy plugin install .
# 或开发模式软链：
ln -s "$(pwd)" ~/.gemini/config/plugins/prompt-forge
```

## License

[MIT](LICENSE)

