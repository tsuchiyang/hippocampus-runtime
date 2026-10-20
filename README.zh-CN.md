# Hippocampus Runtime（海马体）

面向编程智能体的本地记忆与上下文恢复系统，目前主要对接 Codex。

[English](README.md) · [安装、备份与停用](docs/OPERATIONS.md) · [测试范围](docs/TESTING.md)

实验性 Alpha：给 Codex 提供存放在本地的结构化记忆与压缩恢复线索，
不是无限上下文，也不是 OpenAI 官方产品。安装不会注册 Hook，不默认开启 dense 或 active。
Hippocampus Runtime 现已在 GitHub 公开开源，采用 MIT 许可证，当前定位仍为实验性 Alpha，而非生产稳定版。
Windows 完整 CI 与 Linux 限定核心 CI 已通过；结果及验证边界见[已知问题与当前证据](docs/KNOWN_ISSUES.md)。

## 名称与兼容性

项目对外名称由 Codex Memory Runtime 调整为 **Hippocampus Runtime（海马体）**。
本次只调整品牌，不迁移包或接口：

- Python 包名：`codex-memory-runtime`
- Python 导入路径：`codex_memory`
- 安装后的命令：`codex-memory-hook`

现有命令、`.codex-memory/` 数据路径和安装标识均保持不变。
历史源码和版权声明可以保留原项目名称，不因品牌调整而改写。

## 能做什么

- 保存观察到的提示、工具事件和压缩 checkpoint；压缩后可以恢复经过核验的线索。
- 用 SQLite 追加式账本记录永久记忆、证据和生命周期。
- 区分同 session 的未审核临时记忆与审核后的永久记忆。
- 提供确定性 exact / BM25 / symbol / state 检索；索引只是可删除重建的缓存。
- 恢复输出失败时保留可重试的 checkpoint，不将写入 stdout 当成宿主已经确认接收。

不会自动从自然语言抽取“事实”或自动批准记忆。需要操作者/模型显式 capture 和 review；
相关性分数不会把记忆变成真理或覆盖用户指令。

## 环境和安装

需要 CPython 3.12.x、Git、可写的普通本地磁盘。主要测试平台是 Windows。
Linux 已有本地确定性范围验证，Windows 专属的安全句柄/实验增强不能视为 Linux 兼容。
Runtime 没有第三方运行依赖；开发和构建依赖另算。

在源码根目录，用新的虚拟环境安装（不会改变系统安全设置）：

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install .
& .\.venv\Scripts\python.exe -c "from importlib.metadata import version; print(version('codex-memory-runtime'))"
```

Linux 使用 `python3.12 -m venv .venv`、`.venv/bin/python -m pip install .`。
源码安装可能需要下载构建工具；已有 wheel 可用 `--no-index --no-deps` 离线安装。
目前不宣称已在 PyPI 发布，不要把索引中的同名包当成此项目。

## 怎么试

先在独立、非敏感 Git 项目试用，用安装入口的绝对路径运行命令，
不要复制其他机器的 `.venv`、私人记忆数据库或 Hook 配置。
命令包括 `record`、`ledger verify`、`memory capture/evidence/pending/recall/review/expire`、
`retrieval init/status/rebuild/verify/query`。先读 [命令手册](docs/CLI.md)。
此 Alpha 保留原有严格 CLI，尚未实现 `--help`/`--version`；上面的 Python 命令用于查看安装版本。

状态保存在项目 `.codex-memory/`。它可能含完整提示和工具输出，应当按敏感数据管理。
安装、备份、恢复、卸载与清理见 [运维说明](docs/OPERATIONS.md)。
真实宿主接线必须另行选择项目/session，审核配置与信任边界，见 [接线边界](docs/INTEGRATION.md)。
本仓库没有有效 `.codex` 配置，不会因为 clone 或安装就启用 Hook。

## 开发与限制

```text
uv sync --locked
uv run --no-sync pytest -q -rs
uv run --no-sync python scripts/check_release.py --history
uv build --no-sources
```

上面的完整测试命令面向 Windows。Linux 使用 [测试说明](docs/TESTING.md) 中明确列出的本地核心集，
不把 Windows 专属增强当成跨平台能力。
公开测试使用合成数据和 loopback 测试服务，不调用真实模型、不读取登录、不启用真实 Hook。
平台跳过不算 PASS；CI 配置存在不代表 GitHub 已实际执行。原工程历史专用验收没有混入当前分数。
详见 [测试范围](docs/TESTING.md) 与 [版本来源](docs/PROVENANCE.md)。

Alpha 不承诺生产稳定、任意宿主版本兼容、exactly-once 或长期零故障；
专用 sparse/reranker live、全局 active 和默认 dense 不属于本次上线范围。
安全问题不要贴原始记忆、认证或 transcript，先读 [SECURITY.md](SECURITY.md)。

MIT 许可证见 [LICENSE](LICENSE)，贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。
