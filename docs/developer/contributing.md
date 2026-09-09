---
myst:
  heading_anchors: 2
---

# 贡献流程

从一个可描述、可验证的问题开始贡献。错误报告给出最小复现，新增模型说明来源和用途，
接口改动先查看对应模块的讨论。代码与设计入口见 [项目导航](project_layout.md)。

## 配置开发环境

在独立 Python 环境中克隆仓库；外部贡献者可以先在 GitHub 创建 fork，再克隆自己的副本。
以下命令克隆主仓库，安装开发依赖和提交检查工具：

```bash
git clone https://github.com/chaobrain/braincell.git
cd braincell
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

Windows PowerShell 将激活命令替换为 `.venv\Scripts\Activate.ps1`。
已有独立环境时可直接安装。Python 版本要求以 `pyproject.toml` 的 `requires-python` 为准。

`dev` 包含测试、文档和 pre-commit 依赖，定义见
[pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml)。
CPU/GPU 安装选择见 [安装指南](../getting_started/installation.ipynb)，
环境问题见 [开发排错](troubleshooting.md)。下文命令均在仓库根目录执行。

新增依赖写入 `pyproject.toml`；`requirements*.txt` 仅供 CI 等工具引用这些 extras。

## 修改与验证

1. **定位事项**：阅读 [模块 TODO](https://github.com/chaobrain/braincell/blob/main/docs/design/TODO.md) 和对应 Current；涉及新设计时，说明要解决的问题并链接相关 proposal。
2. **建立工作分支**：例如 `git switch -c fix/swc-validation`，让一个 PR 聚焦一个问题。
3. **形成可验证的修改**：修复错误时先写失败的复现测试；新增模型按 [扩展指南](extending.md) 验证其动力学和集成行为。
4. **运行相关检查**：先运行修改模块的测试及受影响的示例，共享行为变更再扩大测试范围，具体命令见 [测试指南](testing.md)。
5. **提交前核对**：检查本次修改涉及的 Design、实现与测试、实际相关示例，补齐受影响的说明和调用。

物理单位、随机数、编译循环、导入、许可证及 docstring 的约定集中在
[仓库规则](https://github.com/chaobrain/braincell/blob/main/AGENTS.md)。
提交前同步检查的范围见
[Design、代码与示例](https://github.com/chaobrain/braincell/blob/main/AGENTS.md#design-code-and-examples)。

## 文档修改

贡献步骤写在 Developer；接口签名、公式、架构和设计讨论写在对应 Design 页面。
更新模块契约时，在 Developer 保留到具体主题的链接。说明的分类与写法见
[Design 规范](https://github.com/chaobrain/braincell/blob/main/docs/design/AGENTS.md)。
已有示例在实际位置维护，历史 specs 用于追溯旧决定。

构建网站：

```bash
python -m sphinx -b html docs docs/_build/html
```

结果位于 `docs/_build/html/index.html`。网站构建当前不执行 notebook；修改其中的可运行代码后，
应另行运行相关 notebook 或配套脚本。API reference 从 docstring 生成，Design 通过仓库链接阅读。

## 提交 PR

提交前运行格式与静态检查，检查差异中是否混入运行产物：

```bash
pre-commit run --all-files
git diff --check
git status --short
```

格式和 lint 使用 `pyproject.toml` 中配置的 Ruff，行宽为 120，保留已有引号风格。
pre-commit 可能修改文件；检查其结果后再提交。推送工作分支，按
[PR 模板](https://github.com/chaobrain/braincell/blob/main/.github/PULL_REQUEST_TEMPLATE.md) 创建 PR，描述中说明：

- 问题或使用场景，关联的 Issue、proposal 或模型来源。
- 修改后的行为，必要时给一个修改前后的具体例子。
- 实际执行的测试、示例及结果；尚未验证的部分写明原因。
- 影响已有调用的变化，以及需要的迁移步骤。

数值模型同时注明参照、单位、温度、初态、求解器、步长与比较容差；
性能结论附上测量配置及复现入口。
