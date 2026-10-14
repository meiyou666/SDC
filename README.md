# SDC

## 使用教程

首次使用先搭建环境，进入容器后按任务选择教程。

| 要做什么 | 阅读入口 |
| --- | --- |
| 在本地或云服务器搭建环境、选择和切换容器 | [搭建开发环境](environment/README.md) |
| 准备数据、运行 Qwen BF16/FP8/FP4 训练、查看结果和恢复训练 | [运行训练](docs/training.md) |
| 编译、检查和修改 CUTLASS GEMM，并接入训练 | [开发 GEMM](docs/gemm.md) |

## 项目目录

| 目录 | 用途 |
| --- | --- |
| `src/`、`tests/` | 实现与测试；公共训练在 `src/training/`，CUTLASS 在 `src/cutlass_gemm/`，FP8/FP4 在 `src/low_precision/` |
| `environment/`、`.devcontainer/` | Docker 环境与 VS Code 容器入口 |
| `configs/`、`experiments/` | 公共配置与实验配置 |
| `docs/` | 教程、论文精读及复现记录 |
| `data/`、`results/` | 本地模型、数据与实验输出，不提交 Git |
| `archive/` | 历史实验归档 |

Agent 从 [AGENTS.md](AGENTS.md) 开始，读取[项目研究说明](docs/project.md)和[技术参考](.agent/reference.md)。

## 协作流程

所有成员在个人分支开发，通过 PR 合并到 `main`。拥有者使用 `meiyou666/dev`；其他成员使用自己的分支。公共环境、训练配置和 CI 由 `meiyou666` 维护。

**1. 开始任务。** 在宿主机仓库目录执行，将 `name/task` 换成自己的分支名：

```bash
git fetch origin
git switch -c name/task origin/main
```

已有分支时使用 `git switch name/task`，再执行 `git pull --ff-only` 和 `git merge origin/main`。

**2. 修改并检查。** 代码改动在项目容器中检查：

```bash
bash scripts/check.sh
```

**3. 提交个人分支。** 回到宿主机仓库目录，检查改动并提交选定文件：

```bash
git status
git diff
git add <本次修改的文件>
git commit -m "简述改动"
git push -u origin HEAD
```

也可使用 VS Code 或 GitHub Desktop 提交。随后在 GitHub 网页点击 **Compare & pull request**，目标分支选择 `main`，填写改动与检查结果。CI 通过并获得成员批准后合并。
