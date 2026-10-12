# 团队共享 skills

| Skill | 用途 | 入口 |
| --- | --- | --- |
| `revise-evidence-report` | 整理和润色研究报告，结合数据、引用与已有图表生成交付文档 | [SKILL.md](skills/revise-evidence-report/SKILL.md) |

Agent 根据根目录 `AGENTS.md` 在报告任务开始时读取上述入口，再按任务需要读取 `references/`。团队克隆仓库即可获取完整 skill，无需个人重复下载。可以直接提出：

> 使用 `.agent/skills/revise-evidence-report/SKILL.md`，根据这些实验数据与图表整理研究报告。

来源：[meiyou666/research-skills-system](https://github.com/meiyou666/research-skills-system/tree/6370a59c728febae3560321af2769681aed3c732/skills/revise-evidence-report)。本次引入提交为 `6370a59c728febae3560321af2769681aed3c732`，保留原目录中的说明、参考文件、脚本和样式资源。后续可按团队需要更新。

报告脚本在项目容器中执行。Pandoc、PDF 引擎等写作工具按需放在个人工具环境，与项目实验依赖分开管理。
