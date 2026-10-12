# HTML 与 PDF 导出

按用户要求选择输出格式。命令在项目容器的仓库根目录执行；Pandoc、PDF 引擎等写作工具放在个人工具环境。

## 仅导出 HTML

```bash
pandoc report.md --standalone --embed-resources \
  --css .agent/skills/revise-evidence-report/assets/report.css \
  --metadata pagetitle="研究报告" --output report.html
```

## 同时导出 HTML 与 PDF

使用现有脚本，指定已安装的 Pandoc 和 PDF 引擎。以下以 WeasyPrint 为例：

```bash
python .agent/skills/revise-evidence-report/scripts/render_report.py report.md \
  --html-output report.html --pdf-output report.pdf \
  --manifest-output report.render.json \
  --pandoc pandoc --pdf-engine weasyprint
```

脚本自动保存渲染记录。只需 PDF 时也可使用该命令，并按用户要求交付 PDF。

- 资源在其他目录时，用 `--resource-path 目录` 补充；已有文件需要更新时使用 `--force`。
- 模板样式用 `--css 文件` 指定。WeasyPrint 使用 CSS；LaTeX 引擎使用自身排版设置。
- 正文保留一个报告标题，`--title` 设置网页标题。字体需要在引擎中正确配置，`--font-file` 仅记录字体文件。
- 更多参数通过 `python .agent/skills/revise-evidence-report/scripts/render_report.py --help` 查看。

导出后打开文件，检查中文字体、表格分页、图片清晰度和遮挡，并核对标题、数值与图注。源文修改后重新导出所交付的文件。
