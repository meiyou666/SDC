---
name: revise-evidence-report
description: Audit, co-review, revise, and deliver an existing evidence-backed research, technical, analytical, project, or briefing report. Use when Codex must trace claims to supplied evidence, apply feedback consistently across a report, clarify metrics and existing figures, embed an existing figure bundle, or render a Markdown source into self-contained HTML and PDF with final visual and evidence QA.
---

# Revise Evidence Report

Turn an existing evidence-backed report into a clear, traceable, reviewable delivery. Start with the smallest usable input, preserve accepted content, and increase assurance in proportion to claim and delivery risk.

## Establish the working set

Read [references/boundaries-and-inputs.md](references/boundaries-and-inputs.md). A report source plus its cited or supplied evidence is enough to begin. Accept evidence manifests, decision logs, style instructions, and figure bundles when available.

Record material gaps as findings and continue with a labeled draft when a responsible revision remains possible. Pause only when missing evidence would make a central conclusion misleading, a user decision materially changes the result, or additional access or authority is required.

## Audit before revising

Read [references/evidence-audit-and-coreview.md](references/evidence-audit-and-coreview.md).

1. Read the complete report and the evidence actually used by it.
2. Identify the report purpose, audience, central claims, and each section's job.
3. Map factual, numerical, statistical, and mechanism claims to direct evidence, reproducible derivation, or an explicit unresolved state.
4. Check denominators, exclusions, missing and non-finite values, uncertainty, measurement conditions, and source boundaries.
5. Present a compact, priority-ordered issue list and a section-level revision proposal when user decisions are useful.

Maintain stable issue IDs only when the review needs more than one round. Mark disputed content `UNDER_REVIEW`; keep supported revisions moving.

## Revise the whole report

Read [references/writing-and-explanation.md](references/writing-and-explanation.md).

- Convert accepted feedback into a report-wide rule and search titles, prose, tables, captions, conclusions, contents, and references for every affected instance.
- Lead sections with the finding, judgment, or action they support.
- Keep facts, derivations, interpretation, limitations, and proposals distinguishable.
- Place source tags and qualification near the claim they constrain.
- Explain every unfamiliar metric by meaning, computation, unit or range, direction, use, and limitation.
- Keep the editable source as the single source of report prose.

Treat writing preferences as task inputs or report-local decisions. Keep bundled guidance stable.

## Integrate existing figures

When the report contains figures, read [references/figure-integration.md](references/figure-integration.md). Consume existing publication-ready assets and their manifest or equivalent provenance. Verify asset identity, caption, source mapping, denominator, alt text, and final embedded readability.

Explain each substantive figure near its first use: how to read it, what is visibly supported, and what that observation means within its evidence limits. Return data, specification, or rendering defects to the figure producer as concrete findings; keep figure generation outside this skill.

## Render and validate delivery

Read [references/delivery-and-qa.md](references/delivery-and-qa.md) before creating HTML or PDF. Use `scripts/render_report.py` for the standard path:

```text
Markdown -> self-contained HTML -> PDF
```

Pass the Pandoc executable, actual PDF engine (or an explicitly described wrapper), source,
outputs, and any resource paths explicitly. Use `assets/report.css` unless the task supplies a
report stylesheet. Use `--title` for page metadata only and keep one visible title in Markdown.
Pass known font files with `--font-file`; still verify effective fonts in the current PDF.

After every accepted source change, regenerate derived formats and perform four checks:

1. HTML visual and accessibility QA.
2. HTML claim, number, figure, and source consistency QA.
3. PDF page-level visual QA.
4. PDF claim, number, figure, and source consistency QA.

Deliver the revised source, requested derived formats, render manifest, QA result, and material unresolved decisions. Describe each artifact by its current verified state.

## Resources

- [references/boundaries-and-inputs.md](references/boundaries-and-inputs.md): scope, input tiers, output states, and pause criteria.
- [references/evidence-audit-and-coreview.md](references/evidence-audit-and-coreview.md): evidence map, issue triage, and collaborative review.
- [references/writing-and-explanation.md](references/writing-and-explanation.md): stable report-writing and metric-explanation guidance.
- [references/figure-integration.md](references/figure-integration.md): figure-bundle intake, explanation, accessibility, and embedding checks.
- [references/delivery-and-qa.md](references/delivery-and-qa.md): rendering command, manifests, and final HTML/PDF QA.
- `assets/report.css`: generic paged-report stylesheet.
- `assets/fit-latex-tables.lua`: hashed width-and-wrap fallback for Pandoc LaTeX PDF engines.
- `scripts/render_report.py`: explicit Pandoc Markdown-to-HTML-to-PDF renderer.
- `scripts/self_test.py`: offline renderer interface tests.
