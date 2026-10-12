# Delivery and QA

Render from the current editable source and validate both content and presentation.

## Render with explicit tools

Require Python 3.10+, Pandoc, and a Pandoc-compatible HTML-to-PDF engine. Select the engine for the available environment and record its identity. Run:

```sh
python3 scripts/render_report.py report.md \
  --html-output report.html \
  --pdf-output report.pdf \
  --manifest-output delivery-manifest.json \
  --pandoc pandoc \
  --pdf-engine weasyprint
```

Pass `--css` for a task stylesheet, repeat `--resource-path` for additional asset roots, and repeat `--pdf-engine-option` for engine-specific settings. The default stylesheet is the skill-relative `assets/report.css`. Existing outputs require `--force`.

The renderer resolves both Pandoc and the PDF engine before rendering. The manifest records each
requested basename, actual executable name, version probe, byte count, and SHA256. It omits
controller-local executable and font paths so the delivery remains portable; the hash and version
bind the observed tool bytes. A normal engine must return a usable `--version`.
The renderer prepends the resolved engine directory to the child `PATH` and gives Pandoc the
requested executable basename because Pandoc rejects an absolute `--pdf-engine` pathname. Use a
Pandoc-supported engine basename; an explicit path is still accepted for identity binding when
its basename is supported by the installed Pandoc version.

For `pdflatex`, `xelatex`, `lualatex`, `latexmk`, and `tectonic`, the renderer also applies the
bundled `assets/fit-latex-tables.lua` filter. It assigns equal wrapping widths only to tables whose
HTML representation has no explicit colspec; author-supplied widths remain unchanged. The
manifest hashes this backend asset. Equal widths are a safe fit baseline, not a claim that column
allocation is editorially optimal, so inspect dense tables in the current PDF and override their
source layout when needed.

LaTeX engines do not apply the HTML stylesheet to the PDF. Their PDF styling comes from Pandoc's
LaTeX defaults, engine options, and the table filter. CSS-oriented engines can consume the staged
HTML styling, subject to that engine's feature support. The manifest records the applicable
`pdf_style_mode`; inspect the current PDF instead of inferring style parity from the HTML.

When `--pdf-engine` is an orchestration wrapper rather than the renderer itself, pass:

```sh
--pdf-engine-kind wrapper \
--pdf-engine-wrapper-description "What this pinned wrapper launches and where identity is bound"
```

The manifest then says `kind: wrapper` and hashes the wrapper executable. The wrapper basename
must be one that the installed Pandoc accepts as a PDF engine. A failed wrapper version probe is
recorded as unavailable rather than misrepresented as the downstream engine's version. Keep
downstream service/image identity in the wrapper description or a task-local provenance record.

Use `--title` only for the HTML page title; keep the single visible report title in Markdown. The
renderer passes `pagetitle` metadata for standalone HTML, clears the imported HTML title metadata
during PDF conversion, and rejects duplicate normalized H1 text in generated HTML. The report's
Markdown H1 remains the only visible title in both formats.

The renderer extracts declared CSS font families. Repeat `--font-file PATH` to hash supplied font
files into the manifest. Supplied files are provenance inputs, not proof that the engine selected
or embedded them; the automated QA therefore retains a font warning until the current PDF font
table and rendered pages are inspected.

The renderer builds both artifacts in staging, checks standalone HTML, local-file URIs, page
title, duplicate H1s, and the PDF signature, publishes only after both conversions succeed, and
writes these automated checks into the manifest. It invokes subprocesses without a shell. A
successful converter may still report missing glyphs, fonts, resources, or layout warnings on
stderr. The renderer relays that text to the current operator and records only its line count and
SHA256 in the portable manifest; any such diagnostic leaves a review warning instead of being
silently treated as a clean render.

After HTML, PDF, and manifest staging completes, the renderer rechecks all three destination
paths before changing any of them. An absent path is eligible for creation. An existing path is
eligible only with `--force` and only when it is a regular file; `--force` does not authorize a
symlink, directory, device, socket, or other special file. The renderer first prepares a
same-directory replacement for every artifact and snapshots the bytes of every old regular file.
If any preparation or pre-commit identity check fails, it removes the temporary files and leaves
all destinations unchanged.

Publication replaces HTML, PDF, and then the manifest. If a replacement raises an exception, the
renderer restores each attempted destination to its snapshotted bytes (or to absence), verifies
the restored hashes, removes transaction files, and reports failure. This is a process-level
rollback guarantee for caught publication errors. Each pathname replacement is atomic, but three
independent pathnames cannot be observed as one filesystem operation: a concurrent reader can
briefly see an in-progress set, and process termination, host failure, or storage failure can
interrupt rollback. The guarantee assumes no concurrent writer; pre-commit identity checks detect
preparation-time changes but do not lock external processes. Treat the manifest as the commit
marker and accept a delivery only when its artifact hashes match; use a transactional storage
layer or an atomic directory-generation pointer when crash-atomic multi-file publication is
required.

## Inspect HTML

Check the self-contained artifact in representative viewport widths and print rendering:

- headings, paragraphs, links, lists, code, tables, figures, captions, and source tags;
- keyboard navigation, semantic heading order, link purpose, alt text, and contrast;
- resource embedding and absence of broken local references;
- overflow, overlap, clipping, excessive density, and unreadable text;
- agreement with the current Markdown source.

Confirm that the browser tab title matches the requested page title and that the visible report
title appears once. Multiple genuinely distinct H1 headings are permitted; duplicate normalized
H1 text fails automated rendering as a title-block duplication risk.

If the selected Pandoc version lacks resource embedding support, stop derived delivery and report the renderer requirement. Do not label externally dependent HTML as self-contained.

## Inspect PDF

Render every page to images when a suitable tool is available, then inspect page order, margins, headings, tables, figures, captions, source tags, page numbers, blank pages, orphans, clipping, overlap, and effective text size. Check that text is selectable and fonts render as intended. Inspect the PDF font table with `pdffonts` or an equivalent tool when available; record actual font names, embedding/subsetting state, tool version, and the PDF digest in task-local QA. Reconcile those results with the manifest's declared families and supplied font hashes. Treat missing rasterization or font-table capability as a QA warning unless the user requires publication-grade visual acceptance.

## Reconcile evidence across formats

For Markdown, HTML, and PDF, compare:

- title, section order, central conclusions, and qualification;
- material numbers, units, denominators, exclusions, and uncertainty;
- table and figure counts, versions, captions, and source identities;
- links, source tags, appendices, and unresolved-state labels.

Regenerate all derived artifacts after any accepted source change. A non-critical tooling or metadata gap may remain a warning on a draft; a mismatched central claim, figure, number, source, or stale derived artifact fails delivery acceptance.

## Record QA

Record `PASS`, `WARNING`, or `FAIL` for HTML visual/accessibility, HTML evidence consistency, PDF visual, PDF evidence consistency, and effective PDF fonts. Include the checked artifact digest, QA tool identity, and only material findings. The renderer manifest's `NOT_RUN` page/content checks are not acceptance. Use `PASS` only after checking the current digest.
