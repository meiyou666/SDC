#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class RenderError(RuntimeError):
    """Report rendering failed without publishing staged artifacts."""


LATEX_PDF_ENGINES = {"latexmk", "lualatex", "pdflatex", "tectonic", "xelatex"}


@dataclass
class PublicationEntry:
    staged: Path
    target: Path
    prepared: Path
    backup: Path | None
    old_sha256: str | None
    old_stat: os.stat_result | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RenderError(f"{label} is unavailable: {exc}") from exc
    if not resolved.is_file():
        raise RenderError(f"{label} must be a regular file")
    return resolved


def run_command(
    command: Sequence[str],
    label: str,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
    except OSError as exc:
        raise RenderError(f"cannot execute {label}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise RenderError(f"{label} failed with exit code {result.returncode}: {detail}")
    return result


def conversion_diagnostics(
    result: subprocess.CompletedProcess[str], label: str
) -> dict[str, object]:
    """Expose successful renderer diagnostics without persisting local paths."""
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stderr:
        print(f"{label} produced diagnostics:", file=sys.stderr)
        print(stderr, file=sys.stderr)
    encoded = stderr.encode("utf-8")
    return {
        "label": label,
        "stdout_line_count": len(stdout.splitlines()) if stdout else 0,
        "stderr_line_count": len(stderr.splitlines()) if stderr else 0,
        "stderr_sha256": hashlib.sha256(encoded).hexdigest() if stderr else None,
        "review_required": bool(stderr),
    }


def resolve_executable(executable: str, label: str) -> Path:
    located = shutil.which(executable)
    if located is None:
        raise RenderError(f"cannot resolve {label} executable: {executable!r}")
    resolved = require_regular_file(Path(located), f"{label} executable")
    if not os.access(resolved, os.X_OK):
        raise RenderError(f"{label} executable is not executable: {resolved}")
    return resolved


def executable_identity(
    requested: str,
    label: str,
    *,
    kind: str = "executable",
    wrapper_description: str | None = None,
    allow_version_failure: bool = False,
) -> tuple[Path, dict[str, object]]:
    resolved = resolve_executable(requested, label)
    try:
        version_result = subprocess.run(
            (str(resolved), "--version"),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RenderError(f"cannot execute {label} version probe: {exc}") from exc
    version_text = (version_result.stdout or version_result.stderr).strip()
    version = version_text.splitlines()[0].strip() if version_text else None
    if version_result.returncode != 0 or not version:
        if not allow_version_failure:
            raise RenderError(
                f"{label} version probe failed with exit code {version_result.returncode}"
            )
        version = None
        version_status = "UNAVAILABLE_WRAPPER_PROBE"
    else:
        version_status = "OBSERVED"
    identity: dict[str, object] = {
        "kind": kind,
        "requested_name": Path(requested).name,
        "executable": resolved.name,
        "resolution_scope": "caller_environment",
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "version": version,
        "version_status": version_status,
    }
    if wrapper_description is not None:
        identity["wrapper_description"] = wrapper_description
    return resolved, identity


def pandoc_identity(executable: str) -> tuple[Path, dict[str, object], str]:
    resolved, identity = executable_identity(executable, "Pandoc")
    help_result = run_command((str(resolved), "--help"), "Pandoc help probe")
    if "--embed-resources" in help_result.stdout:
        resource_option = "--embed-resources"
    elif "--self-contained" in help_result.stdout:
        resource_option = "--self-contained"
    else:
        raise RenderError("Pandoc does not advertise a self-contained HTML option")
    return resolved, identity, resource_option


class OutlineParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.capture: str | None = None
        self.buffer: list[str] = []
        self.title = ""
        self.h1: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"title", "h1"}:
            self.capture = tag.lower()
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if self.capture is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.capture != tag.lower():
            return
        value = " ".join("".join(self.buffer).split())
        if self.capture == "title":
            self.title = value
        elif value:
            self.h1.append(value)
        self.capture = None
        self.buffer = []


def validate_html(path: Path, expected_page_title: str | None) -> dict[str, object]:
    if path.stat().st_size == 0:
        raise RenderError("Pandoc produced an empty HTML artifact")
    text = path.read_text(encoding="utf-8")
    if "<html" not in text.lower():
        raise RenderError("Pandoc output is not a standalone HTML document")
    if "file://" in text.lower():
        raise RenderError("self-contained HTML contains a local file URI")
    parser = OutlineParser()
    parser.feed(text)
    if expected_page_title is not None and parser.title != expected_page_title:
        raise RenderError(
            f"HTML page title mismatch: expected {expected_page_title!r}, got {parser.title!r}"
        )
    normalized = [heading.casefold() for heading in parser.h1]
    duplicates = sorted(
        {parser.h1[index] for index, value in enumerate(normalized) if normalized.count(value) > 1}
    )
    if duplicates:
        raise RenderError(f"duplicate H1 titles detected: {', '.join(duplicates)}")
    return {
        "standalone_html": "PASS",
        "local_file_uris": "PASS",
        "page_title": parser.title or None,
        "h1_count": len(parser.h1),
        "duplicate_h1_titles": "PASS",
    }


def validate_pdf(path: Path) -> None:
    if path.stat().st_size < 5 or path.read_bytes()[:5] != b"%PDF-":
        raise RenderError("PDF output is missing the PDF signature")


def declared_font_families(css: Path) -> list[str]:
    text = css.read_text(encoding="utf-8")
    families: list[str] = []
    for declaration in re.findall(r"font-family\s*:\s*([^;}]+)", text, flags=re.IGNORECASE):
        for family in declaration.split(","):
            normalized = family.strip().strip("\"'")
            if normalized and normalized not in families:
                families.append(normalized)
    return families


def font_provenance(css: Path, supplied: Sequence[Path]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in supplied:
        resolved = require_regular_file(path, "font file")
        records.append(
            {
                "name": resolved.name,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "declared_families": declared_font_families(css),
        "supplied_files": records,
        "provenance_status": "SUPPLIED_FILES_HASHED" if records else "DECLARATIONS_ONLY",
        "effective_pdf_fonts_verified": False,
        "qa_note": (
            "Supplied files are provenance inputs, not proof that the PDF engine embedded them; "
            "inspect the current PDF font table and rendered pages."
        ),
    }


def inspect_target(target: Path, force: bool) -> os.stat_result | None:
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RenderError(f"cannot inspect output target {target.name}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise RenderError(f"output target must not be a symbolic link: {target.name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise RenderError(f"output target must be a regular file: {target.name}")
    if not force:
        raise RenderError(f"output already exists: {target.name}; pass --force to replace it")
    return metadata


def output_path(target: Path) -> Path:
    absolute = Path(os.path.abspath(target))
    try:
        absolute.parent.mkdir(parents=True, exist_ok=True)
        parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise RenderError(f"cannot prepare output directory for {absolute.name}: {exc}") from exc
    if not parent.is_dir():
        raise RenderError(f"output parent must be a directory: {absolute.name}")
    return parent / absolute.name


def ensure_output_targets(source: Path, targets: Sequence[Path], force: bool) -> list[Path]:
    resolved_targets = [output_path(target) for target in targets]
    if len(set(resolved_targets)) != len(resolved_targets):
        raise RenderError("output paths must be distinct")
    if source in resolved_targets:
        raise RenderError("an output path must not replace the Markdown source")
    for target in resolved_targets:
        inspect_target(target, force)
    return resolved_targets


def copy_to_temporary(source: Path, target: Path, purpose: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.{purpose}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(descriptor, "wb") as output_handle:
            descriptor = -1
            shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        shutil.copystat(source, temporary, follow_symlinks=False)
        return temporary
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def target_matches_snapshot(entry: PublicationEntry) -> bool:
    try:
        current = entry.target.lstat()
    except FileNotFoundError:
        return entry.old_stat is None
    except OSError:
        return False
    if entry.old_stat is None or not stat.S_ISREG(current.st_mode):
        return False
    identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    old_identity = (
        entry.old_stat.st_dev,
        entry.old_stat.st_ino,
        entry.old_stat.st_size,
        entry.old_stat.st_mtime_ns,
    )
    return identity == old_identity and sha256_file(entry.target) == entry.old_sha256


def target_has_snapshotted_bytes(entry: PublicationEntry) -> bool:
    try:
        current = entry.target.lstat()
    except FileNotFoundError:
        return entry.old_stat is None
    except OSError:
        return False
    if entry.old_stat is None or not stat.S_ISREG(current.st_mode):
        return False
    return current.st_size == entry.old_stat.st_size and sha256_file(entry.target) == entry.old_sha256


def cleanup_publication_files(entries: Sequence[PublicationEntry]) -> None:
    for entry in entries:
        for path in (entry.prepared, entry.backup):
            if path is not None:
                path.unlink(missing_ok=True)


def rollback_publication(
    entries: Sequence[PublicationEntry], attempted: Sequence[PublicationEntry]
) -> list[str]:
    restore_failures: dict[Path, OSError] = {}
    for entry in reversed(attempted):
        try:
            if entry.old_stat is None:
                entry.target.unlink(missing_ok=True)
            elif entry.backup is not None:
                os.replace(entry.backup, entry.target)
                entry.backup = None
        except OSError as exc:
            restore_failures[entry.target] = exc
    errors: list[str] = []
    for entry in entries:
        try:
            restored = target_has_snapshotted_bytes(entry)
        except OSError as exc:
            errors.append(f"{entry.target.name}: restore verification failed: {exc}")
            continue
        if not restored:
            detail = restore_failures.get(entry.target)
            suffix = f"; restore failed: {detail}" if detail is not None else ""
            errors.append(
                f"{entry.target.name}: restored bytes do not match the snapshot{suffix}"
            )
    return errors


def publish_transaction(
    staged_targets: Sequence[tuple[Path, Path]], force: bool
) -> None:
    entries: list[PublicationEntry] = []
    attempted: list[PublicationEntry] = []
    try:
        for staged, target in staged_targets:
            staged_file = require_regular_file(staged, "staged publication artifact")
            old_stat = inspect_target(target, force)
            prepared = copy_to_temporary(staged_file, target, "publish")
            entry = PublicationEntry(
                staged=staged_file,
                target=target,
                prepared=prepared,
                backup=None,
                old_sha256=None,
                old_stat=old_stat,
            )
            entries.append(entry)
            if old_stat is not None:
                entry.backup = copy_to_temporary(target, target, "backup")
                entry.old_sha256 = sha256_file(entry.backup)

        changed = [entry.target.name for entry in entries if not target_matches_snapshot(entry)]
        if changed:
            raise RenderError(
                "output target changed while publication was prepared: " + ", ".join(changed)
            )

        try:
            for entry in entries:
                attempted.append(entry)
                os.replace(entry.prepared, entry.target)
        except Exception as exc:
            rollback_errors = rollback_publication(entries, attempted)
            if rollback_errors:
                raise RenderError(
                    "publication failed and rollback could not restore a coherent delivery: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise RenderError(
                f"publication failed; previous delivery was restored: {exc}"
            ) from exc
    finally:
        cleanup_publication_files(entries)


def build_manifest(
    source: Path,
    css: Path,
    staged_html: Path,
    staged_pdf: Path,
    html_name: str,
    pdf_name: str,
    pandoc: dict[str, object],
    pdf_engine: dict[str, object],
    fonts: dict[str, object],
    html_checks: dict[str, object],
    page_title: str | None,
    pdf_backend_assets: list[dict[str, object]],
    pdf_style_mode: str,
    diagnostics: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "evidence_report_delivery_manifest",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {"name": source.name, "bytes": source.stat().st_size, "sha256": sha256_file(source)},
        "style": {"name": css.name, "bytes": css.stat().st_size, "sha256": sha256_file(css)},
        "renderer": {
            "pandoc": pandoc,
            "pdf_engine": pdf_engine,
            "conversion_diagnostics": diagnostics,
        },
        "render_options": {
            "page_title": page_title,
            "visible_title_source": "Markdown body; --title sets HTML pagetitle metadata only",
            "pdf_backend_assets": pdf_backend_assets,
            "pdf_style_mode": pdf_style_mode,
        },
        "fonts": fonts,
        "automated_qa": {
            "html": html_checks,
            "pdf_signature": "PASS",
            "font_provenance": (
                "WARNING_EFFECTIVE_FONTS_NOT_VERIFIED"
                if not fonts.get("effective_pdf_fonts_verified")
                else "PASS"
            ),
            "renderer_diagnostics": (
                "WARNING_REVIEW_RENDERER_DIAGNOSTICS"
                if any(item["review_required"] for item in diagnostics.values())
                else "PASS"
            ),
            "page_level_visual_qa": "NOT_RUN",
            "content_evidence_qa": "NOT_RUN",
        },
        "artifacts": [
            {
                "role": "self_contained_html",
                "name": html_name,
                "bytes": staged_html.stat().st_size,
                "sha256": sha256_file(staged_html),
            },
            {
                "role": "pdf",
                "name": pdf_name,
                "bytes": staged_pdf.stat().st_size,
                "sha256": sha256_file(staged_pdf),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_css = script_dir.parent / "assets" / "report.css"
    default_latex_table_filter = script_dir.parent / "assets" / "fit-latex-tables.lua"
    parser = argparse.ArgumentParser(description="Render Markdown to self-contained HTML and PDF with explicit Pandoc tools.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--html-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--pandoc", default="pandoc", help="Pandoc executable name or path")
    parser.add_argument("--pdf-engine", required=True, help="Pandoc-compatible HTML-to-PDF engine")
    parser.add_argument(
        "--pdf-engine-kind",
        choices=("executable", "wrapper"),
        default="executable",
        help="declare whether --pdf-engine is the engine executable or an explicit wrapper",
    )
    parser.add_argument(
        "--pdf-engine-wrapper-description",
        help="required when --pdf-engine-kind=wrapper; explain what the wrapper launches",
    )
    parser.add_argument("--pdf-engine-option", action="append", default=[], help="Option passed to the selected PDF engine")
    parser.add_argument("--css", type=Path, default=default_css)
    parser.add_argument(
        "--latex-table-filter",
        type=Path,
        default=default_latex_table_filter,
        help="Lua filter used for wrapping widthless tables with a LaTeX PDF engine",
    )
    parser.add_argument(
        "--font-file",
        type=Path,
        action="append",
        default=[],
        help="font provenance file to hash in the manifest; repeatable",
    )
    parser.add_argument("--resource-path", type=Path, action="append", default=[])
    parser.add_argument("--title")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source = require_regular_file(args.source, "Markdown source")
        css = require_regular_file(args.css, "report stylesheet")
        html_output, pdf_output, manifest_output = ensure_output_targets(
            source,
            (args.html_output, args.pdf_output, args.manifest_output),
            args.force,
        )
        resource_roots = [source.parent]
        for item in args.resource_path:
            resolved = item.resolve(strict=True)
            if not resolved.is_dir():
                raise RenderError("each resource path must be a directory")
            resource_roots.append(resolved)

        if args.pdf_engine_kind == "wrapper" and not args.pdf_engine_wrapper_description:
            raise RenderError(
                "--pdf-engine-wrapper-description is required when --pdf-engine-kind=wrapper"
            )
        if args.pdf_engine_kind != "wrapper" and args.pdf_engine_wrapper_description:
            raise RenderError(
                "--pdf-engine-wrapper-description requires --pdf-engine-kind=wrapper"
            )
        pandoc_path, pandoc_provenance, resource_option = pandoc_identity(args.pandoc)
        pdf_engine_path, pdf_engine_provenance = executable_identity(
            args.pdf_engine,
            "PDF engine",
            kind=args.pdf_engine_kind,
            wrapper_description=args.pdf_engine_wrapper_description,
            allow_version_failure=args.pdf_engine_kind == "wrapper",
        )
        # Pandoc validates the engine by executable basename and rejects an
        # absolute pathname even when it resolves to a supported engine. Put
        # the identified executable's lookup directory first, invoke it by the
        # caller-requested basename, and retain the resolved identity in the
        # manifest.
        located_pdf_engine = shutil.which(args.pdf_engine)
        if located_pdf_engine is None:
            raise RenderError(f"cannot resolve PDF engine executable: {args.pdf_engine!r}")
        pdf_engine_environment = os.environ.copy()
        pdf_engine_directory = str(Path(located_pdf_engine).absolute().parent)
        inherited_path = pdf_engine_environment.get("PATH", "")
        pdf_engine_environment["PATH"] = (
            pdf_engine_directory
            if not inherited_path
            else pdf_engine_directory + os.pathsep + inherited_path
        )
        pdf_engine_invocation = Path(args.pdf_engine).name
        pdf_backend_assets: list[dict[str, object]] = []
        latex_table_filter: Path | None = None
        if pdf_engine_invocation.casefold() in LATEX_PDF_ENGINES:
            latex_table_filter = require_regular_file(
                args.latex_table_filter,
                "LaTeX table-width filter",
            )
            pdf_backend_assets.append(
                {
                    "role": "latex_table_width_filter",
                    "name": latex_table_filter.name,
                    "bytes": latex_table_filter.stat().st_size,
                    "sha256": sha256_file(latex_table_filter),
                }
            )
            pdf_style_mode = "pandoc_latex_defaults_with_bundled_table_filter"
        else:
            pdf_style_mode = "html_css_or_engine_specific_verify_in_current_pdf"
        fonts = font_provenance(css, args.font_file)
        with tempfile.TemporaryDirectory(prefix="evidence-report-render-") as directory:
            staging = Path(directory)
            staged_html = staging / "report.html"
            staged_pdf = staging / "report.pdf"
            staged_manifest = staging / "delivery-manifest.json"

            html_command = [
                str(pandoc_path),
                str(source),
                "--from",
                "gfm",
                "--to",
                "html5",
                "--standalone",
                resource_option,
                "--css",
                str(css),
                "--resource-path",
                os.pathsep.join(str(path) for path in resource_roots),
                "--output",
                str(staged_html),
            ]
            if args.title:
                html_command.extend(("--metadata", f"pagetitle={args.title}"))
            html_result = run_command(html_command, "Markdown to HTML conversion")
            html_diagnostics = conversion_diagnostics(
                html_result, "Markdown to HTML conversion"
            )
            html_checks = validate_html(staged_html, args.title)

            pdf_command = [
                str(pandoc_path),
                str(staged_html),
                "--from",
                "html",
                # The standalone HTML needs a <title>, but Pandoc's HTML
                # reader otherwise promotes it to a visible PDF title block.
                # Keep the Markdown H1 as the only visible report title.
                "--metadata",
                "title=",
                "--pdf-engine",
                pdf_engine_invocation,
                "--output",
                str(staged_pdf),
            ]
            for option in args.pdf_engine_option:
                pdf_command.append(f"--pdf-engine-opt={option}")
            if latex_table_filter is not None:
                pdf_command.extend(("--lua-filter", str(latex_table_filter)))
            pdf_result = run_command(
                pdf_command,
                "HTML to PDF conversion",
                environment=pdf_engine_environment,
            )
            pdf_diagnostics = conversion_diagnostics(
                pdf_result, "HTML to PDF conversion"
            )
            validate_pdf(staged_pdf)

            manifest = build_manifest(
                source,
                css,
                staged_html,
                staged_pdf,
                html_output.name,
                pdf_output.name,
                pandoc_provenance,
                pdf_engine_provenance,
                fonts,
                html_checks,
                args.title,
                pdf_backend_assets,
                pdf_style_mode,
                {"html": html_diagnostics, "pdf": pdf_diagnostics},
            )
            staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            publish_transaction(
                (
                    (staged_html, html_output),
                    (staged_pdf, pdf_output),
                    (staged_manifest, manifest_output),
                ),
                args.force,
            )

        print(
            json.dumps(
                {
                    "status": "RENDERED",
                    "html": html_output.name,
                    "pdf": pdf_output.name,
                    "manifest": manifest_output.name,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RenderError) as exc:
        print(f"render_report: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
