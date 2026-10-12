#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import render_report as renderer


SCRIPT_DIR = Path(__file__).resolve().parent


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, env=environment)
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def write_fake_pandoc(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import html
import os
import sys
from pathlib import Path

if "--help" in sys.argv:
    if os.environ.get("FAKE_PANDOC_LEGACY"):
        print("--self-contained")
    else:
        print("--embed-resources")
    raise SystemExit(0)
if "--version" in sys.argv:
    print("pandoc 99.0-test")
    raise SystemExit(0)
output = Path(sys.argv[sys.argv.index("--output") + 1])
if os.environ.get("FAKE_PANDOC_WARN"):
    print("simulated renderer diagnostic", file=sys.stderr)
if output.suffix == ".pdf":
    metadata_values = [
        sys.argv[index + 1]
        for index, value in enumerate(sys.argv[:-1])
        if value == "--metadata"
    ]
    if "title=" not in metadata_values:
        print("PDF conversion must suppress the HTML title block", file=sys.stderr)
        raise SystemExit(11)
    engine = sys.argv[sys.argv.index("--pdf-engine") + 1]
    if "/" in engine or "\\\\" in engine:
        print("pdf engine must be passed to Pandoc by basename", file=sys.stderr)
        raise SystemExit(10)
    if os.environ.get("FAKE_PANDOC_FAIL_PDF"):
        print("simulated PDF failure", file=sys.stderr)
        raise SystemExit(9)
    output.write_bytes(b"%PDF-1.4\\n% fake report\\n")
    late_directory = os.environ.get("FAKE_PANDOC_CREATE_DIRECTORY")
    if late_directory:
        Path(late_directory).mkdir()
else:
    metadata = sys.argv[sys.argv.index("--metadata") + 1] if "--metadata" in sys.argv else ""
    if metadata.startswith("title="):
        print("visible title metadata is forbidden", file=sys.stderr)
        raise SystemExit(8)
    title = metadata.removeprefix("pagetitle=") if metadata else ""
    heading = "<h1>Evidence report</h1>"
    if os.environ.get("FAKE_PANDOC_DUPLICATE_TITLE"):
        heading += "<h1>Evidence report</h1>"
    output.write_text(
        "<!doctype html><html><head><title>" + html.escape(title)
        + "</title></head><body>" + heading + "</body></html>\\n",
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def write_fake_engine(path: Path, *, version_ok: bool = True) -> None:
    version_behavior = '    print("fake-pdf-engine 7.1")\n' if version_ok else "    raise SystemExit(3)\n"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        + version_behavior,
        encoding="utf-8",
    )
    path.chmod(0o700)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_no_transaction_files(root: Path) -> None:
    leftovers = [
        path.name
        for path in root.rglob(".*")
        if ".publish." in path.name or ".backup." in path.name
    ]
    assert leftovers == [], leftovers


def exercise_replace_failure(root: Path, fail_at: int, existing: bool) -> None:
    case = root / f"replace-{fail_at}-{'existing' if existing else 'absent'}"
    staging = case / "staging"
    staging.mkdir(parents=True)
    staged = [staging / name for name in ("report.html", "report.pdf", "delivery.json")]
    targets = [case / path.name for path in staged]
    for index, path in enumerate(staged, start=1):
        path.write_bytes(f"new-{index}".encode("ascii"))

    original: dict[Path, bytes] = {}
    if existing:
        for index, target in enumerate(targets, start=1):
            target.write_bytes(f"old-{index}".encode("ascii"))
            original[target] = target.read_bytes()

    real_replace = renderer.os.replace
    replace_count = 0

    def failing_replace(source: os.PathLike[str], target: os.PathLike[str]) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == fail_at:
            raise OSError(f"simulated replace failure {fail_at}")
        real_replace(source, target)

    renderer.os.replace = failing_replace
    try:
        try:
            renderer.publish_transaction(tuple(zip(staged, targets)), force=existing)
        except renderer.RenderError as exc:
            assert "previous delivery was restored" in str(exc)
        else:
            raise AssertionError("simulated publication failure unexpectedly succeeded")
    finally:
        renderer.os.replace = real_replace

    assert replace_count >= fail_at
    if existing:
        assert {target: target.read_bytes() for target in targets} == original
    else:
        assert not any(target.exists() or target.is_symlink() for target in targets)
    assert_no_transaction_files(case)


def render_command(
    root: Path,
    fake_pandoc: Path,
    fake_engine: Path,
    font_file: Path,
    suffix: str = "",
) -> list[str]:
    return [
        "python3",
        str(SCRIPT_DIR / "render_report.py"),
        str(root / "report.md"),
        "--html-output",
        str(root / f"report{suffix}.html"),
        "--pdf-output",
        str(root / f"report{suffix}.pdf"),
        "--manifest-output",
        str(root / f"delivery{suffix}.json"),
        "--pandoc",
        str(fake_pandoc),
        "--pdf-engine",
        str(fake_engine),
        "--font-file",
        str(font_file),
        "--title",
        "Evidence report",
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="revise-evidence-report-test-") as directory:
        root = Path(directory)
        (root / "report.md").write_text(
            "# Evidence report\n\nSupported text.\n", encoding="utf-8"
        )
        fake_pandoc = root / "pandoc"
        write_fake_pandoc(fake_pandoc)
        fake_engine = root / "fake-pdf-engine"
        write_fake_engine(fake_engine)
        font_file = root / "FixtureSans.woff2"
        font_file.write_bytes(b"fixture-font-metadata")

        rendered = run(render_command(root, fake_pandoc, fake_engine, font_file))
        assert json.loads(rendered.stdout)["status"] == "RENDERED"
        html_text = (root / "report.html").read_text(encoding="utf-8")
        assert html_text.startswith("<!doctype html>")
        assert html_text.count("<h1>Evidence report</h1>") == 1
        assert "<title>Evidence report</title>" in html_text
        assert (root / "report.pdf").read_bytes().startswith(b"%PDF-")
        manifest_text = (root / "delivery.json").read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        assert manifest["artifact_type"] == "evidence_report_delivery_manifest"
        assert {item["role"] for item in manifest["artifacts"]} == {
            "self_contained_html",
            "pdf",
        }
        engine_provenance = manifest["renderer"]["pdf_engine"]
        assert engine_provenance["kind"] == "executable"
        assert engine_provenance["requested_name"] == fake_engine.name
        assert engine_provenance["resolution_scope"] == "caller_environment"
        assert "resolved_path" not in engine_provenance
        assert engine_provenance["version"] == "fake-pdf-engine 7.1"
        assert engine_provenance["sha256"] == file_sha256(fake_engine)
        assert manifest["renderer"]["pandoc"]["requested_name"] == fake_pandoc.name
        assert "resolved_path" not in manifest["renderer"]["pandoc"]
        assert manifest["fonts"]["provenance_status"] == "SUPPLIED_FILES_HASHED"
        assert manifest["fonts"]["supplied_files"][0]["sha256"] == file_sha256(font_file)
        assert "resolved_path" not in manifest["fonts"]["supplied_files"][0]
        assert "Noto Serif" in manifest["fonts"]["declared_families"]
        assert manifest["automated_qa"]["html"]["duplicate_h1_titles"] == "PASS"
        assert manifest["automated_qa"]["font_provenance"].startswith("WARNING_")
        assert manifest["automated_qa"]["renderer_diagnostics"] == "PASS"
        assert manifest["renderer"]["conversion_diagnostics"]["pdf"]["stderr_sha256"] is None
        assert all("/" not in item["name"] for item in manifest["artifacts"])

        duplicate = run(
            render_command(root, fake_pandoc, fake_engine, font_file), check=False
        )
        assert duplicate.returncode != 0 and "already exists" in duplicate.stderr
        forced = run(
            render_command(root, fake_pandoc, fake_engine, font_file) + ["--force"]
        )
        assert json.loads(forced.stdout)["status"] == "RENDERED"

        warning_environment = os.environ.copy()
        warning_environment["FAKE_PANDOC_WARN"] = "1"
        warning_render = run(
            render_command(root, fake_pandoc, fake_engine, font_file, "-warning"),
            environment=warning_environment,
        )
        assert "simulated renderer diagnostic" in warning_render.stderr
        warning_manifest = json.loads(
            (root / "delivery-warning.json").read_text(encoding="utf-8")
        )
        warning_diagnostics = warning_manifest["renderer"]["conversion_diagnostics"]
        assert warning_manifest["automated_qa"]["renderer_diagnostics"] == (
            "WARNING_REVIEW_RENDERER_DIAGNOSTICS"
        )
        assert warning_diagnostics["html"]["stderr_line_count"] == 1
        assert warning_diagnostics["pdf"]["stderr_line_count"] == 1
        assert warning_diagnostics["pdf"]["stderr_sha256"]

        latex_engine = root / "xelatex"
        write_fake_engine(latex_engine)
        run(render_command(root, fake_pandoc, latex_engine, font_file, "-latex"))
        latex_manifest = json.loads(
            (root / "delivery-latex.json").read_text(encoding="utf-8")
        )
        backend_assets = latex_manifest["render_options"]["pdf_backend_assets"]
        assert len(backend_assets) == 1
        assert backend_assets[0]["role"] == "latex_table_width_filter"
        assert backend_assets[0]["name"] == "fit-latex-tables.lua"
        assert backend_assets[0]["sha256"] == file_sha256(
            SCRIPT_DIR.parent / "assets" / "fit-latex-tables.lua"
        )

        legacy_environment = os.environ.copy()
        legacy_environment["FAKE_PANDOC_LEGACY"] = "1"
        run(
            render_command(root, fake_pandoc, fake_engine, font_file, "-legacy"),
            environment=legacy_environment,
        )
        assert (root / "report-legacy.html").exists()
        legacy_manifest = json.loads(
            (root / "delivery-legacy.json").read_text(encoding="utf-8")
        )
        assert {item["name"] for item in legacy_manifest["artifacts"]} == {
            "report-legacy.html",
            "report-legacy.pdf",
        }

        duplicate_title_environment = os.environ.copy()
        duplicate_title_environment["FAKE_PANDOC_DUPLICATE_TITLE"] = "1"
        duplicate_title = run(
            render_command(
                root, fake_pandoc, fake_engine, font_file, "-duplicate-title"
            ),
            environment=duplicate_title_environment,
            check=False,
        )
        assert (
            duplicate_title.returncode != 0
            and "duplicate H1 titles" in duplicate_title.stderr
        )
        assert not (root / "report-duplicate-title.html").exists()
        assert not (root / "report-duplicate-title.pdf").exists()

        wrapper = root / "pdf-wrapper"
        write_fake_engine(wrapper, version_ok=False)
        wrapper_command = render_command(
            root, fake_pandoc, wrapper, font_file, "-wrapper"
        )
        wrapper_command.extend(
            [
                "--pdf-engine-kind",
                "wrapper",
                "--pdf-engine-wrapper-description",
                "Fixture wrapper delegates to a pinned PDF service.",
            ]
        )
        run(wrapper_command)
        wrapper_manifest = json.loads(
            (root / "delivery-wrapper.json").read_text(encoding="utf-8")
        )
        wrapper_provenance = wrapper_manifest["renderer"]["pdf_engine"]
        assert wrapper_provenance["kind"] == "wrapper"
        assert wrapper_provenance["version"] is None
        assert wrapper_provenance["version_status"] == "UNAVAILABLE_WRAPPER_PROBE"
        assert "pinned PDF service" in wrapper_provenance["wrapper_description"]

        missing_engine = run(
            render_command(
                root, fake_pandoc, root / "missing-engine", font_file, "-missing-engine"
            ),
            check=False,
        )
        assert (
            missing_engine.returncode != 0
            and "cannot resolve PDF engine" in missing_engine.stderr
        )

        original_html = (root / "report.html").read_bytes()
        original_pdf = (root / "report.pdf").read_bytes()
        original_manifest = (root / "delivery.json").read_bytes()
        failing_environment = os.environ.copy()
        failing_environment["FAKE_PANDOC_FAIL_PDF"] = "1"
        failed = run(
            render_command(root, fake_pandoc, fake_engine, font_file) + ["--force"],
            environment=failing_environment,
            check=False,
        )
        assert failed.returncode != 0 and "simulated PDF failure" in failed.stderr
        assert (root / "report.html").read_bytes() == original_html
        assert (root / "report.pdf").read_bytes() == original_pdf
        assert (root / "delivery.json").read_bytes() == original_manifest

        late_html = root / "report-late-directory.html"
        late_pdf = root / "report-late-directory.pdf"
        late_manifest = root / "delivery-late-directory.json"
        late_html.write_bytes(b"old html")
        late_manifest.write_bytes(b"old manifest")
        late_environment = os.environ.copy()
        late_environment["FAKE_PANDOC_CREATE_DIRECTORY"] = str(late_pdf)
        late_directory = run(
            render_command(
                root, fake_pandoc, fake_engine, font_file, "-late-directory"
            )
            + ["--force"],
            environment=late_environment,
            check=False,
        )
        assert late_directory.returncode != 0
        assert "must be a regular file" in late_directory.stderr
        assert late_pdf.is_dir() and not late_pdf.is_symlink()
        assert late_html.read_bytes() == b"old html"
        assert late_manifest.read_bytes() == b"old manifest"

        symlink_source = root / "symlink-source.pdf"
        symlink_source.write_bytes(b"stable symlink target")
        symlink_output = root / "report-symlink.pdf"
        symlink_output.symlink_to(symlink_source)
        symlink_failure = run(
            render_command(root, fake_pandoc, fake_engine, font_file, "-symlink")
            + ["--force"],
            check=False,
        )
        assert symlink_failure.returncode != 0
        assert "must not be a symbolic link" in symlink_failure.stderr
        assert symlink_output.is_symlink()
        assert symlink_source.read_bytes() == b"stable symlink target"
        assert not (root / "report-symlink.html").exists()
        assert not (root / "delivery-symlink.json").exists()

        exercise_replace_failure(root, fail_at=2, existing=True)
        exercise_replace_failure(root, fail_at=3, existing=True)
        exercise_replace_failure(root, fail_at=2, existing=False)
        assert_no_transaction_files(root)

    print("All render_report offline provenance, QA, and publication rollback tests passed.")


if __name__ == "__main__":
    main()
