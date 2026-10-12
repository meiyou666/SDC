# Figure Integration

Consume publication-ready figure assets; verify their use in the report and final page layout.

## Accept the figure bundle

Prefer a manifest that provides, for each figure:

- stable figure and panel IDs;
- asset path, format, byte count, and digest;
- source-data and figure-spec identities and digests;
- title, caption, units, denominator, and uncertainty definition;
- intended display width and minimum effective text size;
- alt text and accessibility notes;
- figure-level data and visual QA status.

Accept equivalent asset-plus-provenance inputs for a draft. Record missing material fields as findings. Verify file digests before embedding when a manifest is supplied.

## Explain the figure

Place three compact elements near the first substantive use:

1. **How to read it:** axes, units, encodings, panels, reference lines, denominator, normalization, and uncertainty.
2. **What it shows:** visible direction, magnitude, interval, sample support, and comparison.
3. **What it means:** the supported interpretation and its immediate limit.

Keep the caption focused on object, condition, definitions, and provenance. Keep the prose focused on comprehension and inference. Do not claim that visual prominence establishes statistical or causal importance.

## Check accessibility and embedding

Verify:

- every substantive figure has meaningful alt text;
- color is not the only carrier of category or state;
- panel labels, legends, markers, and line styles remain distinguishable;
- the embedded asset preserves aspect ratio and intended dimensions;
- effective text and line sizes remain readable at final placement;
- labels, legends, annotations, and panels are neither overlapped nor clipped;
- captions and figures stay together where practical;
- HTML and PDF show the same figure version and caption.

## Route figure defects

Describe a defect with its figure ID, affected claim, evidence, acceptance condition, and required corrected asset or metadata. Keep the current figure in draft state or isolate the affected claim until a corrected bundle arrives. Do not regenerate plots, transform source data, or change the figure specification within this skill.
