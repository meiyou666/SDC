# Evidence Audit and Co-review

Audit at the claim level, then revise at the report level.

## Build a compact evidence map

For each material claim, record enough information to answer:

- What exactly is asserted?
- Which observation or authoritative source supports it?
- Is support direct, reproducibly derived, or unresolved?
- What population, time, system, condition, and measurement boundary applies?
- Which table, figure, calculation, or conclusion depends on it?

Use stable source tags in the report when readers need local traceability. Put full source identity, artifact path or locator, digest, and derivation details in an appendix or evidence map rather than interrupting the prose.

For numerical and statistical claims, verify the underlying numerator, denominator, unit, aggregation, exclusion rule, missing state, non-finite state, uncertainty method, and rounding. Recompute central values when the supplied machine-readable evidence permits it.

## Triage findings

Use the smallest issue record that supports the review. For multi-round work, assign a stable ID and record:

- priority and affected scope;
- current text or observation;
- evidence checked and evidence status;
- proposed action and acceptance condition;
- work state and user decision state.

Use priorities consistently:

- `P0`: unsafe, materially false, or invalidates the central conclusion;
- `P1`: changes an important conclusion, denominator, method, or interpretation;
- `P2`: impairs comprehension, traceability, or comparability;
- `P3`: local language, consistency, or layout improvement.

Do not turn the ledger into a second report. Keep history only when it is needed to understand an active decision or audit trail.

## Co-review efficiently

Present the central findings and a section-level proposal before broad substantive rewriting when user choices matter. Combine independent decisions so the user can review them together. Continue supported edits while a disputed passage remains `UNDER_REVIEW`.

When feedback identifies a class of defect, derive a report-wide rule and search every surface: title, summary, headings, prose, tables, captions, conclusions, contents, appendices, and references. Re-audit affected claims after revision.

## Resolve evidence conflicts

Prefer evidence closest to the observation. Preserve valid observations even when a broader interpretation fails. Narrow the statement to the strongest form supported by all required evidence, and place the limitation next to the affected conclusion.

Separate source facts from cross-source synthesis. Label a synthesis as an interpretation or derived observation and retain plausible alternatives when evidence does not distinguish them.
