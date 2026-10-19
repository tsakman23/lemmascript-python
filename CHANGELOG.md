# Changelog (Python frontend fork)

One entry per upstream sync and per milestone.

## Unreleased

- **M2:** lists, `while`/`for` loops with invariants, recursion with `decreases`, comprehensions, `all`/`any`, and quantifiers in specs. The fresh-local rule keeps list aliasing unobservable. Examples: `binary_search`, `array_sum`, `has_duplicate`, `keep_positive`, and `prune_window` from hono-rate-limiter. The differential harness now handles lists, bounded quantifiers and argument mutation.
- **Soundness reviews:** adversarial reviews of M1 and M2 found 15 ways to get a "verified" result that Python contradicts. All are fixed, with regression tests in `python/tests/test_review_m1.py` and `test_review_m2.py`.
- **M1:** Python frontend for `int`/`bool` functions with `#@ requires`/`ensures`, end to end (gen → check → differential).
- **Fork-only engine patches**, carried on each upstream sync and marked `HOOK(external-frontend)` in `tools/src`:
  - external-frontend Raw IR envelope and the `frontends` config key;
  - pre-parsed spec expressions;
  - `floordiv`/`floormod` ops;
  - `equality: "structural"` envelope directive.
- **Upstream sync:** based on LemmaScript v0.6.5 (`ffc07fc`). This is a version bump plus `context7.json`, with no engine changes. It was previously based on v0.6.4 (`4b77a79`).
