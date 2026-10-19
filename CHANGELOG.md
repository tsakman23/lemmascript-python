# Changelog (Python frontend fork)

One entry per upstream sync and per milestone.

## Unreleased

- **M1:** Python frontend for `int`/`bool` functions with `#@ requires`/`ensures`, end to end (gen → check → differential).
- **Fork-only engine patches**, carried on each upstream sync and marked `HOOK(external-frontend)` in `tools/src`:
  - external-frontend Raw IR envelope and the `frontends` config key;
  - pre-parsed spec expressions;
  - `floordiv`/`floormod` ops.
- **Upstream sync:** based on LemmaScript v0.6.5 (`ffc07fc`). This is a version bump plus `context7.json`, with no engine changes. It was previously based on v0.6.4 (`4b77a79`).
