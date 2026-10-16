# Changelog (Python frontend fork)

One entry per upstream sync and per milestone.

## Unreleased

- **M1:** Python frontend for `int`/`bool` functions with `#@ requires`/`ensures`, end to end (gen → check → differential).
- **Engine hooks** (proposed upstream):
  - external-frontend Raw IR envelope and the `frontends` config key;
  - pre-parsed spec expressions;
  - `floordiv`/`floormod` ops.
- **Upstream sync:** based on LemmaScript v0.6.4 (`4b77a79`).
