# lemmascript-python

Python frontend for [LemmaScript](../README.md). It reads fully type-annotated Python with `#@` contract comments, extracts LemmaScript Raw IR, and hands it to `lsc`, which verifies it through the Dafny backend.

```python
def clamp(x: int, lo: int, hi: int) -> int:
    #@ requires lo <= hi
    #@ ensures lo <= result <= hi
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
```

```sh
npx lsc check --backend=dafny examples/clamp.py   # gen + additions-only check + dafny verify
python3 -m lemmascript_python extract clamp.py    # print the Raw IR envelope lsc consumes
```

**Trust assumptions to keep in mind:**

- **`#@ type p nat` on a parameter is a precondition.** The proof covers only `p >= 0`, and the differential harness treats it as `requires p >= 0`.
- **No lone surrogates in strings** (from M4): verified code assumes no string contains a lone surrogate.

Python semantics are modelled as Python, not JavaScript: `//` and `%` floor, ints are unbounded, and truthiness follows Python. Constructs outside the supported subset are rejected with a located error.

**Soundness guard:** `python3 -m lemmascript_python.difftest foo.py` compiles the verified `foo.dfy` to Python with `dafny translate py` and runs it against the original `foo.py` on Hypothesis-generated inputs, edge cases included. It checks four things:

- Both readings of `requires` agree.
- The real code never raises inside the precondition.
- The results are equal.
- Every `ensures` holds on the real result.

Layout: `lemmascript_python/` (extractor; `difftest/` is the harness), `examples/` (verified examples, listed in `LemmaScript-files.txt`), `tests/` (unit, golden `.dfy.gen`, verification and differential tests).
