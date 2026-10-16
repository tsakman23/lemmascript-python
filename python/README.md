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

Python semantics are modelled as Python, not JavaScript: `//` and `%` floor, ints are unbounded, and truthiness follows Python. Constructs outside the supported subset are rejected with a located error.

Layout: `lemmascript_python/` (extractor), `examples/` (verified examples, listed in `LemmaScript-files.txt`), `tests/` (unit, golden `.dfy.gen`, verification and differential tests).
