# LemmaScript for Python: the supported subset

This document lists which Python the Python frontend accepts and what each construct means in the verification model. It mirrors [SUBSET.md](SUBSET.md), the TypeScript subset, and gives each row's TypeScript counterpart.

Getting started is in [python/README.md](python/README.md), and runnable examples are in [python/examples](python/examples).

## The idea in one minute

You write ordinary, fully type-annotated Python and add `#@` comments that state what the code must do. The frontend translates the code into LemmaScript's intermediate representation (Raw IR). The existing engine then turns it into Dafny, and Dafny proves the code meets its contract.

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

The function body is plain Python that runs unchanged. The `#@` lines are comments, so Python ignores them at runtime and the prover checks them.

**Why a subset.** Untyped Python is ambiguous: what `a + b`, `x[i]` or `if x:` means depends on runtime types, and classes can redefine operators. The subset removes that ambiguity in three ways:

- **Static types everywhere.** Every parameter, return and local has a static type, so each construct has exactly one meaning.
- **Python's meaning, not JavaScript's.** Each construct lowers to IR that means exactly what the Python means. Where the engine's defaults are JavaScript's, the frontend says so explicitly; for example, `%` lowers to floored modulo and `if n:` becomes `n != 0`.
- **No approximations.** Anything outside the subset is rejected with a `file:line` error, never approximated.

A differential harness backs this up. It runs the original Python against the compiled Dafny model on generated inputs (see [python/README.md](python/README.md)).

**Status legend.**

- ✅ **supported:** the construct has a golden `.dfy.gen` test, passing differential tests and a verified example.
- 🚧 **planned:** the milestone is noted, e.g. `M2`.
- ❌ **excluded:** rejected by design.

## Types

| You write | Modelled as | TypeScript counterpart | Status |
|-----------|-------------|------------------------|--------|
| `int` | unbounded integer: Python ints never overflow, so this is exact | `number` (unbounded only by assumption) / `bigint` | ✅ |
| `bool` | `bool`. Mixing it with `int` arithmetic or `==` (`True + 1`, `1 == True`) is rejected | `boolean` | ✅ |
| `#@ type n nat` | non-negative integer. On a parameter it is a precondition (`n >= 0`) | `//@ type n nat` | ✅ |
| `list[T]` | sequence, with value semantics under the aliasing rule below | `T[]` | 🚧 M2 |
| `tuple[A, B]` | tuple (fixed arity) | `[A, B]` | 🚧 M2 |
| `@dataclass(frozen=True)`, `NamedTuple`, `TypedDict` | record datatype | `type R = { … }` | 🚧 M3 |
| `Literal["a", "b"]`, `Enum` | enum datatype | string-literal union | 🚧 M3 |
| union of frozen dataclasses | sum datatype, dispatched by class | discriminated union | 🚧 M3 |
| `Optional[T]` / `T \| None` | option | `T \| undefined` | 🚧 M3 |
| `dict[K, V]`, `set[T]` | map, set. Iteration order is treated as arbitrary, which is sound | `Map`, `Set` | 🚧 M4 |
| `str` | string of Unicode scalar values (no lone surrogates) | `string` | 🚧 M4 |
| `float`, `complex`, `Any`, `object`, `bytes`, `TypeVar`, `Protocol` | — | — | ❌ (`float` rounds, `Any` is untyped) |

## Operators

| Python | Meaning in the model | TypeScript counterpart | Status |
|--------|----------------------|------------------------|--------|
| `+ - *`, unary `-`/`+`, `< <= > >= == !=` | exact integer arithmetic and comparison | same | ✅ |
| `a // b` | floor division (rounds toward −∞); `b == 0` must be proven impossible | `Math.floor(a / b)` | ✅ |
| `a % b` | floor modulo (the result has the sign of `b`); `b == 0` must be proven impossible | — (JS `%` truncates) | ✅ |
| `a ** k`, literal `k` from 0 to 4 | repeated multiplication | — | ✅ |
| `a < b <= c` | `a < b and b <= c`. The middle operand must be a name or literal | — | ✅ |
| `and`, `or`, `not`, `x if c else y` | boolean logic with Python truthiness. `and`/`or` in a value position need `bool` operands | `&&`, `\|\|`, `!`, `?:` | ✅ |
| `abs`, `min`, `max`, `bool(x)`, `int(x)` on int | exact | `Math.abs/min/max` | ✅ |
| `a / b` | — | `/` (real division) | ❌ (produces a float; use `//`) |
| `is`, `is not` | only against `None` | `=== undefined` | 🚧 M3 |
| `in`, `not in` | membership | `.includes`, `.has`, `in` | 🚧 M2/M4 |
| `xs[i]`, negative `i` | index from the end; out of range must be proven impossible | `xs[i]` (no negative index) | 🚧 M2/M3 |
| `xs[a:b]` | Python's clamping slice | `slice` under `safe-slice` | 🚧 M2 |
| bitwise `& \| ^ ~ << >>` | — | `&`, `\|`, `<<`, `>>` | 🚧 later |

**Truthiness** follows Python, and every condition is made explicit:

- `if n:` on an `int` means `n != 0`.
- On containers, `if xs:` means `len(xs) > 0` (M2). This differs from JavaScript, where `[]` is truthy.

## Statements and control flow

| Python | TypeScript counterpart | Status |
|--------|------------------------|--------|
| `x = e`, `x: T = e`, `x += e` (also `-=`, `*=`, `//=`, `%=`) | `let`, `const`, `x += e` | ✅ |
| `if` / `elif` / `else`, `return`, `pass` | `if` / `else if` / `else`, `return` | ✅ |
| `raise …`, `assert c`: modelled as unreachable, so the proof must show they never fire | `throw new Error(…)` | ✅ |
| calls to functions in the same module (positional arguments) | function calls | ✅ |
| `while` with `#@ invariant` / `#@ decreases` | `while` | 🚧 M2 |
| `for i in range(a, b)`, `for x in xs` | `for (let i…)`, `for (const x of xs)` | 🚧 M2 |
| `break`, `continue` | same | 🚧 M2 |
| recursion with `#@ decreases` | same | 🚧 M2 |
| comprehensions, `all`/`any`/`sum`/`len` | `.map`, `.filter`, `.every`, `.some` | 🚧 M2 |
| `match` on enums and record unions | `switch` on a discriminant | 🚧 M3 |
| imports across modules, `#@ extern` | cross-file calls, `//@ extern` | 🚧 M5 |
| `try`/`except`, `with`, `global`/`nonlocal`, nested functions, classes with methods, generators, `async`, `lambda` outside builtins, default/keyword/`*args` parameters | — | ❌ |

**Scoping.** Python variables are function-scoped, but the model's are block-scoped. So a variable must be first assigned at a level that encloses every use. For example, write `y = 0` before an `if` that sets `y` in each branch.

**Must return.** A function with a non-`None` return type must return or raise on every path. Python would silently return `None`, which the model could not represent.

**Aliasing (from M2).** Python lists, dicts and sets are shared references, while the model treats them as values. The following keep any sharing invisible:

- Only locals built from fresh values may be modified in place.
- Such a local may not be stored or passed anywhere a second reference could observe the change.
- Parameters are never modified in place. Copy first, e.g. `xs = list(xs)`.

## The annotation layer

Annotations are `#@` comments at the start of a function body, before or after its docstring. Loop annotations go at the start of the loop body.

| Annotation | Meaning | Status |
|------------|---------|--------|
| `#@ requires P`, `#@ ensures P` | precondition and postcondition; `result` is the return value | ✅ |
| `#@ verify` | opt-in mode: once any function has it, only marked functions are verified | ✅ |
| `#@ pure`, `#@ type x nat`, `#@ contract …` | as upstream | ✅ |
| `#@ backend dafny`, `#@ option key value` | file-level, before the first statement | ✅ |
| `#@ invariant P`, `#@ decreases E`, `#@ done_with P` | loops and recursion | 🚧 M2 |
| `#@ assert P`, `#@ ghost x = E` | statement-level proof hints | 🚧 M2 |

**Spec expressions** are Python expressions, with Python semantics, plus a few extensions:

- `A ==> B` (implies) and `A <==> B` (if and only if). Both bind looser than `or` and are right-associative.
- `result`, the return value. It is only allowed in `ensures`, and a parameter named `result` is rejected.
- Calls to functions in the same module. Calls to ghost functions defined only in the `.dfy` proof file pass through to Dafny.
- Quantifiers (M2): `forall(k, P)` / `exists(k, P)` with an optional binder type, `forall(k: int, P)`, and bounded `all(P for k in range(a, b))`.

Specs are parsed with Python's own parser and lowered by the same code as function bodies. So `%`, `//` and truthiness mean the same thing in a contract as in the code.

## Trust assumptions

These assumptions are stated explicitly rather than modelled:

- `#@ type p nat` on a parameter is a precondition: the proof covers only `p >= 0`.
- From M4, verified code assumes no string contains a lone surrogate.
- `assert` is modelled as always on: it must be proven never to fire. Under `python -O` it is removed, which cannot change a verified result.
