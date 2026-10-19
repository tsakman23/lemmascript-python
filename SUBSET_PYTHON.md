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
| `list[T]` | sequence, with value semantics under the aliasing rule below | `T[]` | ✅ |
| `tuple[A, B]` | tuple (fixed arity) | `[A, B]` | 🚧 M3 |
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
| `in`, `not in` on lists | membership | `.includes` | ✅ (dicts, sets, strings: M4) |
| `xs[i]` | an out-of-range index must be proven impossible. A negative literal (`xs[-1]`) counts from the end; any other index must be proven non-negative | `xs[i]` | ✅ (non-literal negative indices: M3) |
| `xs[a:b]`, `xs[a:]`, `xs[:b]` | Python's clamping slice (no step) | `slice` under `safe-slice` | ✅ |
| bitwise `& \| ^ ~ << >>` | — | `&`, `\|`, `<<`, `>>` | 🚧 later |

**Truthiness** follows Python, and every condition is made explicit:

- `if n:` on an `int` means `n != 0`.
- On lists, `if xs:` means `len(xs) > 0`. This differs from JavaScript, where `[]` is truthy.

## Statements and control flow

| Python | TypeScript counterpart | Status |
|--------|------------------------|--------|
| `x = e`, `x: T = e`, `x += e` (also `-=`, `*=`, `//=`, `%=`) | `let`, `const`, `x += e` | ✅ |
| `if` / `elif` / `else`, `return`, `pass` | `if` / `else if` / `else`, `return` | ✅ |
| `raise …`, `assert c`: modelled as unreachable, so the proof must show they never fire | `throw new Error(…)` | ✅ |
| calls to functions in the same module (positional arguments) | function calls | ✅ |
| `while` with `#@ invariant` / `#@ decreases` | `while` | ✅ |
| `for i in range(a, b)`, `for i, x in enumerate(xs)`, `for x in xs` | `for (let i…)`, `for (const x of xs)` | ✅ (range step: later) |
| `break`, `continue` | same | ✅ |
| recursion with `#@ decreases` | same | ✅ |
| list comprehensions (one `for`, optional `if`), `all`/`any`, `len`, `min`/`max` of a list | `.map`, `.filter`, `.every`, `.some` | ✅ (`sum`: later) |
| `match` on enums and record unions | `switch` on a discriminant | 🚧 M3 |
| imports across modules, `#@ extern` | cross-file calls, `//@ extern` | 🚧 M5 |
| `try`/`except`, `with`, `global`/`nonlocal`, nested functions, classes with methods, generators, `async`, `lambda` outside builtins, default/keyword/`*args` parameters | — | ❌ |

**Scoping.** Python variables are function-scoped, but the model's are block-scoped. So a variable must be first assigned at a level that encloses every use. For example, write `y = 0` before an `if` that sets `y` in each branch.

**Calls inside `and` / `or` / conditional expressions.** The engine may evaluate a call to a loop-bearing function before the short-circuit test, even though Python would skip it. This never makes a proof wrong, but it can make one fail spuriously, with a precondition or termination error for a call Python never makes. Bind the call to a variable inside an `if` instead.

**Must return.** A function with a non-`None` return type must return or raise on every path. Python would silently return `None`, which the model could not represent.

**Aliasing.** Python lists, dicts and sets are shared references, while the model treats them as values. The following keep any sharing invisible:

- Only locals built from fresh values may be modified in place.
- Such a local may not be stored or passed anywhere a second reference could observe the change.
- Parameters are never modified in place. Copy into a new variable first, e.g. `ys = list(xs)`.
- A loop variable over a list of lists refers to the inner list, so it is never modified in place. Nor is a list element that is itself a list: `xs[i] += ys` extends the inner list in place.
- Python iterates the list object it got at loop entry. Two rules keep the model in step:
  - a list being iterated may not be modified in place;
  - a loop over a variable may not reassign that variable.

  A loop over any other expression (`for x in xs[:]`) iterates a snapshot.
- Nested loops may not share a loop variable.

## The annotation layer

Annotations are `#@` comments at the start of a function body, before or after its docstring. Loop annotations go at the start of the loop body.

| Annotation | Meaning | Status |
|------------|---------|--------|
| `#@ requires P`, `#@ ensures P` | precondition and postcondition; `result` is the return value | ✅ |
| `#@ verify` | opt-in mode: once any function has it, only marked functions are verified | ✅ |
| `#@ pure`, `#@ type x nat`, `#@ contract …` | as upstream | ✅ |
| `#@ backend dafny`, `#@ option key value` | file-level, before the first statement | ✅ |
| `#@ invariant P`, `#@ decreases E`, `#@ done_with P` | loops and recursion | ✅ |
| `#@ assert P`, `#@ ghost x: T = E`, `#@ ghost x = E` | statement-level proof hints | ✅ |

**Spec expressions** are Python expressions, with Python semantics, plus a few extensions:

- `A ==> B` (implies) and `A <==> B` (if and only if). Both bind looser than `or` and are right-associative.
- `result`, the return value. It is only allowed in `ensures`, and a parameter named `result` is rejected.
- Calls to functions in the same module. Calls to ghost functions defined only in the `.dfy` proof file pass through to Dafny.
- Quantifiers: `forall(k, P)` / `exists(k, P)` with an optional binder type, `forall(k: int, P)`, and bounded `all(P for k in range(a, b))`.

Specs are parsed with Python's own parser and lowered by the same code as function bodies. So `%`, `//` and truthiness mean the same thing in a contract as in the code.

## Trust assumptions

These assumptions are stated explicitly rather than modelled:

- **Recursion depth.** Python stops deep recursion with `RecursionError` (by default after about 1000 nested calls). The model has no such limit, so a proof about a recursive function holds for inputs within Python's recursion limit.
- **No monkeypatching.** Verified code calls the module's functions as defined in the file. Replacing them at runtime from other code (`mod.f = ...`, `setattr`, `globals()`) is outside the model. Within the file, the frontend rejects every way to rebind them: import aliases, `import *`, local assignments to function or builtin names, and bindings in the `__main__` block.

- `#@ type p nat` on a parameter is a precondition: the proof covers only `p >= 0`.
- From M4, verified code assumes no string contains a lone surrogate.
- `assert` is modelled as always on: it must be proven never to fire. Under `python -O` it is removed, which cannot change a verified result.
