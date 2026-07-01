# Agent Instructions

This document outlines the general conventions and guidelines for all AI agents contributing to the `monsab` project. Please read and adhere to these rules before taking action.

## Project Structure & Polyglot Architecture

* All Python library code must reside under `src/monsab/`.
* **Rust Backend:** All performance-critical, zero-overhead routines must be implemented in Rust, located exclusively in the `backend/` directory.
* **The Python/Rust Boundary & Convenience Layer Architecture:**
* The Rust code is compiled into a binary extension module named `monsab._backend`.
* **Direct Exporting:** Performance-critical data structures and classes (e.g., `Permutation`, `MonomialMatrix`, `PcGroup`, `SABTransform`) must be implemented natively in Rust and exposed directly to Python via PyO3. Public submodules (e.g., `src/monsab/core/__init__.py`) should import and re-export these classes directly from `monsab._backend`.
* **No Boilerplate Wrappers:** Do **not** create Python proxy classes or wrapper files whose sole purpose is storing a Rust handle and forwarding method calls. Python should be used strictly as a high-level convenience and orchestration layer (e.g., complex SciPy sparse matrix assembly, algorithm staging, or ecosystem integration).
* **Automated Type Stubs:** A type stub file **`src/monsab/_backend.pyi`** must reflect all exposed Rust definitions. Whenever you modify bindings in `backend/src/lib.rs`, regenerate the stubs automatically by running `uv run maturin develop --bindings pyo3` and format the output with Ruff. Do not attempt to sync stubs manually.


* **Python Submodules:** The library consists of Python submodules which each have their own directory under `src/monsab/` (e.g., `src/monsab/core/`). The respective `__init__.py` files expose the exact public API of that submodule. If an exported class is implemented entirely in Rust, do not create an empty or wrapper `_unit.py` file; re-export it directly in `__init__.py`. Underscore-prefixed Python files (e.g. `_baum_clausen.py`) should only be created for non-trivial Python algorithm implementations and helpers.
* **Imports:**
* Submodule-internal imports: `from ._<unit> import thing`
* Cross-submodule imports: `from monsab.<submodule> import thing`
* Internal backend imports: `from monsab import _backend`



## Developing Safely & Speeding Up Iteration

Because this project compiles a Rust extension via Maturin, unguided changes can cause massive build times, excessive token usage, and cryptic error states. Agents must follow this strict development loop:

1. **Isolate the Logic:** If a bug or feature is in the Rust code, write and run native Rust unit tests first (`uv run cargo test`). **Do not compile the Python extension or run `pytest` until the pure Rust tests pass.**
2. **Minimize Python Boundary Rebuilds:** The Python extension only needs to be recompiled via `uv sync` when you modify the `#[pymodule]` bindings, change signatures crossing the boundary, or are ready to run the Python integration tests.
3. **Keep Stubs Updated:** If you modify a function signature or add a function in `backend/src/lib.rs` that is exposed to Python, regenerate `src/monsab/_backend.pyi` automatically by running `uv run maturin develop --bindings pyo3` and formatting with `uv run ruff format .`.

## Testing and Verification

* The authoritative paths for verifying correctness are the native Cargo test suite and the Python `pytest` suite.
* **Rust Isolation Testing:** Run `uv run cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info` to run native Rust tests and calculate coverage.
* **Python/Integration Testing:** Run `uv run pytest` to execute the Python test suite. To run specific files, use `uv run pytest tests/test_<filename>.py`.
* **Test Alignment:** Every Python unit has exactly one test file associated with it (`tests/test_<submodule>_<unit>.py`). Rust routines should have internal module unit tests or integration tests inside `backend/tests/`.
* We strive for close to 100% coverage across both languages.

## Formatting and Linting

All changes must be completely formatted and linted before declaring a task complete.

| Language | Tool | Command |
| --- | --- | --- |
| **Rust** | `rustfmt` | `uv run cargo fmt --all` |
| **Rust** | `clippy` | `uv run cargo clippy --all-targets --all-features -- -D warnings` |
| **Python** | `ruff` (Lint) | `uv run ruff check . --fix` |
| **Python** | `ruff` (Format) | `uv run ruff format .` |

## Python Version and Typing

* **Target Python 3.14 Semantics**: Write code anticipating Python 3.14. Because annotations are evaluated lazily by default, write annotations directly rather than string-quoting forward references. Do NOT use `from __future__ import annotations`.
* **Modern Generics**: Use modern built-in generic syntax such as `list[int]`, `dict[str, int]`, `tuple[int, ...]`, `set[int]`. Explicitly importing types like `typing.List` is NOT necessary and should be avoided.
* **Type Aliases**: Use the modern `type` statement for aliases (e.g., `type Exponent = tuple[int, ...]`).
* **Abstract Containers**: Use `collections.abc` types in signatures where appropriate (e.g., `Iterable`, `Sequence`, `Mapping`, `Collection`, `Iterator`).
* **Strict Typing**:
* Provide explicit return annotations for all functions and methods.
* Annotate all module-level constants.
* Avoid untyped helper functions or raw `Any` variables.
* Use `typing.Self` for methods returning the instance type.


* The code must be highly type-checker-friendly and written in a style suitable for Pyright/mypy.

## Documentation

* **Docstrings**: Every publicly exported piece of code MUST have a docstring with a brief description of its purpose. Use `pdoc3`-compatible docstrings everywhere.
* **Consistency**: Use plain Markdown in docstrings.
* **Placement**: Module docstrings must begin at the absolute top of the file. Class/function docstrings must follow standard Python placement.
* **Content Requirements**: Public APIs must have full docstrings describing: purpose, parameters, return values, raised exceptions, and examples where helpful.
