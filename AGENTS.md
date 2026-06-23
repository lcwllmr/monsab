# Agent Instructions

This document outlines the general conventions and guidelines for all AI agents contributing to the `symred` project. Please read and adhere to these rules before taking action.

## Project Structure
- All library code must reside under `src/monsab/`.
- The library consists of submodules which each have their own directory under `src/monsab/` (e.g. `src/monsab/core/`). The respective `__init__.py` files expose the exact public API of the submodule and all actual code must be placed in separate logical hidden unit files starting with an underscore (e.g. `src/monsab/core/_matrix.py` contains a class `MonomialMatrix` which is exposed in `__init__.py` only).
- All corresponding test files must reside under `tests/` and be named `test_<submodule>_<filename>.py`, matching the module being tested (e.g., `src/monsab/core/my_module.py` -> `tests/test_core_my_module.py`).
- The project layout must remain suitable for installation as a standard Python package from `src/`.

## Testing and Verification
- The authoritative path for verifying correctness is the `pytest` suite. Do not rely on ad hoc standalone scripts as the primary verification path.
- All unit tests must be runnable via `uv run pytest`.
- The standard command to run tests in this `uv`-managed project is `uv run pytest` (or `uv run pytest tests/test_<filename>.py` for specific files).

## Formatting and Linting
- All changes must be formatted and linted before completing a task.
- Run `uv run ruff check` to lint your code and fix any reported issues.
- Run `uv run ruff format` to automatically format your code according to project conventions.

## Python Version and Typing
- **Target Python 3.14 Semantics**: Write code anticipating Python 3.14. Because annotations are evaluated lazily by default, write annotations directly rather than string-quoting forward references. Do NOT use `from __future__ import annotations`.
- **Modern Generics**: Use modern built-in generic syntax such as `list[int]`, `dict[str, int]`, `tuple[int, ...]`, `set[int]`.
- **Type Aliases**: Use the modern `type` statement for aliases (e.g., `type Exponent = tuple[int, ...]`).
- **Abstract Containers**: Use `collections.abc` types in signatures where appropriate (e.g., `Iterable`, `Sequence`, `Mapping`, `Collection`, `Iterator`).
- **Strict Typing**:
  - Provide explicit return annotations for all functions and methods.
  - Annotate all module-level constants.
  - Avoid untyped helper functions.
  - Avoid `Any` unless it is truly unavoidable; prefer precise types.
  - Use `typing.Self` for methods returning the instance type.
- **Dataclasses**: Use `dataclasses` for internal immutable or structured state containers where appropriate.
- The code must be highly type-checker-friendly and written in a style suitable for Pyright/mypy.

## Documentation
- **Docstrings**: Use `pdoc3`-compatible docstrings everywhere.
- **Consistency**: Use one consistent docstring style across the project. Plain Markdown is preferred. If you use a structured style like Google or NumPy, keep it consistent and explicitly set `__docformat__ = "google"` (or `"numpy"`) at the module's top level.
- **Placement**: Module docstrings must begin at the absolute top of the file. Class/function docstrings must follow standard Python placement.
- **Content Requirements**:
  - Public APIs must have full docstrings describing: purpose, parameters, return values, raised exceptions, and examples where helpful.
  - Important internal helpers should also be documented.
  - Include doctest-like or fenced-code examples where useful, but prioritize clarity over verbosity.

