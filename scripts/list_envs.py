# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""
Script to print the available environments in this repository.

The script scans task registration files and prints the environment name,
entry point, and config entry point without importing the Isaac Sim runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "source" / "NoetixBumi" / "NoetixBumi" / "tasks"


def _module_name_from_init(file_path: Path) -> str:
    """Convert an ``__init__.py`` path to its dotted module name."""
    relative_path = file_path.relative_to(REPO_ROOT / "source" / "NoetixBumi")
    return ".".join(relative_path.parent.parts)


def _resolve_expr(node: ast.AST, module_name: str, aliases: dict[str, str]) -> str | None:
    """Resolve the small expression subset used in gym.register calls."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.Name):
        if node.id == "__name__":
            return module_name
        return aliases.get(node.id, node.id)

    if isinstance(node, ast.Attribute):
        base = _resolve_expr(node.value, module_name, aliases)
        if base is None:
            return None
        if node.attr == "__name__":
            return base
        return f"{base}.{node.attr}"

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                resolved = _resolve_expr(value.value, module_name, aliases)
                if resolved is None:
                    return None
                parts.append(resolved)
            elif isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)

    return None


def _collect_registers(file_path: Path) -> list[dict[str, str]]:
    """Collect gym.register metadata from one task module."""
    module_name = _module_name_from_init(file_path)
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    aliases: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                continue

            module_parts = module_name.split(".")
            if node.level == 1:
                base_module = module_name
            else:
                base_module = ".".join(module_parts[: -(node.level - 1)])

            imported_module = f"{base_module}.{node.module}" if node.module else base_module
            for alias in node.names:
                alias_name = alias.asname or alias.name
                aliases[alias_name] = f"{imported_module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                alias_name = alias.asname or alias.name
                aliases[alias_name] = alias.name

    registers: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "register":
            continue
        if not isinstance(func.value, ast.Name):
            continue
        if aliases.get(func.value.id, func.value.id) != "gymnasium":
            continue

        task_id = None
        entry_point = None
        config_entry = None

        for keyword in node.keywords:
            if keyword.arg == "id":
                task_id = _resolve_expr(keyword.value, module_name, aliases)
            elif keyword.arg == "entry_point":
                entry_point = _resolve_expr(keyword.value, module_name, aliases)
            elif keyword.arg == "kwargs" and isinstance(keyword.value, ast.Dict):
                for key_node, value_node in zip(keyword.value.keys, keyword.value.values):
                    key = _resolve_expr(key_node, module_name, aliases)
                    if key == "env_cfg_entry_point":
                        config_entry = _resolve_expr(value_node, module_name, aliases)

        if task_id and entry_point and config_entry:
            registers.append(
                {
                    "task_id": task_id,
                    "entry_point": entry_point,
                    "config_entry": config_entry,
                }
            )

    return registers


def collect_available_environments() -> list[dict[str, str]]:
    """Collect all registered Bumi environments from task modules."""
    environments: list[dict[str, str]] = []
    for file_path in sorted(TASKS_ROOT.glob("**/__init__.py")):
        if "__pycache__" in file_path.parts:
            continue
        environments.extend(_collect_registers(file_path))
    return sorted(environments, key=lambda item: item["task_id"])


def _print_table(rows: list[dict[str, str]]) -> None:
    """Print a simple table without third-party dependencies."""
    headers = ["S. No.", "Task Name", "Entry Point", "Config"]
    formatted_rows = [
        [str(index + 1), row["task_id"], row["entry_point"], row["config_entry"]]
        for index, row in enumerate(rows)
    ]
    widths = [len(header) for header in headers]
    for row in formatted_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print("Available Environments in Isaac Lab")
    print(format_row(headers))
    print(separator)
    for row in formatted_rows:
        print(format_row(row))


def main():
    """Print all environments registered in `NoetixBumi` extension."""
    environments = collect_available_environments()
    if not environments:
        raise RuntimeError(f"No task registrations were found under {TASKS_ROOT}")
    _print_table(environments)


if __name__ == "__main__":
    main()
