#!/usr/bin/env python3
"""
Cython to Python Stub Generator.

Parses .pyx files using Cython's AST and generates .pyi stub files.
"""

import argparse
import sys
from pathlib import Path

from cython_parser import (
    analyze_cython_code,
    ClassInfo,
    FunctionInfo,
    GlobalVariable,
    MemberVariable,
    MethodInfo,
)


class StubGenerator:
    """Generates .pyi stub files from parsed Cython code."""

    def __init__(self, symbol_file: Path | None = None):
        self.symbol_imports: set[str] = set()
        self.symbol_file = symbol_file

    def generate(self, pyx_path: Path) -> str:
        """Generate .pyi content from a .pyx file."""
        code = pyx_path.read_text(encoding="utf-8")
        analyzer = analyze_cython_code(pyx_path.name, code)

        lines: list[str] = []

        # Collect imports needed
        imports = self._collect_imports(analyzer)

        # Add header
        lines.append(self._generate_header(pyx_path))

        # Add imports
        if imports:
            lines.extend(imports)
            lines.append("")

        # Add global variables (include all, even constants and private)
        for var in analyzer.global_variables:
            lines.append(self._generate_global_variable(var))

        # Add functions (include all non-cdef, even private)
        for func in analyzer.functions:
            if not func.is_cdef:
                lines.append(self._generate_function(func))

        # Add classes
        for cls in analyzer.classes:
            lines.append(self._generate_class(cls))
            lines.append("")

        return "\n".join(lines)

    def _collect_imports(self, analyzer) -> list[str]:
        """Collect necessary imports for the stub file."""
        imports = []
        imports.append("from typing import ClassVar")
        return imports

    def _generate_header(self, pyx_path: Path) -> str:
        """Generate stub file header."""
        return f'"""Stub file for {pyx_path.name}."""\n'

    def _generate_global_variable(self, var: GlobalVariable) -> str:
        """Generate a global variable stub."""
        type_hint = f": {var.type_hint}" if var.type_hint else ""
        value = f" = {var.value}" if var.value else ""
        return f"{var.name}{type_hint}{value}"

    def _generate_function(self, func: FunctionInfo) -> str:
        """Generate a function stub."""
        args = ", ".join(func.args) if func.args else ""
        returns = f" -> {func.return_type}" if func.return_type else ""
        body = "    ..."

        lines = []
        if func.docstring:
            lines.append(f"def {func.name}({args}){returns}:")
            lines.append(f'    """{func.docstring}"""')
            lines.append(body)
        else:
            lines.append(f"def {func.name}({args}){returns}:")
            lines.append(body)

        return "\n".join(lines)

    def _generate_class(self, cls: ClassInfo) -> str:
        """Generate a class stub."""
        lines = []

        # Class declaration - always use 'class' in .pyi files
        bases = f"({', '.join(cls.base_classes)})" if cls.base_classes else ""
        lines.append(f"class {cls.name}{bases}:")

        # Docstring
        if cls.docstring:
            lines.append(f'    """{cls.docstring}"""')

        # Member variables (from __init__ assignments)
        for var in cls.member_variables:
            if not var.is_private and var.is_instance:
                var_name = var.name.replace("self.", "")
                if var_name and not var_name.startswith("_"):
                    type_hint = f": {var.type_hint}" if var.type_hint else ""
                    lines.append(f"    {var_name}{type_hint}")

        # Methods (include cpdef as regular methods, skip pure cdef)
        # Include dunder methods (__init__, __eq__, __hash__, __repr__, etc.)
        # cpdef methods are Python-accessible even if private
        # def methods (non-cdef) are all Python-accessible
        for method in cls.methods:
            is_dunder = method.name.startswith("__") and method.name.endswith("__")
            should_include = (
                is_dunder  # Always include dunder methods
                or method.is_cpdef  # cpdef methods are Python-accessible
                or not method.is_cdef  # def methods are Python-accessible
            )

            if should_include:
                method_stub = self._generate_method(method)
                for line in method_stub.split("\n"):
                    lines.append(f"    {line}")

        if len(lines) == 1:
            lines.append("    pass")

        return "\n".join(lines)

    def _generate_member_variable(self, var: MemberVariable) -> str:
        """Generate a member variable stub."""
        type_hint = f": {var.type_hint}" if var.type_hint else ""
        if var.is_class:
            return (
                f"{var.name}: ClassVar[{type_hint}]"
                if type_hint
                else f"{var.name}: ClassVar[Any]"
            )
        return f"{var.name}{type_hint}"

    def _generate_method(self, method: MethodInfo) -> str:
        """Generate a method stub."""
        decorators = []
        if method.is_property:
            decorators.append("@property")
        if method.is_static:
            decorators.append("@staticmethod")
        if method.is_classmethod:
            decorators.append("@classmethod")

        args = ", ".join(method.args) if method.args else ""
        returns = f" -> {method.return_type}" if method.return_type else ""

        lines = []
        for dec in decorators:
            lines.append(dec)

        lines.append(f"def {method.name}({args}){returns}:")
        if method.docstring:
            lines.append(f'    """{method.docstring}"""')
        lines.append("    ...")

        return "\n".join(lines)


def generate_all(source_dir: Path, output_dir: Path) -> tuple[int, int]:
    """Generate stubs for all .pyx files in source directory."""
    generator = StubGenerator()
    generated = 0
    failed = 0

    for pyx_file in sorted(source_dir.rglob("*.pyx")):
        relative = pyx_file.relative_to(source_dir)
        pyi_path = output_dir / (relative.with_suffix(".pyi"))

        try:
            stub_content = generator.generate(pyx_file)
            pyi_path.parent.mkdir(parents=True, exist_ok=True)
            pyi_path.write_text(stub_content, encoding="utf-8")
            print(f"Generated: {pyx_file.name} -> {pyi_path}")
            generated += 1
        except Exception as e:
            print(f"FAILED: {pyx_file}: {e}")
            failed += 1

    return generated, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate .pyi stubs from Cython .pyx files"
    )
    parser.add_argument("pyx_file", type=Path, nargs="?", help="Path to .pyx file")
    parser.add_argument("-o", "--output", type=Path, help="Output .pyi file path")
    parser.add_argument(
        "--all", action="store_true", help="Generate stubs for all .pyx files"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("nautilus_trader/nautilus_trader"),
        help="Source directory with .pyx files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stubs"),
        help="Output directory for .pyi files",
    )
    args = parser.parse_args()

    if args.all:
        generated, failed = generate_all(args.source_dir, args.output_dir)
        print(f"\nGenerated: {generated}, Failed: {failed}")
        sys.exit(0 if failed == 0 else 1)

    if not args.pyx_file:
        parser.error("pyx_file is required unless --all is specified")

    generator = StubGenerator()
    stub_content = generator.generate(args.pyx_file)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(stub_content, encoding="utf-8")
        print(f"Generated: {args.output}")
    else:
        print(stub_content)


if __name__ == "__main__":
    main()
