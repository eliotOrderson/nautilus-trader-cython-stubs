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

    def _build_type_import_map(self, analyzer) -> dict[str, str]:
        """Build comprehensive type → module mapping from source imports.

        Returns:
            Dictionary mapping type names to their import modules
        """
        type_map = {}

        # Extract from source file imports
        for imp in analyzer.imports:
            if imp.names:
                module = imp.module
                for name in imp.names:
                    # Map each imported name to its module
                    type_map[name] = module

        # Add standard library mappings
        type_map.update(
            {
                "asyncio": "asyncio",
                "date": "datetime",
                "datetime": "datetime",
                "time": "time",
                "pathlib": "pathlib",
                "Path": "pathlib",
            }
        )

        # Add common nautilus_trader type mappings (fallback for types not in imports)
        type_map.update(
            {
                # Identifiers
                "TraderId": "nautilus_trader.model.identifiers",
                "AccountId": "nautilus_trader.model.identifiers",
                "InstrumentId": "nautilus_trader.model.identifiers",
                "ClientOrderId": "nautilus_trader.model.identifiers",
                "VenueOrderId": "nautilus_trader.model.identifiers",
                "PositionId": "nautilus_trader.model.identifiers",
                "StrategyId": "nautilus_trader.model.identifiers",
                "ComponentId": "nautilus_trader.model.identifiers",
                "Symbol": "nautilus_trader.model.identifiers",
                "Venue": "nautilus_trader.model.identifiers",
                "ClientId": "nautilus_trader.model.identifiers",
                "OrderListId": "nautilus_trader.model.identifiers",
                "ExecAlgorithmId": "nautilus_trader.model.identifiers",
                # Common components
                "Logger": "nautilus_trader.common.component",
                "Actor": "nautilus_trader.common.actor",
                "Component": "nautilus_trader.common.component",
                "Clock": "nautilus_trader.common.component",
                "CacheFacade": "nautilus_trader.cache.facade",
                # Account types
                "Account": "nautilus_trader.model.objects",
                "AccountState": "nautilus_trader.model.objects",
                # Order types
                "Order": "nautilus_trader.model.orders.base",
                "OrderList": "nautilus_trader.model.orders.list",
                "MarketOrder": "nautilus_trader.model.orders.market",
                "OrderFilled": "nautilus_trader.model.orders.base",
                "OrderEvent": "nautilus_trader.model.orders.base",
                # Position types
                "Position": "nautilus_trader.model.position",
                "Portfolio": "nautilus_trader.portfolio.portfolio",
                # Price/Quantity types
                "Price": "nautilus_trader.model.objects",
                "Quantity": "nautilus_trader.model.objects",
                "Money": "nautilus_trader.model.objects",
                "Currency": "nautilus_trader.model.objects",
                # Data types
                "QuoteTick": "nautilus_trader.model.data",
                "TradeTick": "nautilus_trader.model.data",
                "Bar": "nautilus_trader.model.data",
                "OrderBook": "nautilus_trader.model.book",
                "DataType": "nautilus_trader.model.data",
                "Instrument": "nautilus_trader.model.instruments.base",
                # Enums and constants
                "OrderSide": "nautilus_trader.model.enums",
                "OrderType": "nautilus_trader.model.enums",
                "PositionSide": "nautilus_trader.model.enums",
                "TimeInForce": "nautilus_trader.model.enums",
                "LiquiditySide": "nautilus_trader.model.enums",
                "TrailingOffsetType": "nautilus_trader.model.enums",
                "OmsType": "nautilus_trader.model.enums",
                "CurrencyType": "nautilus_trader.model.enums",
                "BookType": "nautilus_trader.model.enums",
                "PositionAdjustmentType": "nautilus_trader.model.enums",
                # Rust FFI enum types (from nautilus_trader.core.rust.model)
                "TriggerType": "nautilus_trader.core.rust.model",
                "PriceType": "nautilus_trader.core.rust.model",
                "OrderStatus": "nautilus_trader.core.rust.model",
                "AssetClass": "nautilus_trader.core.rust.model",
                "InstrumentClass": "nautilus_trader.core.rust.model",
                "OptionKind": "nautilus_trader.core.rust.model",
                "ContingencyType": "nautilus_trader.core.rust.model",
                "MarketStatus": "nautilus_trader.core.rust.model",
                "MarketStatusAction": "nautilus_trader.core.rust.model",
                "AggressorSide": "nautilus_trader.core.rust.model",
                "AggregationSource": "nautilus_trader.core.rust.model",
                "AccountType": "nautilus_trader.core.rust.model",
                "RecordFlag": "nautilus_trader.core.rust.model",
                "BookAction": "nautilus_trader.core.rust.model",
                "BarAggregation": "nautilus_trader.core.rust.model",
                "InstrumentCloseType": "nautilus_trader.core.rust.model",
                "MovingAverageType": "nautilus_trader.core.rust.model",
                # Domain types
                "TradingState": "nautilus_trader.trading.state",
                "TimeRangeGenerator": "nautilus_trader.data.generator",
                "StopLimitOrder": "nautilus_trader.model.orders.stop_limit",
                "Data": "nautilus_trader.model.data",
                "RequestData": "nautilus_trader.data.messages",
                "FiniteStateMachine": "nautilus_trader.core.fsm",
                "RedisCacheDatabase": "nautilus_trader.cache.database",
                "OrderManager": "nautilus_trader.execution.manager",
                "DataEngine": "nautilus_trader.data.engine",
                # Execution types
                "Request": "nautilus_trader.execution.messages",
                "Response": "nautilus_trader.execution.messages",
                "Command": "nautilus_trader.execution.messages",
                "BatchCancelOrders": "nautilus_trader.execution.messages",
                # Other common types
                "TimeEvent": "nautilus_trader.common.component",
                "UUID4": "nautilus_trader.core.uuid",
                "Event": "nautilus_trader.core.message",
                "InstrumentStatus": "nautilus_trader.model.data",
                "memoryview": "typing",
            }
        )

        return type_map

    def _collect_imports(self, analyzer) -> list[str]:
        """Collect necessary imports for the stub file."""
        imports_by_module: dict[str, set[str]] = {}

        # Modules to exclude (Cython internal C libraries only)
        EXCLUDED_MODULES = {
            "libc",
            "cpython",
            "libc.stdint",
            "cpython.datetime",
        }

        # Add standard typing imports
        imports_by_module["typing"] = {"Any", "ClassVar"}

        # Add imports extracted from source (filter excluded modules)
        for imp in analyzer.imports:
            module = imp.module

            # Skip Cython internal modules
            if any(
                module.startswith(excl) or module == excl for excl in EXCLUDED_MODULES
            ):
                continue

            if module not in imports_by_module:
                imports_by_module[module] = set()

            if imp.names:
                imports_by_module[module].update(imp.names)

        # Build comprehensive type import map
        type_import_map = self._build_type_import_map(analyzer)

        # Scan for inferred types that might need imports
        inferred_types = set()

        # Scan member variables (existing)
        for cls in analyzer.classes:
            for var in cls.member_variables:
                if var.type_hint:
                    # Extract type names from type hints
                    types = self._extract_type_names(var.type_hint)
                    inferred_types.update(types)

            # Scan method parameters and return types
            for method in cls.methods:
                if method.return_type:
                    types = self._extract_type_names(method.return_type)
                    inferred_types.update(types)
                # Extract types from method parameters
                if method.args:
                    for arg in method.args:
                        # Parse parameter: "name: Type" or "name"
                        if ":" in arg:
                            param_type = arg.split(":", 1)[1].strip()
                            if "=" in param_type:
                                param_type = param_type.split("=", 1)[0].strip()
                            types = self._extract_type_names(param_type)
                            inferred_types.update(types)

            # Scan base classes
            for base in cls.base_classes:
                types = self._extract_type_names(base)
                inferred_types.update(types)

        # Scan function parameters and return types
        for func in analyzer.functions:
            if func.return_type:
                types = self._extract_type_names(func.return_type)
                inferred_types.update(types)
            if func.args:
                for arg in func.args:
                    if ":" in arg:
                        param_type = arg.split(":", 1)[1].strip()
                        if "=" in param_type:
                            param_type = param_type.split("=", 1)[0].strip()
                        types = self._extract_type_names(param_type)
                        inferred_types.update(types)

        # Scan global variable types
        for var in analyzer.global_variables:
            if var.type_hint:
                types = self._extract_type_names(var.type_hint)
                inferred_types.update(types)

        # Add missing imports for inferred types
        for inferred_type in inferred_types:
            if inferred_type in type_import_map:
                module = type_import_map[inferred_type]
                # Skip excluded modules
                if any(
                    module.startswith(excl) or module == excl
                    for excl in EXCLUDED_MODULES
                ):
                    continue
                if module not in imports_by_module:
                    imports_by_module[module] = set()
                imports_by_module[module].add(inferred_type)

        # Build import statements - one import per line
        imports = []
        for module in sorted(imports_by_module.keys()):
            names = sorted(imports_by_module[module])
            if names:
                # One import per line for cleaner output
                for name in names:
                    imports.append(f"from {module} import {name}")

        return imports

    def _extract_type_names(self, type_hint: str) -> set[str]:
        """Extract type names from a type hint string."""
        import re

        # Remove generics, Union, etc.
        # Pattern: extract simple type names
        type_names = set()

        # Split by common separators
        parts = re.split(r"[\[\],|]", type_hint)

        for part in parts:
            part = part.strip()
            # Skip built-in types and special names
            if part and part not in {
                "Any",
                "None",
                "ClassVar",
                "Optional",
                "Union",
                "List",
                "Dict",
                "Set",
                "Tuple",
                "Callable",
                "int",
                "str",
                "float",
                "bool",
                "list",
                "dict",
                "set",
                "tuple",
            }:
                # NEW: Preserve qualified names (module.Type)
                # For 'asyncio.AbstractEventLoop', keep both forms
                if "." in part:
                    # Add the module name for import generation (e.g., 'asyncio')
                    module_name = part.split(".")[0]
                    type_names.add(module_name)
                    # Also add the simple type name as fallback
                    type_name = part.split(".")[-1]
                    type_names.add(type_name)
                else:
                    type_names.add(part)

        return type_names

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
        # Include both public and private members
        for var in cls.member_variables:
            if var.is_instance:
                var_name = var.name.replace("self.", "")
                if var_name:
                    if var.type_hint:
                        lines.append(f"    {var_name}: {var.type_hint}")
                    else:
                        # Fallback to Any for unknown types
                        lines.append(f"    {var_name}: Any")

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
