#!/usr/bin/env python3

import traceback
from dataclasses import dataclass
from dataclasses import field

from Cython.Compiler import Errors
from Cython.Compiler import PyrexTypes
from Cython.Compiler.Main import CompilationOptions
from Cython.Compiler.Main import Context
from Cython.Compiler.Main import default_options
from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Visitor import ScopeTrackingTransform


@dataclass
class MemberVariable:
    name: str
    type_hint: str | None = None
    is_private: bool = False
    default_value: str | None = None
    is_public: bool = False
    is_readonly: bool = False
    is_class: bool = False
    is_instance: bool = False
    line_number: int | None = None


@dataclass
class MethodInfo:
    name: str
    args: list[str] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    is_private: bool = False
    is_static: bool = False
    is_classmethod: bool = False
    is_cdef: bool = False
    is_cpdef: bool = False
    is_property: bool = False
    line_number: int | None = None


@dataclass
class FunctionInfo:
    name: str
    args: list[str] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    is_cdef: bool = False
    is_cpdef: bool = False
    line_number: int | None = None


@dataclass
class GlobalVariable:
    name: str
    type_hint: str | None = None
    value: str | None = None
    is_constant: bool = False
    line_number: int | None = None


@dataclass
class ImportInfo:
    """Stores import statement information."""

    module: str
    names: list[str] = field(default_factory=list)  # For 'from X import A, B'
    is_cimport: bool = False
    alias: str | None = None
    line_number: int | None = None


@dataclass
class TypeRegistry:
    """Registry for tracking member variable types from multiple sources."""

    # member_name -> {method_name: type_hint}
    type_sources: dict[str, dict[str, str]] = field(default_factory=dict)
    # member_name -> has_none_assignment
    has_none: set[str] = field(default_factory=set)
    # private member types (for self._xxx references)
    private_member_types: dict[str, str] = field(default_factory=dict)

    def record_type(self, member: str, type_hint: str, method_name: str):
        """Record a type hint from a specific method."""
        if member not in self.type_sources:
            self.type_sources[member] = {}
        self.type_sources[member][method_name] = type_hint

    def record_none_assignment(self, member: str):
        """Record that a member is assigned None."""
        self.has_none.add(member)

    def record_private_member_type(self, member: str, type_hint: str):
        """Record type for private member (self._xxx)."""
        self.private_member_types[member] = type_hint

    def get_final_type(self, member: str) -> str | None:
        """Get the final merged type for a member."""
        if member not in self.type_sources:
            if member in self.has_none:
                return "Any | None"  # Fallback
            return None

        # Get the most specific type (prefer non-Any types)
        types = list(self.type_sources[member].values())
        best_type = None
        for t in types:
            if t != "Any":
                best_type = t
                break
        if best_type is None:
            best_type = types[0] if types else "Any"

        # Add | None if needed
        if member in self.has_none:
            if "| None" not in best_type and "None" not in best_type:
                best_type = f"{best_type} | None"

        return best_type


@dataclass
class ClassInfo:
    name: str
    base_classes: list[str] = field(default_factory=list)
    docstring: str | None = None
    member_variables: list[MemberVariable] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)
    is_cdef_class: bool = False
    is_extension_type: bool = False
    line_number: int | None = None


class CythonCodeAnalyzer(ScopeTrackingTransform):
    """
    Analyzes Cython code and extracts information about classes, functions, and variables.
    """

    def __init__(self, context):
        super().__init__(context=context)
        self.imports: list[ImportInfo] = []
        self.classes: list[ClassInfo] = []
        self.functions: list[FunctionInfo] = []
        self.global_variables: list[GlobalVariable] = []
        self.current_class: ClassInfo | None = None
        self.current_function: FunctionInfo | MethodInfo | None = None
        self.class_stack: list[ClassInfo] = []
        self.type_registry = TypeRegistry()
        self.current_method_name: str | None = None

    def visit_ModuleNode(self, node):
        self.visitchildren(node)
        return node

    def visit_ImportNode(self, node):
        """Handle 'import X' and 'from X import A, B' statements."""
        # Extract module name
        module_name = None
        if hasattr(node, "module_name"):
            mod_node = node.module_name
            if hasattr(mod_node, "value"):
                module_name = str(mod_node.value)

        # Extract imported names (for 'from X import A, B')
        imported_names = []
        if hasattr(node, "imported_names") and node.imported_names:
            imported_names = [
                str(name.value) if hasattr(name, "value") else str(name)
                for name in node.imported_names
            ]

        if module_name:
            import_info = ImportInfo(
                module=module_name,
                names=imported_names,
                is_cimport=False,
                line_number=node.pos[1] if hasattr(node, "pos") else None,
            )
            self.imports.append(import_info)

        self.visitchildren(node)
        return node

    def visit_FromCImportStatNode(self, node):
        """Handle 'from X cimport A, B' statements."""
        # Extract module name
        module_name = None
        if hasattr(node, "module_name"):
            module_name = str(node.module_name)

        # Extract imported names from tuple format: ((pos, 'Name', alias), ...)
        imported_names = []
        if hasattr(node, "imported_names") and node.imported_names:
            for item in node.imported_names:
                if isinstance(item, tuple) and len(item) >= 2:
                    name = item[1]
                    imported_names.append(str(name))

        if module_name:
            import_info = ImportInfo(
                module=module_name,
                names=imported_names,
                is_cimport=True,
                line_number=node.pos[1] if hasattr(node, "pos") else None,
            )
            self.imports.append(import_info)

        self.visitchildren(node)
        return node

    def visit_CClassDefNode(self, node):
        return self._visit_class_node(node, is_cdef_class=True, is_extension_type=True)

    def visit_PyClassDefNode(self, node):
        return self._visit_class_node(
            node, is_cdef_class=False, is_extension_type=False
        )

    def _visit_class_node(
        self, node, is_cdef_class: bool = False, is_extension_type: bool = False
    ):
        base_classes = []
        if hasattr(node, "bases") and node.bases:
            base_classes = [
                name
                for arg in node.bases.args
                if (name := self._extract_name_from_node(arg))
            ]

        docstring = self._extract_doc_from_node(node)
        class_name = getattr(node, "class_name", None) or getattr(
            node, "name", "Unknown"
        )

        class_info = ClassInfo(
            name=class_name,
            base_classes=base_classes,
            docstring=docstring,
            is_cdef_class=is_cdef_class,
            is_extension_type=is_extension_type,
            line_number=node.pos[1],
        )

        self.class_stack.append(class_info)
        self.current_class = class_info

        self.visitchildren(node)

        self.class_stack.pop()
        self.current_class = self.class_stack[-1] if self.class_stack else None

        self.classes.append(class_info)
        return node

    def visit_CFuncDefNode(self, node):
        return self._visit_function_node(
            node, is_cdef=not node.overridable, is_cpdef=node.overridable
        )

    def visit_DefNode(self, node):
        return self._visit_function_node(node)

    def _visit_function_node(self, node, is_cdef: bool = False, is_cpdef: bool = False):
        if (
            self.current_function
        ):  # If we are already in a function, we should not process this node
            self.visitchildren(node)
            self.current_function = None
            return node

        is_cfunc = is_cdef or is_cpdef
        function_name = node.name if not is_cfunc else node.declarator.base.name

        args = self._extract_function_args(node, is_cfunc)
        return_type = self._extract_return_type(node, is_cfunc)
        decorators = self._analyze_decorators(node)
        docstring = self._extract_doc_from_node(node)

        is_private = (
            function_name.startswith("_") and not function_name.startswith("__")
        ) or function_name.endswith("__")

        # Store method name for assignment tracking
        if self.current_class:
            self.current_method_name = function_name
            method_info = MethodInfo(
                name=function_name,
                args=args,
                return_type=return_type,
                docstring=docstring,
                is_private=is_private,
                is_static=decorators.get("staticmethod", False),
                is_classmethod=decorators.get("classmethod", False),
                is_property=decorators.get("property", False),
                is_cdef=is_cdef,
                is_cpdef=is_cpdef,
                line_number=node.pos[1],
            )
            self.current_class.methods.append(method_info)
            self.current_function = method_info
        else:
            self.current_method_name = None
            function_info = FunctionInfo(
                name=function_name,
                args=args,
                return_type=return_type,
                docstring=docstring,
                is_cdef=is_cdef,
                is_cpdef=is_cpdef,
                line_number=node.pos[1],
            )
            self.functions.append(function_info)
            self.current_function = function_info

        self.visitchildren(node)
        self.current_function = None
        self.current_method_name = None
        return node

    def visit_SingleAssignmentNode(self, node):
        if (
            hasattr(node, "lhs")
            and hasattr(node, "rhs")
            and not hasattr(node.rhs, "module_name")
        ):
            var_name = self._extract_name_from_node(node.lhs)
            if var_name:
                type_hint = None

                # Track member variable assignments for type inference
                if var_name.startswith("self."):
                    member_name = var_name.replace("self.", "")

                    # Check if RHS is None (NoneNode in Cython AST)
                    rhs_class_name = node.rhs.__class__.__name__
                    if rhs_class_name == "NoneNode":
                        self.type_registry.record_none_assignment(member_name)

                    # Check if RHS is a parameter reference
                    elif hasattr(node.rhs, "name") and self.current_method_name:
                        rhs_name = node.rhs.name

                        # Handle private member references (self.xxx = self._yyy)
                        if rhs_name.startswith("_"):
                            # Look up type from private member registry
                            if rhs_name in self.type_registry.private_member_types:
                                inferred_type = self.type_registry.private_member_types[
                                    rhs_name
                                ]
                                self.type_registry.record_type(
                                    member_name, inferred_type, self.current_method_name
                                )
                        else:
                            # Regular parameter reference
                            param_type = self._get_param_type(rhs_name)
                            if param_type:
                                self.type_registry.record_type(
                                    member_name, param_type, self.current_method_name
                                )

                    # Check if RHS is a constructor call (self.xxx = ClassName(...))
                    elif hasattr(node.rhs, "function") and self.current_method_name:
                        constructor_type = self._extract_constructor_type(node.rhs)
                        if constructor_type:
                            self.type_registry.record_type(
                                member_name, constructor_type, self.current_method_name
                            )

                    # NEW: Handle property access chains (self.x = obj.property)
                    # Check if RHS is an attribute access (has 'obj' and 'attribute' fields)
                    elif (
                        hasattr(node.rhs, "obj")
                        and hasattr(node.rhs, "attribute")
                        and self.current_method_name
                    ):
                        # Get the object being accessed (e.g., 'event' from 'event.account_id')
                        obj_name = self._extract_name_from_node(node.rhs.obj)
                        if obj_name:
                            # Get the object's type from parameters
                            obj_type = self._get_param_type(obj_name)
                            if obj_type:
                                # Get the property name being accessed
                                prop_name = node.rhs.attribute
                                # Look up the property type from the object's class
                                prop_type = self._lookup_property_type(
                                    obj_type, prop_name
                                )
                                if prop_type:
                                    self.type_registry.record_type(
                                        member_name, prop_type, self.current_method_name
                                    )

                    # NEW: Try to infer type from expression patterns
                    # This handles: type(self).__name__, comparisons, len(), etc.
                    elif self.current_method_name:
                        expr_type = self._infer_type_from_expression(node.rhs)
                        if expr_type:
                            self.type_registry.record_type(
                                member_name, expr_type, self.current_method_name
                            )

                    # Track private member types for future reference
                    if member_name.startswith("_"):
                        if hasattr(node.lhs, "annotation") and node.lhs.annotation:
                            private_type = self._extract_type_from_node(
                                node.lhs.annotation
                            )
                            if private_type:
                                self.type_registry.record_private_member_type(
                                    member_name, private_type
                                )

                # Try to extract type from annotation
                if hasattr(node.lhs, "annotation") and node.lhs.annotation:
                    type_hint = self._extract_type_from_node(node.lhs.annotation)

                # If no annotation, try to infer type from parameter
                if (
                    not type_hint
                    and self.current_function
                    and var_name.startswith("self.")
                ):
                    # Extract the actual member name (remove "self.")
                    member_name = var_name.replace("self.", "")

                    # Check if RHS is a parameter reference
                    rhs_name = None
                    if hasattr(node.rhs, "name"):
                        rhs_name = node.rhs.name

                    # If RHS is a parameter, get its type
                    if rhs_name:
                        param_type = self._get_param_type(rhs_name)
                        if param_type:
                            type_hint = param_type

                    # If RHS is None, add "| None" to type
                    rhs_class_name = node.rhs.__class__.__name__
                    if rhs_class_name == "NoneNode":
                        if type_hint:
                            if "| None" not in type_hint and "None" not in type_hint:
                                type_hint = f"{type_hint} | None"
                        else:
                            type_hint = "Any | None"

                value = self._extract_value_from_node(node.rhs)

                self._add_variable(
                    var_name,
                    type_hint,
                    value,
                    is_class=self.current_class is not None,
                    is_instance=self.current_function is not None,
                    line_number=node.pos[1],
                )

        self.visitchildren(node)
        return node

    def _get_param_type(self, param_name: str) -> str | None:
        """Get the type annotation of a function/method parameter."""
        if not self.current_function or not hasattr(self.current_function, "args"):
            return None

        for arg in self.current_function.args:
            # Parse argument format: "name: type" or "name: type = default"
            if ":" in arg:
                parts = arg.split(":", 1)
                name = parts[0].strip()
                type_part = parts[1].strip()

                # Remove default value if present
                if "=" in type_part:
                    type_part = type_part.split("=", 1)[0].strip()

                if name == param_name:
                    return type_part

        return None

    def _extract_constructor_type(self, node) -> str | None:
        """Extract type from constructor call like 'GreeksCalculator(...)'."""
        if hasattr(node, "function"):
            if hasattr(node.function, "name"):
                # The class name is the return type
                return node.function.name
            elif hasattr(node.function, "attribute"):
                # Handle module.ClassName() calls
                return node.function.attribute
        return None

    def _infer_type_from_imports(self, member_name: str) -> str | None:
        """Infer type from import statements based on naming patterns."""
        # Common naming patterns - even if not imported, these are standard types
        name_to_type_map = {
            # Existing patterns
            "trader_id": "TraderId",
            "msgbus": "MessageBus",
            "cache": "CacheFacade",
            "clock": "Clock",
            "log": "Logger",
            "logger": "Logger",
            "portfolio": "PortfolioFacade",
            # NEW: Common identifier patterns
            "account_id": "AccountId",
            "instrument_id": "InstrumentId",
            "client_order_id": "ClientOrderId",
            "venue_order_id": "VenueOrderId",
            "order_list_id": "OrderListId",
            "position_id": "PositionId",
            "strategy_id": "StrategyId",
            "component_id": "ComponentId",
            "actor_id": "ActorId",
            # Common data structures
            "order": "Order",
            "order_list": "OrderList",
            "position": "Position",
            "account": "Account",
            "instrument": "Instrument",
            "portfolio": "Portfolio",
            # Common config types
            "config": "Config",
            "order_config": "OrderConfig",
            # Common execution types
            "exec_report": "ExecutionReport",
            "exec_command": "ExecutionCommand",
            # Common data types
            "bar": "Bar",
            "quote": "QuoteTick",
            "trade": "TradeTick",
            "order_book": "OrderBook",
            "book_order": "BookOrder",
            # Common identifiers (cont.)
            "symbol": "Symbol",
            "venue": "Venue",
            "instrument_id": "InstrumentId",
            # Common enum/status types
            "side": "OrderSide",
            "status": "OrderStatus",
            "type": "OrderType",
            "time_in_force": "TimeInForce",
            # Common numeric types
            "price": "Price",
            "quantity": "Quantity",
            "amount": "Money",
        }

        if member_name in name_to_type_map:
            suggested_type = name_to_type_map[member_name]
            # Verify this type is imported
            for imp in self.imports:
                if suggested_type in imp.names or suggested_type == imp.names:
                    return suggested_type
                # Also check if the type might be in the module path
                if imp.module and suggested_type in imp.module:
                    return suggested_type

            # If not found in imports, still return the type (it will need to be added to imports)
            # This helps generate better stubs even if imports are incomplete
            return suggested_type

        return None

    def _infer_type_from_default(self, default_value: str | None) -> str | None:
        """Infer type from default value literals.

        Args:
            default_value: String representation of the default value

        Returns:
            Inferred type hint if determinable, None otherwise
        """
        if not default_value:
            return None

        # Strip whitespace
        val = default_value.strip()

        # Boolean literals
        if val == "True" or val == "False":
            return "bool"

        # None
        if val == "None":
            return "None"

        # Integer literals (check for no decimal point)
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            return "int"

        # Float literals (contains decimal point)
        if "." in val and not val.startswith('"') and not val.startswith("'"):
            try:
                float(val)
                return "float"
            except ValueError:
                pass

        # String literals
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            return "str"

        # List literals
        if val.startswith("[") and val.endswith("]"):
            return "list"

        # Dict literals
        if val.startswith("{") and val.endswith("}"):
            return "dict"

        # Set literals
        if val.startswith("{") and "}" in val and ":" not in val:
            return "set"

        # Tuple literals
        if val.startswith("(") and val.endswith(")"):
            return "tuple"

        # Common patterns for config attributes
        # Pattern: use_xxx, is_xxx, has_xxx often boolean
        if val in ["True", "False"]:
            return "bool"

        return None

    def _infer_type_from_expression(self, node) -> str | None:
        """Infer type from common expression patterns.

        Handles:
        - type(self).__name__ → str
        - Comparison operators (>, <, ==, etc.) → bool
        - len() calls → int
        - Boolean operators (and, or) → bool
        """
        node_class = node.__class__.__name__

        # Pattern 1: String operations (type(self).__name__, str(x), etc.)
        if node_class == "AttributeNode":
            if hasattr(node, "attribute"):
                attr_name = node.attribute
                # type(self).__name__ -> str
                if attr_name == "__name__":
                    return "str"
                # .name, .id, .value often return strings
                if attr_name in ["name", "id", "value", "code", "symbol"]:
                    return "str"

        # Pattern 2: Comparison operators (x > y, x == y, etc.)
        if node_class in ["ComparisonNode", "PrimaryCmpNode"]:
            return "bool"

        # Pattern 3: Function calls with known return types
        if node_class == "CallNode":
            if hasattr(node, "function"):
                func_name = None
                if hasattr(node.function, "name"):
                    func_name = node.function.name
                elif hasattr(node.function, "attribute"):
                    func_name = node.function.attribute

                if func_name:
                    # len() -> int
                    if func_name == "len":
                        return "int"
                    # str() -> str
                    if func_name == "str":
                        return "str"
                    # int() -> int
                    if func_name == "int":
                        return "int"
                    # float() -> float
                    if func_name == "float":
                        return "float"
                    # bool() -> bool
                    if func_name == "bool":
                        return "bool"
                    # abs(), max(), min() -> numeric
                    if func_name in ["abs", "max", "min", "round"]:
                        return "float"

        # Pattern 4: Binary operations
        if node_class == "BinOpNode":
            # String concatenation, arithmetic, etc.
            # Default to checking operator
            if hasattr(node, "operator"):
                op = node.operator
                # Comparison-like operators in binary context
                if op in ["==", "!=", "<", ">", "<=", ">="]:
                    return "bool"
                # Arithmetic operators
                if op in ["+", "-", "*", "/", "//", "%"]:
                    return "int | float"  # Could be either

        return None

    def _infer_param_type_from_name(
        self, param_name: str, default_val: str | None = None
    ) -> str | None:
        """Infer parameter type from name patterns.

        Args:
            param_name: Name of the parameter
            default_val: Default value if any

        Returns:
            Inferred type hint if determinable, None otherwise
        """
        if not param_name:
            return None

        # Common parameter name patterns
        param_type_map = {
            # Identifiers
            "instrument_id": "InstrumentId",
            "account_id": "AccountId",
            "strategy_id": "StrategyId",
            "client_order_id": "ClientOrderId",
            "venue_order_id": "VenueOrderId",
            "position_id": "PositionId",
            "order_list_id": "OrderListId",
            "trader_id": "TraderId",
            # Types
            "price_type": "PriceType",
            "order_type": "OrderType",
            "order_side": "OrderSide",
            "time_in_force": "TimeInForce",
            "trigger_type": "TriggerType",
            "contingency_type": "ContingencyType",
            "aggregation_source": "AggregationSource",
            # Objects
            "price": "Price",
            "quantity": "Quantity",
            "instrument": "Instrument",
            "order": "Order",
            "position": "Position",
            "account": "Account",
            "portfolio": "Portfolio",
            "clock": "Clock",
            "logger": "Logger",
            "cache": "CacheFacade",
            "handler": "Callable",
            "callback": "Callable",
        }

        if param_name in param_type_map:
            base_type = param_type_map[param_name]
            # If has None default, make it optional
            if default_val == "None":
                return f"{base_type} | None"
            return base_type

        # Interval/duration parameters
        if param_name.endswith("_interval_seconds") or param_name.endswith(
            "_interval_ms"
        ):
            return "int | None" if default_val == "None" else "int"

        # Delay parameters
        if param_name.endswith("_delay"):
            return "int"

        return None

    def _lookup_property_type(self, class_name: str, prop_name: str) -> str | None:
        """Look up property type from class definition.

        Args:
            class_name: Name of the class (e.g., 'AccountState')
            prop_name: Name of the property (e.g., 'account_id')

        Returns:
            Type hint if found, None otherwise
        """
        # Check if class is in current file
        for cls in self.classes:
            if cls.name == class_name:
                # Search member variables for the property
                for var in cls.member_variables:
                    var_name = var.name.replace("self.", "")
                    if var_name == prop_name:
                        return var.type_hint

                # Also check methods (in case it's a property method)
                for method in cls.methods:
                    if method.name == prop_name and method.is_property:
                        return method.return_type

        return None

    def _add_variable(
        self,
        name: str,
        type_hint: str | None,
        value: str | None,
        is_public: bool = False,
        is_readonly: bool = False,
        is_class: bool = False,
        is_instance: bool = False,
        line_number: int | None = None,
    ):
        is_private = name.replace("self.", "").startswith("_") and not name.startswith(
            "__"
        )
        is_constant = name.isupper()

        # Use TypeRegistry for member variables to get the most accurate type
        # This is called after all type sources have been recorded
        if name.startswith("self."):
            member_name = name.replace("self.", "")
            final_type = self.type_registry.get_final_type(member_name)
            if final_type:
                type_hint = final_type

        if (
            is_class
            and is_instance
            and name.startswith("self.")
            and self.current_function
            and self.current_function.name == "__init__"
        ):
            member_var = MemberVariable(
                name=name,
                type_hint=type_hint,
                is_private=is_private,
                default_value=value,
                is_public=is_public,
                is_readonly=is_readonly,
                is_instance=True,
                line_number=line_number,
            )
            if self.current_class:
                self.current_class.member_variables.append(member_var)
        elif is_class and not is_instance:
            member_var = MemberVariable(
                name=name,
                type_hint=type_hint,
                is_private=is_private,
                default_value=value,
                is_public=is_public,
                is_readonly=is_readonly,
                is_class=True,
                line_number=line_number,
            )
            if self.current_class:
                self.current_class.member_variables.append(member_var)
        elif not is_class and not is_instance:
            global_var = GlobalVariable(
                name=name,
                type_hint=type_hint,
                value=value,
                is_constant=is_constant,
                line_number=line_number,
            )
            self.global_variables.append(global_var)

    def finalize_member_types(self):
        """Finalize member variable types after all type sources have been collected."""
        for cls in self.classes:
            for var in cls.member_variables:
                if var.name.startswith("self."):
                    member_name = var.name.replace("self.", "")
                    final_type = self.type_registry.get_final_type(member_name)

                    # If no type found or just "Any | None", try import-based inference
                    if not final_type or final_type == "Any | None":
                        inferred_type = self._infer_type_from_imports(member_name)
                        if inferred_type:
                            # Add | None if needed
                            if member_name in self.type_registry.has_none:
                                if "| None" not in inferred_type:
                                    inferred_type = f"{inferred_type} | None"
                            final_type = inferred_type

                    # NEW: Try inferring from default value
                    if not final_type or final_type == "Any":
                        if var.default_value:
                            default_type = self._infer_type_from_default(
                                var.default_value
                            )
                            if default_type and default_type != "None":
                                final_type = default_type
                                if member_name in self.type_registry.has_none:
                                    if "| None" not in final_type:
                                        final_type = f"{final_type} | None"

                    # Special handling for 'log' member (common pattern)
                    if member_name == "log" and (
                        not final_type or final_type == "Any | None"
                    ):
                        final_type = "Logger"
                        if member_name in self.type_registry.has_none:
                            final_type = "Logger | None"

                    # NEW: Pattern-based inference for common config attribute names
                    if not final_type or final_type == "Any":
                        # Boolean flags: use_xxx, is_xxx, has_xxx, enable_xxx
                        if (
                            member_name.startswith("use_")
                            or member_name.startswith("is_")
                            or member_name.startswith("has_")
                            or member_name.startswith("enable_")
                            or member_name.startswith("allow_")
                            or member_name.startswith("debug")
                            or member_name.startswith("manage_")
                            or member_name.startswith("persist_")
                            or member_name.startswith("snapshot_")
                            or member_name.endswith("_enabled")
                        ):
                            final_type = "bool"
                        # Timestamp fields: ts_xxx
                        elif (
                            member_name.startswith("ts_")
                            or member_name.endswith("_ns")
                            or member_name.endswith("_timestamp")
                            or member_name.endswith("_nanos")
                            or member_name.endswith("_secs")
                        ):
                            final_type = "int"
                        # Count fields
                        elif member_name.endswith("_count"):
                            final_type = "int"
                        # Capacity/size fields
                        elif member_name.endswith("_capacity") or member_name.endswith(
                            "_size"
                        ):
                            final_type = "int"
                        # Precision fields
                        elif member_name.endswith("_precision"):
                            final_type = "int"
                        # Known type patterns from nautilus_trader domain
                        elif (
                            member_name == "price_increment"
                            or member_name == "min_price"
                            or member_name == "max_price"
                        ):
                            final_type = "Price"
                        elif member_name == "multiplier":
                            final_type = "Quantity"
                        elif (
                            member_name == "quote_currency"
                            or member_name == "base_currency"
                        ):
                            final_type = "Currency"
                        elif member_name == "quantity_change":
                            final_type = "Quantity"
                        elif member_name == "order_type":
                            final_type = "OrderType"
                        elif member_name == "oms_type":
                            final_type = "OmsType"
                        elif member_name == "ma_type":
                            final_type = "MovingAverageType"
                        elif member_name == "market_status":
                            final_type = "MarketStatus"
                        elif member_name.endswith("_order_id"):
                            final_type = "ClientOrderId"
                        elif member_name == "trigger_instrument_id":
                            final_type = "InstrumentId"
                        elif member_name == "tags":
                            final_type = "list[str]"
                        elif member_name == "params":
                            final_type = "dict[str, Any]"
                        elif member_name == "metadata":
                            final_type = "dict[str, Any]"
                        elif member_name == "options":
                            final_type = "dict[str, Any]"
                        elif member_name == "weights":
                            final_type = "list[float]"
                        elif member_name == "margins":
                            final_type = "dict[str, Any]"
                        elif member_name == "modules":
                            final_type = "list[Any]"
                        elif member_name == "data":
                            # data field is dynamic, use object (more specific than Any)
                            final_type = "object"
                        elif member_name == "name":
                            final_type = "str"
                        elif member_name == "topic":
                            final_type = "str"
                        elif member_name == "rate":
                            final_type = "float"
                        elif member_name.startswith("alpha"):
                            final_type = "float"
                        elif member_name == "init_id":
                            final_type = "UUID4"
                        elif member_name == "exec_algorithm_id":
                            final_type = "ExecAlgorithmId"
                        elif member_name == "exec_algorithm_params":
                            final_type = "dict[str, Any]"
                        elif member_name == "exec_spawn_id":
                            final_type = "ClientOrderId"
                        elif member_name == "linked_order_ids":
                            final_type = "list[ClientOrderId]"
                        elif member_name == "liquidity_side":
                            final_type = "LiquiditySide"
                        elif member_name == "instrument_class":
                            final_type = "InstrumentClass"
                        elif member_name == "contingency_type":
                            final_type = "ContingencyType"
                        elif member_name == "emulation_trigger":
                            final_type = "TriggerType"
                        elif member_name == "balances":
                            final_type = "list[Any]"
                        elif member_name == "expiry_timers":
                            final_type = "dict[str, Any]"
                        elif member_name == "price_protection_points":
                            final_type = "float"
                        elif member_name == "id":
                            # Context-dependent, often UUID4
                            final_type = "UUID4"
                        elif member_name == "entry":
                            final_type = "OrderSide"
                        elif member_name == "first":
                            final_type = "Order"
                        # Private members from Actor
                        elif member_name == "_log_events":
                            final_type = "bool"
                        elif member_name == "_log_commands":
                            final_type = "bool"
                        elif member_name == "_warning_events":
                            final_type = "set[type]"
                        # Topic cache members (all dict[str, str])
                        elif member_name.startswith("_topic_cache_"):
                            final_type = "dict[str, str]"
                        # Timer name
                        elif member_name == "_timer_name":
                            final_type = "str"
                        # Time bars config
                        elif member_name == "_time_bars_interval_type":
                            final_type = "str"
                        elif member_name == "_time_bars_timestamp_on_close":
                            final_type = "bool"
                        elif member_name == "_time_bars_skip_first_non_full_bar":
                            final_type = "bool"
                        elif member_name == "_time_bars_build_with_no_updates":
                            final_type = "bool"
                        elif member_name == "_time_bars_build_delay":
                            final_type = "int"
                        elif member_name == "_time_bars_origin_offset":
                            final_type = "dict[str, Any]"
                        # Precision members
                        elif (
                            member_name == "_price_prec" or member_name == "_size_prec"
                        ):
                            final_type = "int"
                        # Other private members
                        elif member_name == "_price_protection_points":
                            final_type = "float"
                        elif member_name == "_log_rejected_due_post_only_as_warning":
                            final_type = "bool"
                        elif member_name == "_validate_data_sequence":
                            final_type = "bool"
                        elif member_name == "_rollover_totals":
                            final_type = "dict[str, Any]"
                        elif member_name == "_listeners":
                            final_type = "dict[str, list[Any]]"
                        elif member_name == "_previous_status":
                            final_type = "InstrumentStatus | None"
                        elif member_name == "_response_data":
                            final_type = "dict[str, Any]"
                        elif member_name == "_reason":
                            final_type = "str | None"
                        elif member_name == "_raw_step":
                            final_type = "int"
                        elif member_name == "_rate_data":
                            final_type = "dict[str, Any]"
                        # More private members
                        elif member_name == "_spread_instrument_id":
                            final_type = "InstrumentId | None"
                        elif member_name == "_leg_ids":
                            final_type = "dict[str, InstrumentId]"
                        elif member_name == "_last_quotes":
                            final_type = "dict[InstrumentId, QuoteTick]"
                        elif member_name == "_instance_id":
                            final_type = "UUID4"
                        elif member_name == "_increment_pow10":
                            final_type = "int"
                        elif member_name == "_historical_events":
                            final_type = "list[Event]"
                        elif member_name == "_heap":
                            final_type = "list[Any]"
                        elif member_name == "_encode":
                            final_type = "Callable[[Any], bytes]"
                        elif member_name == "_decode":
                            final_type = "Callable[[bytes], Any]"
                        elif member_name == "_emit_quotes_from_book":
                            final_type = "bool"
                        elif member_name == "_emit_quotes_from_book_depths":
                            final_type = "bool"
                        elif member_name == "_drop_instruments_on_reset":
                            final_type = "bool"
                        elif member_name == "_data_update_function":
                            final_type = "Callable[[Any], None]"
                        elif member_name == "_data_priority":
                            final_type = "int"
                        elif member_name == "_data_name":
                            final_type = "str"
                        elif member_name == "_data_index":
                            final_type = "int"
                        elif member_name == "_current_run_side":
                            final_type = "OrderSide"
                        elif member_name == "_config":
                            final_type = "NautilusConfig"
                        elif member_name == "_commissions":
                            final_type = "dict[str, Any]"
                        elif member_name == "_buffer_deltas":
                            final_type = "list[OrderBookDelta]"
                        elif member_name == "_bid_consumption":
                            final_type = "float"
                        elif member_name == "_ask_consumption":
                            final_type = "float"
                        elif member_name == "_accumulator":
                            final_type = "Any"  # Generic accumulator

                    if final_type:
                        var.type_hint = final_type

    def _extract_function_args(self, node, is_cfunc: bool) -> list[str]:  # noqa: C901
        args = []
        node_args = node.declarator.args if is_cfunc else node.args

        for arg in node_args:
            got_name_from_type = False
            if hasattr(arg, "declarator"):
                arg_name = self._extract_name_from_node(arg.declarator)
                if not arg_name and hasattr(arg, "base_type"):
                    arg_name = self._extract_name_from_node(arg.base_type)
                    got_name_from_type = True

            arg_type = None
            if hasattr(arg, "annotation") and arg.annotation:
                arg_type = self._extract_type_from_node(arg.annotation)
            elif hasattr(arg, "type") and arg.type:
                arg_type = self._extract_type_from_node(arg.type)
            elif hasattr(arg, "base_type") and arg.base_type:
                arg_type = self._extract_type_from_node(arg.base_type)
                if got_name_from_type:
                    arg_type = None

            arg_type = self.map_cython_type(arg_type) if arg_type else None

            # Extract default value first (needed for type inference)
            default_val = None
            if hasattr(arg, "default") and arg.default:
                default_val = self._extract_value_from_node(arg.default)

            # Infer type from parameter name if not specified
            if not arg_type or arg_type == "Any":
                arg_type = self._infer_param_type_from_name(arg_name, default_val)

            if arg_type == "self":
                arg_str = "self"
            else:
                if arg_name == "" and arg_type:  # when there is no name and is type
                    arg_name, arg_type = arg_type, None

                arg_str = arg_name
                if arg_type and arg_type != "self":
                    arg_str += f": {arg_type}"
                if default_val:
                    arg_str += f" = {default_val}"

            args.append(arg_str)

        return args

    def _extract_return_type(self, node, is_cfunc: bool) -> str | None:
        if is_cfunc:
            return self.map_cython_type(self._extract_name_from_node(node.base_type))
        if hasattr(node, "return_type_annotation") and node.return_type_annotation:
            if (
                hasattr(node.return_type_annotation, "string")
                and node.return_type_annotation.string
            ):
                return self._extract_type_from_node(node.return_type_annotation)
            return self._extract_type_from_node(node.return_type_annotation.expr)
        return None

    def _analyze_decorators(self, node) -> dict[str, bool]:
        decorators = {}
        if not hasattr(node, "decorators") or not node.decorators:
            return decorators

        for decorator in node.decorators:
            if hasattr(decorator, "decorator"):
                dec_name = self._extract_name_from_node(decorator.decorator)
                if dec_name in ["staticmethod", "classmethod", "property"]:
                    decorators[dec_name] = True

        return decorators

    def _extract_name_from_node(self, node) -> str | None:
        if node is None:
            return None
        if hasattr(node, "name"):
            return node.name
        if hasattr(node, "attribute") and hasattr(node, "obj"):
            obj_name = self._extract_name_from_node(node.obj)
            return f"{obj_name}.{node.attribute}" if obj_name else node.attribute
        return str(node) if node else None

    def _extract_type_from_node(self, type_node) -> str | None:
        if type_node is None:
            return None
        if hasattr(type_node, "string") and hasattr(
            type_node.string, "constant_result"
        ):
            return type_node.string.constant_result
        if hasattr(type_node, "name"):
            return type_node.name
        if isinstance(type_node, PyrexTypes.BaseType):
            return str(type_node)
        if hasattr(type_node, "__str__"):
            return str(type_node)
        return None

    def _extract_value_from_node(self, node) -> str | None:
        if node is None:
            return None

        # Check if this is a string literal (UnicodeNode or StringNode)
        node_class_name = node.__class__.__name__
        if node_class_name in ("UnicodeNode", "StringNode"):
            if hasattr(node, "value"):
                # String literals need to be wrapped in quotes
                return f'"{node.value}"'

        # Check for bytes literals
        if node_class_name == "BytesNode":
            if hasattr(node, "value"):
                return f'b"{node.value}"'

        if hasattr(node, "value"):
            value = str(node.value)
            return "None" if value == "Py_None" else value
        if hasattr(node, "compile_time_value"):
            return "expr"
        return str(node)

    def _extract_doc_from_node(self, node) -> str | None:
        if node is None or not hasattr(node, "doc"):
            return None
        doc = str(node.doc)
        return doc if doc != "None" else None

    def map_cython_type(self, type_hint: str) -> str:
        type_map = {
            "object": "Any",
            "bint": "bool",
            "double": "float",
            "uint64_t": "int",
            "int64_t": "int",
            "uint32_t": "int",
            "int32_t": "int",
            "uint16_t": "int",
            "int16_t": "int",
            "uint8_t": "int",
            "int8_t": "int",
            "long": "int",
            "void": "None",
        }
        return type_map.get(type_hint, type_hint)


def analyze_cython_code(name: str, code_content: str) -> CythonCodeAnalyzer:
    options = CompilationOptions(default_options)
    context = Context(
        include_directories=["./"], compiler_directives={}, options=options
    )

    try:
        Errors.init_thread()
        tree = parse_from_strings(name, code_content)
        if tree:
            analyzer = CythonCodeAnalyzer(context)
            analyzer.visit(tree)
            # Finalize member variable types after all type sources collected
            analyzer.finalize_member_types()
            return analyzer
        else:
            raise ValueError("Failed to parse code.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        traceback.print_exc()
        return CythonCodeAnalyzer(context)


def print_results(analyzer: CythonCodeAnalyzer):  # noqa: C901
    if analyzer.classes:
        print("\n Classes:")
        for cls in analyzer.classes:
            class_type = "cdef class" if cls.is_cdef_class else "class"
            extension_info = " (extension type)" if cls.is_extension_type else ""
            print(f"\n{class_type}: {cls.name}{extension_info}")

            if cls.base_classes:
                print(f"  Inherits: {', '.join(cls.base_classes)}")

            if cls.docstring:
                print(f'  """{cls.docstring}"""')

            if cls.member_variables:
                print("  Member Variables:")
                for var in cls.member_variables:
                    visibility = "private" if var.is_private else "public"
                    type_info = f": {var.type_hint}" if var.type_hint else ""
                    default_info = (
                        f" = {var.default_value}" if var.default_value else ""
                    )

                    modifiers = []
                    if var.is_public:
                        modifiers.append("public")
                    if var.is_readonly:
                        modifiers.append("readonly")
                    modifier_str = f" ({', '.join(modifiers)})" if modifiers else ""
                    scope = "instance" if var.is_instance else "class"

                    print(
                        f"    - {var.name}{type_info}{default_info} ({visibility}){modifier_str} {scope}"
                    )

            if cls.methods:
                print("  Methods:")
                for method in cls.methods:
                    visibility = "private" if method.is_private else "public"

                    func_type = "def "
                    if method.is_cdef:
                        func_type = "cdef "
                    elif method.is_cpdef:
                        func_type = "cpdef "

                    decorators = []
                    if method.is_static:
                        decorators.append("@staticmethod")
                    if method.is_classmethod:
                        decorators.append("@classmethod")
                    if method.is_property:
                        decorators.append("@property")

                    decorator_str = " ".join(decorators) + " " if decorators else ""
                    args_str = ", ".join(method.args) if method.args else ""
                    return_str = (
                        f" -> {method.return_type}" if method.return_type else ""
                    )

                    print(
                        f"    - {decorator_str}{func_type}{method.name}({args_str}){return_str} ({visibility})"
                    )
                    if method.docstring:
                        print(f'    """{method.docstring}"""')

    if analyzer.functions:
        print("\n  Functions:")
        for func in analyzer.functions:
            func_type = "def "
            if func.is_cdef:
                func_type = "cdef "
            elif func.is_cpdef:
                func_type = "cpdef "

            args_str = ", ".join(func.args) if func.args else ""
            return_str = f" -> {func.return_type}" if func.return_type else ""
            print(f"  - {func_type}{func.name}({args_str}){return_str}")
            if func.docstring:
                print(f'  """{func.docstring}"""')

    if analyzer.global_variables:
        print("\n  Global Variables:")
        for var in analyzer.global_variables:
            type_info = f": {var.type_hint}" if var.type_hint else ""
            value_info = f" = {var.value}" if var.value else ""
            classification = "Constant" if var.is_constant else "Variable"
            print(f"  - {var.name}{type_info}{value_info} ({classification})")
