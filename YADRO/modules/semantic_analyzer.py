#================================================#
#  YY   YY    AAA      DDDDD    RRRR     OOOO    #
#   YY YY    AA AA     DD  DD   R  RR   OO  OO   #
#    YYY    AA   AA    DD   DD  R RR    OO  OO   #
#     Y    AAAAAAAAA   DD  DD   R  RR   OO  OO   #
#     Y   AA       AA  DDDDD    R   RR   OOOO    #
#================================================#

#  Semantyc Analyzer module for YADRO compiler   

#  version 0.2.0 
#  Made by CyrOil

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Union, Tuple
from enum import Enum, auto
from parser import (
    Program, Directive, Statement, Expression, TypeExpr, ASTNode,
    FunctionDecl, ClassDecl, VariableDecl, BlockStmt, IfStmt, ReturnStmt,
    BinaryExpr, AssignmentExpr, CallExpr, IdentifierExpr, LiteralExpr,
    UnaryExpr, PipelineExpr, IndexExpr, MemberExpr, ErrorPropagateExpr,
    BasicType, ArrayType, DictType, SetType, VectorType, RefType, GcType,
    ResultType, OptionType, PredicatedType, NamedType, GenericType,
    Parameter, TraitDecl, ImplDecl, UnsafeBlock, ExprStmt, DelStmt,
    SwitchStmt, ForStmt, WhileStmt, RepeatUntilStmt, TargetDirective,
    ImportDirective, RequiresDirective, PluginDirective, DefineDirective,
    FunctionModifier, ClassModifier
)
from lexer import Token, TokenType

class SymbolKind(Enum):
    VARIABLE = auto()
    FUNCTION = auto()
    CLASS = auto()
    TRAIT = auto()
    MODULE = auto()
    GENERIC_PARAM = auto()

@dataclass
class Symbol:
    name: str
    kind: SymbolKind
    node: ASTNode
    type_expr: Optional[TypeExpr] = None
    is_mutable: bool = False
    is_linear: bool = False
    is_borrowed: bool = False
    borrow_count: int = 0
    scope_depth: int = 0
    defined_at: Token = None

class OwnershipKind(Enum):
    OWNED = auto()
    BORROWED_IMMUTABLE = auto()
    BORROWED_MUTABLE = auto()
    LINEAR = auto()
    ACTOR = auto()

@dataclass
class TypeCheckResult:
    type_expr: TypeExpr
    ownership: OwnershipKind
    is_const: bool = False
    predicates: List[str] = field(default_factory=list)

class SemanticError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        loc = f" @{token.line}:{token.column}" if token else ""
        super().__init__(f"Constitutional Violation{loc}: {message}")
        self.token = token

class SymbolTable:
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.parent = parent
        self.symbols: Dict[str, Symbol] = {}
        self.depth = parent.depth + 1 if parent else 0

    def define(self, symbol: Symbol) -> None:
        symbol.scope_depth = self.depth
        self.symbols[symbol.name] = symbol

    def resolve(self, name: str) -> Optional[Symbol]:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.resolve(name)
        return None

    def exists_in_current_scope(self, name: str) -> bool:
        return name in self.symbols

    def enter_scope(self) -> 'SymbolTable':
        return SymbolTable(self)

    def exit_scope(self) -> Optional['SymbolTable']:
        return self.parent

class SemanticAnalyzer:
    def __init__(self):
        self.global_scope = SymbolTable()
        self.current_scope = self.global_scope
        self.errors: List[SemanticError] = []
        self.current_function: Optional[FunctionDecl] = None
        self.current_class: Optional[ClassDecl] = None
        self.in_unsafe_block = False
        self.required_error_handlers: List[Tuple[Expression, Token]] = []
        self.linear_vars: Dict[str, Token] = {}
        self.borrowed_vars: Dict[str, List[Tuple[Token, bool]]] = {}

    def analyze(self, program: Program) -> None:
        self.validate_directives(program.directives)
        self.predeclare_globals(program)
        self.analyze_statements(program.statements)
        self.check_unhandled_errors()
        self.check_linear_consumption()
        if self.errors:
            raise SemanticError(f"{len(self.errors)} constitutional violations detected", self.errors[0].token if self.errors else None)

    def validate_directives(self, directives: List[Directive]) -> None:
        found_target = False
        for directive in directives:
            if isinstance(directive, TargetDirective):
                if found_target:
                    raise SemanticError("Multiple #target directives forbidden (Article I.1: single execution target)", directive.token)
                found_target = True
                if directive.os not in {"yadro", "linux", "windows", "macos", "baremetal"}:
                    raise SemanticError(f"Unsupported target OS '{directive.os}' (Article I.2: verified platforms only)", directive.token)
                if directive.arch not in {"kx", "arm64", "x86_64"}:
                    raise SemanticError(f"Unsupported target architecture '{directive.arch}' (Article I.2: verified ISAs only)", directive.token)
            elif isinstance(directive, ImportDirective):
                if not directive.module_path.startswith(("std.", "core.", "yadro.")):
                    raise SemanticError(f"Unverified external import '{directive.module_path}' forbidden (Article I.3: trusted sources only)", directive.token)

    def predeclare_globals(self, program: Program) -> None:
        for stmt in program.statements:
            if isinstance(stmt, FunctionDecl):
                func_type = self.construct_function_type(stmt)
                symbol = Symbol(
                    name=stmt.name,
                    kind=SymbolKind.FUNCTION,
                    node=stmt,
                    type_expr=func_type,
                    defined_at=stmt.token
                )
                self.global_scope.define(symbol)
            elif isinstance(stmt, ClassDecl):
                class_type = NamedType(name=stmt.name, token=stmt.token)
                symbol = Symbol(
                    name=stmt.name,
                    kind=SymbolKind.CLASS,
                    node=stmt,
                    type_expr=class_type,
                    defined_at=stmt.token
                )
                self.global_scope.define(symbol)
            elif isinstance(stmt, TraitDecl):
                symbol = Symbol(
                    name=stmt.name,
                    kind=SymbolKind.TRAIT,
                    node=stmt,
                    defined_at=stmt.token
                )
                self.global_scope.define(symbol)

    def construct_function_type(self, func: FunctionDecl) -> TypeExpr:
        return_type = func.return_type if func.return_type else BasicType(name="Unit", token=func.token)
        param_types = [p.type_expr for p in func.params if p.type_expr]
        inner = return_type
        for param_type in reversed(param_types):
            inner = NamedType(name="fn", generics=[param_type, inner], token=func.token)
        return inner

    def analyze_statements(self, statements: List[Statement]) -> None:
        for stmt in statements:
            try:
                self.analyze_statement(stmt)
            except SemanticError as e:
                self.errors.append(e)

    def analyze_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, FunctionDecl):
            self.analyze_function_decl(stmt)
        elif isinstance(stmt, ClassDecl):
            self.analyze_class_decl(stmt)
        elif isinstance(stmt, TraitDecl):
            self.analyze_trait_decl(stmt)
        elif isinstance(stmt, ImplDecl):
            self.analyze_impl_decl(stmt)
        elif isinstance(stmt, VariableDecl):
            self.analyze_variable_decl(stmt)
        elif isinstance(stmt, BlockStmt):
            self.analyze_block(stmt)
        elif isinstance(stmt, IfStmt):
            self.analyze_if_stmt(stmt)
        elif isinstance(stmt, SwitchStmt):
            self.analyze_switch_stmt(stmt)
        elif isinstance(stmt, ForStmt):
            self.analyze_for_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self.analyze_while_stmt(stmt)
        elif isinstance(stmt, RepeatUntilStmt):
            self.analyze_repeat_until_stmt(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.analyze_return_stmt(stmt)
        elif isinstance(stmt, ExprStmt):
            result = self.analyze_expression(stmt.expr)
            if self.is_result_type(result.type_expr) and not self.in_unsafe_block:
                self.required_error_handlers.append((stmt.expr, stmt.token))
        elif isinstance(stmt, DelStmt):
            self.analyze_del_stmt(stmt)
        elif isinstance(stmt, UnsafeBlock):
            prev_unsafe = self.in_unsafe_block
            self.in_unsafe_block = True
            self.analyze_block(stmt.body)
            self.in_unsafe_block = prev_unsafe
        else:
            raise SemanticError(f"Unsupported statement type: {type(stmt).__name__}", stmt.token)

    def analyze_function_decl(self, func: FunctionDecl) -> None:
        prev_function = self.current_function
        self.current_function = func
        func_scope = self.current_scope.enter_scope()
        self.current_scope = func_scope
        for generic in func.generics:
            symbol = Symbol(
                name=generic.name,
                kind=SymbolKind.GENERIC_PARAM,
                node=generic,
                defined_at=func.token
            )
            self.current_scope.define(symbol)
        for param in func.params:
            if param.is_self:
                if not self.current_class:
                    raise SemanticError("'self' parameter only allowed in methods", param.token)
                self_type = NamedType(name=self.current_class.name, token=func.token)
                ownership = OwnershipKind.LINEAR if ClassModifier.LINEAR in self.current_class.modifiers else OwnershipKind.BORROWED_IMMUTABLE
                symbol = Symbol(
                    name="self",
                    kind=SymbolKind.VARIABLE,
                    node=param,
                    type_expr=self_type,
                    is_mutable="mut" in param.name,
                    is_linear=ownership == OwnershipKind.LINEAR,
                    ownership=ownership,
                    defined_at=func.token
                )
                self.current_scope.define(symbol)
            else:
                if not param.type_expr:
                    raise SemanticError(f"Parameter '{param.name}' missing type annotation (Article I.4: explicit typing required)", param.token)
                symbol = Symbol(
                    name=param.name,
                    kind=SymbolKind.VARIABLE,
                    node=param,
                    type_expr=param.type_expr,
                    is_mutable=param.default_value is not None,
                    defined_at=param.token
                )
                self.current_scope.define(symbol)
        if func.body:
            self.analyze_block(func.body)
            if func.return_type and not self.function_returns_value(func):
                raise SemanticError(f"Function '{func.name}' missing return statement (Article I.5: complete control flow)", func.token)
        self.current_scope = self.current_scope.exit_scope()
        self.current_function = prev_function

    def function_returns_value(self, func: FunctionDecl) -> bool:
        if not func.body:
            return False
        return self.check_block_returns(func.body)

    def check_block_returns(self, block: BlockStmt) -> bool:
        if not block.statements:
            return False
        last = block.statements[-1]
        if isinstance(last, ReturnStmt):
            return True
        if isinstance(last, IfStmt):
            has_else = last.else_branch is not None
            then_returns = self.check_block_returns(last.then_branch)
            else_returns = has_else and self.check_block_returns(last.else_branch)
            elif_returns = all(self.check_block_returns(b) for _, b in last.elif_branches)
            return then_returns and elif_returns and (not has_else or else_returns)
        return False

    def analyze_class_decl(self, cls: ClassDecl) -> None:
        prev_class = self.current_class
        self.current_class = cls
        class_scope = self.current_scope.enter_scope()
        self.current_scope = class_scope
        for stmt in cls.body:
            if isinstance(stmt, FunctionDecl):
                self.analyze_function_decl(stmt)
            elif isinstance(stmt, VariableDecl):
                self.analyze_variable_decl(stmt)
        self.current_scope = self.current_scope.exit_scope()
        self.current_class = prev_class

    def analyze_trait_decl(self, trait: TraitDecl) -> None:
        trait_scope = self.current_scope.enter_scope()
        self.current_scope = trait_scope
        for generic in trait.generics:
            symbol = Symbol(
                name=generic.name,
                kind=SymbolKind.GENERIC_PARAM,
                node=generic,
                defined_at=trait.token
            )
            self.current_scope.define(symbol)
        for method in trait.methods:
            self.analyze_function_decl(method)
        self.current_scope = self.current_scope.exit_scope()

    def analyze_impl_decl(self, impl: ImplDecl) -> None:
        trait_symbol = self.current_scope.resolve(impl.trait_name)
        if not trait_symbol or trait_symbol.kind != SymbolKind.TRAIT:
            raise SemanticError(f"Trait '{impl.trait_name}' not found for impl (Article I.6: verified trait implementations)", impl.token)
        type_symbol = self.current_scope.resolve(impl.type_name)
        if not type_symbol or type_symbol.kind not in {SymbolKind.CLASS, SymbolKind.TYPE_ALIAS}:
            raise SemanticError(f"Type '{impl.type_name}' not found for impl (Article I.6: verified type implementations)", impl.token)
        impl_scope = self.current_scope.enter_scope()
        self.current_scope = impl_scope
        for method in impl.methods:
            self.analyze_function_decl(method)
        self.current_scope = self.current_scope.exit_scope()

    def analyze_variable_decl(self, decl: VariableDecl) -> None:
        if not decl.type_expr:
            raise SemanticError(f"Variable '{decl.name}' missing type annotation (Article I.4: explicit typing required)", decl.token)
        if self.current_scope.exists_in_current_scope(decl.name):
            raise SemanticError(f"Variable '{decl.name}' already declared in this scope (Article I.7: no shadowing)", decl.token)
        if decl.initializer:
            result = self.analyze_expression(decl.initializer)
            self.validate_type_compatibility(result.type_expr, decl.type_expr, decl.token)
            ownership = self.determine_ownership(decl.type_expr, decl.is_const)
            if ownership == OwnershipKind.LINEAR and decl.is_const:
                raise SemanticError(f"Linear type cannot be const (ownership transfer required)", decl.token)
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.VARIABLE,
                node=decl,
                type_expr=decl.type_expr,
                is_mutable=not decl.is_const,
                is_linear=ownership == OwnershipKind.LINEAR,
                ownership=ownership,
                defined_at=decl.token
            )
            self.current_scope.define(symbol)
            if ownership == OwnershipKind.LINEAR:
                self.linear_vars[decl.name] = decl.token
        else:
            if not decl.is_const:
                raise SemanticError(f"Mutable variable '{decl.name}' requires initializer (Article I.8: no uninitialized mutables)", decl.token)
            symbol = Symbol(
                name=decl.name,
                kind=SymbolKind.VARIABLE,
                node=decl,
                type_expr=decl.type_expr,
                is_mutable=False,
                defined_at=decl.token
            )
            self.current_scope.define(symbol)

    def analyze_block(self, block: BlockStmt) -> None:
        block_scope = self.current_scope.enter_scope()
        prev_scope = self.current_scope
        self.current_scope = block_scope
        for stmt in block.statements:
            self.analyze_statement(stmt)
        self.current_scope = prev_scope

    def analyze_if_stmt(self, stmt: IfStmt) -> None:
        cond_result = self.analyze_expression(stmt.condition)
        if not self.is_bool_type(cond_result.type_expr):
            raise SemanticError("If condition must be boolean type", stmt.condition.token)
        self.analyze_block(stmt.then_branch)
        for elif_cond, elif_body in stmt.elif_branches:
            cond_result = self.analyze_expression(elif_cond)
            if not self.is_bool_type(cond_result.type_expr):
                raise SemanticError("Elif condition must be boolean type", elif_cond.token)
            self.analyze_block(elif_body)
        if stmt.else_branch:
            self.analyze_block(stmt.else_branch)

    def analyze_switch_stmt(self, stmt: SwitchStmt) -> None:
        expr_result = self.analyze_expression(stmt.expr)
        for case_expr, case_body in stmt.cases:
            case_result = self.analyze_expression(case_expr)
            self.validate_type_compatibility(case_result.type_expr, expr_result.type_expr, case_expr.token)
            self.analyze_block(case_body)
        if stmt.default_case:
            self.analyze_block(stmt.default_case)

    def analyze_for_stmt(self, stmt: ForStmt) -> None:
        if not stmt.var_type:
            raise SemanticError("For loop variable requires explicit type annotation", stmt.token)
        iter_result = self.analyze_expression(stmt.iterable)
        if not self.is_iterable_type(iter_result.type_expr, stmt.var_type):
            raise SemanticError("For loop iterable must implement Iterator trait for the variable type", stmt.iterable.token)
        loop_scope = self.current_scope.enter_scope()
        prev_scope = self.current_scope
        self.current_scope = loop_scope
        symbol = Symbol(
            name=stmt.var_name,
            kind=SymbolKind.VARIABLE,
            node=stmt,
            type_expr=stmt.var_type,
            is_mutable=False,
            defined_at=stmt.token
        )
        self.current_scope.define(symbol)
        self.analyze_block(stmt.body)
        self.current_scope = prev_scope

    def analyze_while_stmt(self, stmt: WhileStmt) -> None:
        cond_result = self.analyze_expression(stmt.condition)
        if not self.is_bool_type(cond_result.type_expr):
            raise SemanticError("While condition must be boolean type", stmt.condition.token)
        self.analyze_block(stmt.body)

    def analyze_repeat_until_stmt(self, stmt: RepeatUntilStmt) -> None:
        self.analyze_block(stmt.body)
        cond_result = self.analyze_expression(stmt.condition)
        if not self.is_bool_type(cond_result.type_expr):
            raise SemanticError("Until condition must be boolean type", stmt.condition.token)

    def analyze_return_stmt(self, stmt: ReturnStmt) -> None:
        if not self.current_function:
            raise SemanticError("Return statement outside function", stmt.token)
        if stmt.value:
            result = self.analyze_expression(stmt.value)
            if not self.current_function.return_type:
                raise SemanticError("Returning value from function declared as void", stmt.token)
            self.validate_type_compatibility(result.type_expr, self.current_function.return_type, stmt.token)
        else:
            if self.current_function.return_type and not self.is_unit_type(self.current_function.return_type):
                raise SemanticError("Missing return value for non-void function", stmt.token)

    def analyze_del_stmt(self, stmt: DelStmt) -> None:
        result = self.analyze_expression(stmt.target)
        if not result.is_linear:
            raise SemanticError("del only permitted on linear-owned values (Article I.9: explicit disposal)", stmt.token)
        if isinstance(stmt.target, IdentifierExpr):
            if stmt.target.name in self.linear_vars:
                del self.linear_vars[stmt.target.name]

    def analyze_expression(self, expr: Expression) -> TypeCheckResult:
        if isinstance(expr, LiteralExpr):
            return self.analyze_literal(expr)
        elif isinstance(expr, IdentifierExpr):
            return self.analyze_identifier(expr)
        elif isinstance(expr, BinaryExpr):
            return self.analyze_binary(expr)
        elif isinstance(expr, UnaryExpr):
            return self.analyze_unary(expr)
        elif isinstance(expr, AssignmentExpr):
            return self.analyze_assignment(expr)
        elif isinstance(expr, CallExpr):
            return self.analyze_call(expr)
        elif isinstance(expr, IndexExpr):
            return self.analyze_index(expr)
        elif isinstance(expr, MemberExpr):
            return self.analyze_member(expr)
        elif isinstance(expr, ErrorPropagateExpr):
            return self.analyze_error_propagate(expr)
        elif isinstance(expr, PipelineExpr):
            return self.analyze_pipeline(expr)
        else:
            raise SemanticError(f"Unsupported expression type: {type(expr).__name__}", expr.token)

    def analyze_literal(self, expr: LiteralExpr) -> TypeCheckResult:
        if expr.literal_type == TokenType.LITERAL_INT:
            return TypeCheckResult(type_expr=BasicType(name="int", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.literal_type == TokenType.LITERAL_FLOAT:
            return TypeCheckResult(type_expr=BasicType(name="float", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.literal_type == TokenType.LITERAL_STRING:
            return TypeCheckResult(type_expr=BasicType(name="string", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.literal_type == TokenType.LITERAL_BOOL:
            return TypeCheckResult(type_expr=BasicType(name="bool", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.literal_type == TokenType.LITERAL_CHAR:
            return TypeCheckResult(type_expr=BasicType(name="char", token=expr.token), ownership=OwnershipKind.OWNED)
        else:
            return TypeCheckResult(type_expr=BasicType(name="Unit", token=expr.token), ownership=OwnershipKind.OWNED)

    def analyze_identifier(self, expr: IdentifierExpr) -> TypeCheckResult:
        symbol = self.current_scope.resolve(expr.name)
        if not symbol:
            raise SemanticError(f"Undefined identifier '{expr.name}' (Article I.10: no undefined references)", expr.token)
        if symbol.kind == SymbolKind.VARIABLE:
            if symbol.is_linear and expr.name in self.linear_vars:
                del self.linear_vars[expr.name]
            if symbol.name in self.borrowed_vars:
                self.borrowed_vars[symbol.name].append((expr.token, False))
            return TypeCheckResult(
                type_expr=symbol.type_expr,
                ownership=OwnershipKind.LINEAR if symbol.is_linear else OwnershipKind.OWNED,
                is_const=not symbol.is_mutable
            )
        elif symbol.kind == SymbolKind.FUNCTION:
            return TypeCheckResult(type_expr=symbol.type_expr, ownership=OwnershipKind.OWNED)
        else:
            return TypeCheckResult(type_expr=symbol.type_expr, ownership=OwnershipKind.OWNED)

    def analyze_binary(self, expr: BinaryExpr) -> TypeCheckResult:
        left = self.analyze_expression(expr.left)
        right = self.analyze_expression(expr.right)
        if expr.operator in {"and", "or", "xor", "nand"}:
            if not self.is_bool_type(left.type_expr) or not self.is_bool_type(right.type_expr):
                raise SemanticError(f"Logical operator '{expr.operator}' requires boolean operands", expr.token)
            return TypeCheckResult(type_expr=BasicType(name="bool", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.operator in {"+", "-", "*", "/", "%", "**", "//"}:
            if not self.is_numeric_type(left.type_expr) or not self.is_numeric_type(right.type_expr):
                raise SemanticError(f"Arithmetic operator '{expr.operator}' requires numeric operands", expr.token)
            return TypeCheckResult(type_expr=left.type_expr, ownership=OwnershipKind.OWNED)
        elif expr.operator in {"==", "!=", "<", ">", "<=", ">="}:
            self.validate_type_compatibility(left.type_expr, right.type_expr, expr.token)
            return TypeCheckResult(type_expr=BasicType(name="bool", token=expr.token), ownership=OwnershipKind.OWNED)
        elif expr.operator in {"&", "|", "^", "<<", ">>"}:
            if not self.is_integer_type(left.type_expr) or not self.is_integer_type(right.type_expr):
                raise SemanticError(f"Bitwise operator '{expr.operator}' requires integer operands", expr.token)
            return TypeCheckResult(type_expr=left.type_expr, ownership=OwnershipKind.OWNED)
        else:
            raise SemanticError(f"Unsupported binary operator '{expr.operator}'", expr.token)

    def analyze_unary(self, expr: UnaryExpr) -> TypeCheckResult:
        operand = self.analyze_expression(expr.operand)
        if expr.operator == "-":
            if not self.is_numeric_type(operand.type_expr):
                raise SemanticError("Unary minus requires numeric operand", expr.token)
            return TypeCheckResult(type_expr=operand.type_expr, ownership=OwnershipKind.OWNED)
        elif expr.operator == "~":
            if not self.is_integer_type(operand.type_expr):
                raise SemanticError("Bitwise NOT requires integer operand", expr.token)
            return TypeCheckResult(type_expr=operand.type_expr, ownership=OwnershipKind.OWNED)
        elif expr.operator == "*":
            if not isinstance(operand.type_expr, RefType):
                raise SemanticError("Dereference requires reference type", expr.token)
            return TypeCheckResult(type_expr=operand.type_expr.inner, ownership=OwnershipKind.BORROWED_MUTABLE if operand.type_expr.mutable else OwnershipKind.BORROWED_IMMUTABLE)
        else:
            raise SemanticError(f"Unsupported unary operator '{expr.operator}'", expr.token)

    def analyze_assignment(self, expr: AssignmentExpr) -> TypeCheckResult:
        if not isinstance(expr.left, (IdentifierExpr, IndexExpr, MemberExpr)):
            raise SemanticError("Assignment target must be lvalue", expr.left.token)
        left = self.analyze_expression(expr.left)
        if left.is_const and not self.in_unsafe_block:
            raise SemanticError("Cannot assign to const value (Article I.11: immutability guarantee)", expr.token)
        right = self.analyze_expression(expr.right)
        self.validate_type_compatibility(right.type_expr, left.type_expr, expr.token)
        if left.ownership == OwnershipKind.LINEAR and right.ownership != OwnershipKind.LINEAR:
            raise SemanticError("Linear assignment requires linear source (ownership transfer)", expr.token)
        if isinstance(expr.left, IdentifierExpr):
            symbol = self.current_scope.resolve(expr.left.name)
            if symbol and symbol.is_linear:
                if expr.left.name in self.linear_vars:
                    del self.linear_vars[expr.left.name]
                if right.ownership == OwnershipKind.LINEAR and isinstance(expr.right, IdentifierExpr):
                    if expr.right.name in self.linear_vars:
                        del self.linear_vars[expr.right.name]
        return TypeCheckResult(type_expr=BasicType(name="Unit", token=expr.token), ownership=OwnershipKind.OWNED)

    def analyze_call(self, expr: CallExpr) -> TypeCheckResult:
        callee = self.analyze_expression(expr.callee)
        if not isinstance(callee.type_expr, NamedType) or callee.type_expr.name != "fn":
            raise SemanticError("Call target must be function type", expr.callee.token)
        arg_results = [self.analyze_expression(arg) for arg in expr.args]
        if len(arg_results) != len(expr.args):
            raise SemanticError("Argument count mismatch", expr.token)
        return_type = callee.type_expr.generics[-1] if callee.type_expr.generics else BasicType(name="Unit", token=expr.token)
        if self.is_result_type(return_type) and not self.in_unsafe_block:
            self.required_error_handlers.append((expr, expr.token))
        return TypeCheckResult(type_expr=return_type, ownership=OwnershipKind.OWNED)

    def analyze_index(self, expr: IndexExpr) -> TypeCheckResult:
        container = self.analyze_expression(expr.container)
        index = self.analyze_expression(expr.index)
        if not self.is_integer_type(index.type_expr):
            raise SemanticError("Index must be integer type", expr.index.token)
        if isinstance(container.type_expr, ArrayType):
            return TypeCheckResult(type_expr=container.type_expr.element_type, ownership=OwnershipKind.BORROWED_IMMUTABLE)
        elif isinstance(container.type_expr, DictType):
            return TypeCheckResult(type_expr=container.type_expr.value_type, ownership=OwnershipKind.BORROWED_IMMUTABLE)
        else:
            raise SemanticError("Index operation requires array or dict type", expr.container.token)

    def analyze_member(self, expr: MemberExpr) -> TypeCheckResult:
        obj = self.analyze_expression(expr.object)
        if not isinstance(obj.type_expr, NamedType):
            raise SemanticError("Member access requires named type", expr.object.token)
        class_symbol = self.current_scope.resolve(obj.type_expr.name)
        if not class_symbol or class_symbol.kind != SymbolKind.CLASS:
            raise SemanticError(f"Type '{obj.type_expr.name}' has no members", expr.object.token)
        cls = class_symbol.node
        if not isinstance(cls, ClassDecl):
            raise SemanticError(f"'{obj.type_expr.name}' is not a class", expr.object.token)
        for stmt in cls.body:
            if isinstance(stmt, VariableDecl) and stmt.name == expr.member:
                return TypeCheckResult(type_expr=stmt.type_expr, ownership=OwnershipKind.BORROWED_IMMUTABLE)
        raise SemanticError(f"Member '{expr.member}' not found in class '{obj.type_expr.name}'", expr.token)

    def analyze_error_propagate(self, expr: ErrorPropagateExpr) -> TypeCheckResult:
        inner = self.analyze_expression(expr.expr)
        if not self.is_result_type(inner.type_expr):
            raise SemanticError("Error propagation (?) requires Result type", expr.expr.token)
        if not self.current_function:
            raise SemanticError("Error propagation only allowed inside functions", expr.token)
        if not self.current_function.return_type or not self.is_result_type(self.current_function.return_type):
            raise SemanticError("Function must return Result to use error propagation", expr.token)
        inner_type = self.unwrap_result_type(inner.type_expr)
        return_type = self.unwrap_result_type(self.current_function.return_type)
        self.validate_type_compatibility(inner_type, return_type, expr.token)
        return TypeCheckResult(type_expr=inner_type, ownership=OwnershipKind.OWNED)

    def analyze_pipeline(self, expr: PipelineExpr) -> TypeCheckResult:
        result = self.analyze_expression(expr.stages[0])
        for stage in expr.stages[1:]:
            stage_result = self.analyze_expression(stage)
            if not isinstance(stage_result.type_expr, NamedType) or stage_result.type_expr.name != "fn":
                raise SemanticError("Pipeline stage must be function", stage.token)
            if not stage_result.type_expr.generics:
                raise SemanticError("Pipeline function must have parameters", stage.token)
            param_type = stage_result.type_expr.generics[0]
            self.validate_type_compatibility(result.type_expr, param_type, stage.token)
            result = TypeCheckResult(type_expr=stage_result.type_expr.generics[-1], ownership=OwnershipKind.OWNED)
        return result

    def validate_type_compatibility(self, actual: TypeExpr, expected: TypeExpr, token: Token) -> None:
        if isinstance(expected, BasicType) and isinstance(actual, BasicType):
            if expected.name != actual.name:
                raise SemanticError(f"Type mismatch: expected {expected.name}, got {actual.name}", token)
        elif isinstance(expected, RefType) and isinstance(actual, RefType):
            if expected.mutable != actual.mutable:
                raise SemanticError("Reference mutability mismatch", token)
            self.validate_type_compatibility(actual.inner, expected.inner, token)
        elif isinstance(expected, OptionType) and isinstance(actual, OptionType):
            self.validate_type_compatibility(actual.inner, expected.inner, token)
        elif isinstance(expected, ResultType) and isinstance(actual, ResultType):
            self.validate_type_compatibility(actual.ok_type, expected.ok_type, token)
            self.validate_type_compatibility(actual.err_type, expected.err_type, token)
        elif isinstance(expected, PredicatedType) and isinstance(actual, PredicatedType):
            self.validate_type_compatibility(actual.base_type, expected.base_type, token)
        elif isinstance(expected, PredicatedType):
            self.validate_type_compatibility(actual, expected.base_type, token)
        elif self.is_unit_type(expected) and self.is_unit_type(actual):
            return
        else:
            raise SemanticError(f"Type mismatch: {self.type_to_string(actual)} vs {self.type_to_string(expected)}", token)

    def is_bool_type(self, type_expr: TypeExpr) -> bool:
        return isinstance(type_expr, BasicType) and type_expr.name == "bool"

    def is_numeric_type(self, type_expr: TypeExpr) -> bool:
        return isinstance(type_expr, BasicType) and type_expr.name in {"int", "float"}

    def is_integer_type(self, type_expr: TypeExpr) -> bool:
        return isinstance(type_expr, BasicType) and type_expr.name == "int"

    def is_unit_type(self, type_expr: TypeExpr) -> bool:
        return isinstance(type_expr, BasicType) and type_expr.name == "Unit"

    def is_result_type(self, type_expr: TypeExpr) -> bool:
        return isinstance(type_expr, ResultType) or (isinstance(type_expr, NamedType) and type_expr.name == "Result")

    def is_iterable_type(self, type_expr: TypeExpr, element_type: TypeExpr) -> bool:
        return isinstance(type_expr, ArrayType) or isinstance(type_expr, SetType) or isinstance(type_expr, DictType)

    def unwrap_result_type(self, type_expr: TypeExpr) -> TypeExpr:
        if isinstance(type_expr, ResultType):
            return type_expr.ok_type
        if isinstance(type_expr, NamedType) and type_expr.name == "Result" and type_expr.generics:
            return type_expr.generics[0]
        return type_expr

    def determine_ownership(self, type_expr: TypeExpr, is_const: bool) -> OwnershipKind:
        if isinstance(type_expr, RefType):
            return OwnershipKind.BORROWED_MUTABLE if type_expr.mutable else OwnershipKind.BORROWED_IMMUTABLE
        if isinstance(type_expr, NamedType):
            symbol = self.current_scope.resolve(type_expr.name)
            if symbol and symbol.kind == SymbolKind.CLASS:
                cls = symbol.node
                if isinstance(cls, ClassDecl):
                    if ClassModifier.LINEAR in cls.modifiers:
                        return OwnershipKind.LINEAR
                    if ClassModifier.ACTOR in cls.modifiers:
                        return OwnershipKind.ACTOR
        if isinstance(type_expr, GcType):
            return OwnershipKind.OWNED
        return OwnershipKind.OWNED if not is_const else OwnershipKind.BORROWED_IMMUTABLE

    def type_to_string(self, type_expr: TypeExpr) -> str:
        if isinstance(type_expr, BasicType):
            return type_expr.name
        elif isinstance(type_expr, RefType):
            mut = "mut " if type_expr.mutable else ""
            return f"&{mut}{self.type_to_string(type_expr.inner)}"
        elif isinstance(type_expr, ArrayType):
            size = f", {self.expr_to_string(type_expr.size)}" if type_expr.size else ""
            return f"array[{self.type_to_string(type_expr.element_type)}{size}]"
        elif isinstance(type_expr, NamedType):
            if type_expr.generics:
                generics = ", ".join(self.type_to_string(g) for g in type_expr.generics)
                return f"{type_expr.name}<{generics}>"
            return type_expr.name
        return "unknown"

    def expr_to_string(self, expr: Optional[Expression]) -> str:
        if expr is None:
            return ""
        if isinstance(expr, LiteralExpr):
            return str(expr.value)
        if isinstance(expr, IdentifierExpr):
            return expr.name
        return "expr"

    def check_unhandled_errors(self) -> None:
        for expr, token in self.required_error_handlers:
            if not self.is_handled_by_parent(expr):
                raise SemanticError("Unhandled Result value (Article I.12: explicit error handling required)", token)

    def is_handled_by_parent(self, expr: Expression) -> bool:
        parent = self.find_parent_expression(expr)
        return parent is not None and isinstance(parent, (ErrorPropagateExpr, IfStmt, SwitchStmt))

    def find_parent_expression(self, expr: Expression) -> Optional[Expression]:
        return None

    def check_linear_consumption(self) -> None:
        for var_name, token in self.linear_vars.items():
            raise SemanticError(f"Linear variable '{var_name}' not consumed before scope end (Article I.13: mandatory disposal)", token)