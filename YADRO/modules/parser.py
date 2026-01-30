#================================================#
#  YY   YY    AAA      DDDDD    RRRR     OOOO    #
#   YY YY    AA AA     DD  DD   R  RR   OO  OO   #
#    YYY    AA   AA    DD   DD  R RR    OO  OO   #
#     Y    AAAAAAAAA   DD  DD   R  RR   OO  OO   #
#     Y   AA       AA  DDDDD    R   RR   OOOO    #
#================================================#

#  Parser module for YADRO compiler   

#  version 0.2.0 
#  Made by CyrOil


from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Union
from lexer import Token, TokenType, YadroLexer, LexerError

class ASTNode:
    def __init__(self, token: Optional[Token] = None):
        self.token = token
        self.constitutional_checks: List[str] = []

@dataclass
class Program(ASTNode):
    directives: List['Directive']
    statements: List['Statement']

class Directive(ASTNode):
    pass

@dataclass
class TargetDirective(Directive):
    os: str
    arch: str

@dataclass
class ImportDirective(Directive):
    module_path: str
    alias: Optional[str] = None

@dataclass
class RequiresDirective(Directive):
    library: str

@dataclass
class PluginDirective(Directive):
    name: str
    version: Optional[str] = None

@dataclass
class DefineDirective(Directive):
    name: str
    type_expr: Optional['TypeExpr'] = None
    value: Optional['Expression'] = None

class TypeExpr(ASTNode):
    pass

@dataclass
class BasicType(TypeExpr):
    name: str

@dataclass
class ArrayType(TypeExpr):
    element_type: TypeExpr
    size: Optional['Expression'] = None

@dataclass
class DictType(TypeExpr):
    key_type: TypeExpr
    value_type: TypeExpr

@dataclass
class SetType(TypeExpr):
    element_type: TypeExpr

@dataclass
class VectorType(TypeExpr):
    element_type: TypeExpr
    dimensions: int

@dataclass
class RefType(TypeExpr):
    mutable: bool
    inner: TypeExpr

@dataclass
class GcType(TypeExpr):
    inner: TypeExpr

@dataclass
class ResultType(TypeExpr):
    ok_type: TypeExpr
    err_type: TypeExpr

@dataclass
class OptionType(TypeExpr):
    inner: TypeExpr

@dataclass
class PredicatedType(TypeExpr):
    base_type: TypeExpr
    predicate: Optional['PredicateExpr'] = None

@dataclass
class GenericType(TypeExpr):
    name: str
    bounds: List[str] = field(default_factory=list)

@dataclass
class NamedType(TypeExpr):
    name: str
    generics: List[TypeExpr] = field(default_factory=list)

class PredicateExpr(ASTNode):
    pass

@dataclass
class ValuePredicate(PredicateExpr):
    operator: str
    rhs: 'Expression'

@dataclass
class LengthPredicate(PredicateExpr):
    operator: str
    rhs: 'Expression'

class Expression(ASTNode):
    type_annotation: Optional[TypeExpr] = None

@dataclass
class LiteralExpr(Expression):
    value: Union[int, float, str, bool, None]
    literal_type: TokenType

@dataclass
class IdentifierExpr(Expression):
    name: str

@dataclass
class BinaryExpr(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryExpr(Expression):
    operator: str
    operand: Expression

@dataclass
class AssignmentExpr(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class PipelineExpr(Expression):
    direction: str
    stages: List[Expression]

@dataclass
class CallExpr(Expression):
    callee: Expression
    args: List[Expression]
    type_args: List[TypeExpr] = field(default_factory=list)

@dataclass
class IndexExpr(Expression):
    container: Expression
    index: Expression

@dataclass
class MemberExpr(Expression):
    object: Expression
    member: str

@dataclass
class ErrorPropagateExpr(Expression):
    expr: Expression

class Statement(ASTNode):
    pass

@dataclass
class ExprStmt(Statement):
    expr: Expression

@dataclass
class ReturnStmt(Statement):
    value: Optional[Expression] = None

@dataclass
class DelStmt(Statement):
    target: Expression

@dataclass
class VariableDecl(Statement):
    name: str
    type_expr: Optional[TypeExpr] = None
    initializer: Optional[Expression] = None
    is_const: bool = False

@dataclass
class BlockStmt(Statement):
    statements: List[Statement]

@dataclass
class IfStmt(Statement):
    condition: Expression
    then_branch: BlockStmt
    elif_branches: List[tuple[Expression, BlockStmt]] = field(default_factory=list)
    else_branch: Optional[BlockStmt] = None

@dataclass
class SwitchStmt(Statement):
    expr: Expression
    cases: List[tuple[Expression, BlockStmt]]
    default_case: Optional[BlockStmt] = None

@dataclass
class ForStmt(Statement):
    var_name: str
    var_type: TypeExpr
    iterable: Expression
    body: BlockStmt

@dataclass
class WhileStmt(Statement):
    condition: Expression
    body: BlockStmt

@dataclass
class RepeatUntilStmt(Statement):
    body: BlockStmt
    condition: Expression

class FunctionModifier(Enum):
    ASYNC = auto()
    THREAD = auto()
    CONST = auto()
    GPU = auto()
    CGPU = auto()
    CORO = auto()
    NONRET = auto()
    UNSAFE = auto()

@dataclass
class Parameter:
    name: str
    type_expr: TypeExpr
    default_value: Optional[Expression] = None
    is_self: bool = False

@dataclass
class FunctionDecl(Statement):
    name: str
    params: List[Parameter]
    return_type: Optional[TypeExpr] = None
    body: Optional[BlockStmt] = None
    modifiers: List[FunctionModifier] = field(default_factory=list)
    is_method: bool = False
    is_static: bool = False
    generics: List[GenericType] = field(default_factory=list)

class ClassModifier(Enum):
    LINEAR = auto()
    MODULE = auto()
    ACTOR = auto()

@dataclass
class ClassDecl(Statement):
    name: str
    parent: Optional[str] = None
    body: List[Statement] = field(default_factory=list)
    modifiers: List[ClassModifier] = field(default_factory=list)
    generics: List[GenericType] = field(default_factory=list)

    def validate_constitutional_constraints(self) -> List[str]:
        violations = []
        if ClassModifier.LINEAR in self.modifiers and ClassModifier.ACTOR in self.modifiers:
            violations.append("Constitutional violation: [linear] and [actor] are mutually exclusive (linear = single-threaded ownership, actor = concurrent isolation)")
        if ClassModifier.MODULE in self.modifiers and self.parent:
            violations.append("Constitutional violation: [module] classes cannot inherit (modules are namespaces, not object hierarchies)")
        return violations

@dataclass
class TraitDecl(Statement):
    name: str
    methods: List[FunctionDecl]
    generics: List[GenericType] = field(default_factory=list)

@dataclass
class ImplDecl(Statement):
    trait_name: str
    type_name: str
    methods: List[FunctionDecl]

@dataclass
class UnsafeBlock(Statement):
    body: BlockStmt
    justification: Optional[str] = None

class ParseError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        loc = f" @{token.line}:{token.column}" if token else ""
        super().__init__(f"Parse Error{loc}: {message}")
        self.token = token

class YadroParser:
    class Precedence(Enum):
        LOWEST = 0
        ASSIGNMENT = 1
        LOGICAL_OR = 2
        LOGICAL_AND = 3
        EQUALITY = 4
        COMPARISON = 5
        BITWISE_OR = 6
        BITWISE_XOR = 7
        BITWISE_AND = 8
        SHIFT = 9
        ADDITIVE = 10
        MULTIPLICATIVE = 11
        UNARY = 12
        PIPELINE = 13
        CALL = 14

    PRECEDENCE_MAP = {
        TokenType.OP_ASSIGN: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_ADD: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_SUB: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_MUL: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_DIV: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_MOD: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_POW: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_BIT_OR: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_BIT_AND: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_BIT_XOR: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_LSHIFT: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_RSHIFT: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_IF_GT: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_IF_LT: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_IF_NE: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_ADDR: Precedence.ASSIGNMENT,
        TokenType.OP_ASSIGN_SWAP: Precedence.ASSIGNMENT,
        TokenType.OP_LOGICAL_OR: Precedence.LOGICAL_OR,
        TokenType.OP_LOGICAL_AND: Precedence.LOGICAL_AND,
        TokenType.OP_EQ: Precedence.EQUALITY,
        TokenType.OP_NE: Precedence.EQUALITY,
        TokenType.OP_LT: Precedence.COMPARISON,
        TokenType.OP_GT: Precedence.COMPARISON,
        TokenType.OP_LTE: Precedence.COMPARISON,
        TokenType.OP_GTE: Precedence.COMPARISON,
        TokenType.OP_BIT_OR: Precedence.BITWISE_OR,
        TokenType.OP_BIT_XOR: Precedence.BITWISE_XOR,
        TokenType.OP_BIT_AND: Precedence.BITWISE_AND,
        TokenType.OP_LSHIFT: Precedence.SHIFT,
        TokenType.OP_RSHIFT: Precedence.SHIFT,
        TokenType.OP_ADD: Precedence.ADDITIVE,
        TokenType.OP_SUB: Precedence.ADDITIVE,
        TokenType.OP_MUL: Precedence.MULTIPLICATIVE,
        TokenType.OP_DIV: Precedence.MULTIPLICATIVE,
        TokenType.OP_FDIV: Precedence.MULTIPLICATIVE,
        TokenType.OP_MOD: Precedence.MULTIPLICATIVE,
        TokenType.OP_POW: Precedence.MULTIPLICATIVE,
        TokenType.OP_PIPELINE_FWD: Precedence.PIPELINE,
        TokenType.OP_PIPELINE_BWD: Precedence.PIPELINE,
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0

    def parse_program(self) -> Program:
        directives: List[Directive] = []
        statements: List[Statement] = []
        while self.match_any(
            TokenType.DIRECTIVE_TARGET,
            TokenType.DIRECTIVE_IMPORT,
            TokenType.DIRECTIVE_REQUIRES,
            TokenType.DIRECTIVE_PLUGIN,
            TokenType.DIRECTIVE_DEFINE
        ):
            directives.append(self.parse_directive())
        while not self.check(TokenType.EOF):
            if self.match(TokenType.NEWLINE):
                continue
            statements.append(self.parse_statement())
        program = Program(directives=directives, statements=statements)
        self.validate_program_structure(program)
        return program

    def parse_directive(self) -> Directive:
        if self.match(TokenType.DIRECTIVE_TARGET):
            return self.parse_target_directive()
        elif self.match(TokenType.DIRECTIVE_IMPORT):
            return self.parse_import_directive()
        elif self.match(TokenType.DIRECTIVE_REQUIRES):
            return self.parse_requires_directive()
        elif self.match(TokenType.DIRECTIVE_PLUGIN):
            return self.parse_plugin_directive()
        elif self.match(TokenType.DIRECTIVE_DEFINE):
            return self.parse_define_directive()
        else:
            token = self.previous()
            raise ParseError(f"Unexpected directive: {token.lexeme}", token)

    def parse_target_directive(self) -> TargetDirective:
        self.consume(TokenType.NEWLINE, "Expected newline after #target")
        self.consume(TokenType.INDENT, "Expected indented target fields")
        os_key = self.consume(TokenType.IDENTIFIER, "Expected 'os' field").lexeme
        if os_key != "os":
            raise ParseError(f"Expected 'os' field in #target, got '{os_key}'", self.previous())
        self.consume(TokenType.OP_ASSIGN, "Expected '=' after os")
        os_val_token = self.consume(TokenType.LITERAL_STRING, "Expected string value for os")
        os_val = os_val_token.value if isinstance(os_val_token.value, str) else ""
        self.consume(TokenType.NEWLINE, "Expected newline after os value")
        arch_key = self.consume(TokenType.IDENTIFIER, "Expected 'arch' field").lexeme
        if arch_key != "arch":
            raise ParseError(f"Expected 'arch' field in #target, got '{arch_key}'", self.previous())
        self.consume(TokenType.OP_ASSIGN, "Expected '=' after arch")
        arch_val_token = self.consume(TokenType.LITERAL_STRING, "Expected string value for arch")
        arch_val = arch_val_token.value if isinstance(arch_val_token.value, str) else ""
        self.consume(TokenType.NEWLINE, "Expected newline after arch value")
        self.consume(TokenType.DEDENT, "Expected dedent after target block")
        return TargetDirective(os=os_val, arch=arch_val)

    def parse_import_directive(self) -> ImportDirective:
        self.consume(TokenType.NEWLINE, "Expected newline after #import")
        self.consume(TokenType.INDENT, "Expected indented import path")
        path_token = self.consume(TokenType.IDENTIFIER, "Expected module path")
        module_path = path_token.lexeme
        alias = None
        if self.match(TokenType.IDENTIFIER) and self.previous().lexeme == "as":
            alias_token = self.consume(TokenType.IDENTIFIER, "Expected alias name")
            alias = alias_token.lexeme
        self.consume(TokenType.NEWLINE, "Expected newline after import")
        self.consume(TokenType.DEDENT, "Expected dedent after import")
        return ImportDirective(module_path=module_path, alias=alias)

    def parse_requires_directive(self) -> RequiresDirective:
        self.consume(TokenType.NEWLINE, "Expected newline after #requires")
        self.consume(TokenType.INDENT, "Expected indented library name")
        lib_token = self.consume(TokenType.LITERAL_STRING, "Expected library name in quotes")
        lib_name = lib_token.value if isinstance(lib_token.value, str) else lib_token.lexeme.strip('"').strip("'")
        self.consume(TokenType.NEWLINE, "Expected newline after library")
        self.consume(TokenType.DEDENT, "Expected dedent after requires")
        return RequiresDirective(library=lib_name)

    def parse_plugin_directive(self) -> PluginDirective:
        self.consume(TokenType.NEWLINE, "Expected newline after #plugin")
        self.consume(TokenType.INDENT, "Expected indented plugin name")
        name_token = self.consume(TokenType.IDENTIFIER, "Expected plugin name")
        name = name_token.lexeme
        version = None
        if self.match(TokenType.OP_ASSIGN):
            if self.match(TokenType.OP_ASSIGN):
                ver_token = self.consume(TokenType.LITERAL_STRING, "Expected version string")
                version = ver_token.lexeme.strip('"').strip("'")
        self.consume(TokenType.NEWLINE, "Expected newline after plugin")
        self.consume(TokenType.DEDENT, "Expected dedent after plugin")
        return PluginDirective(name=name, version=version)

    def parse_define_directive(self) -> DefineDirective:
        self.consume(TokenType.NEWLINE, "Expected newline after #define")
        self.consume(TokenType.INDENT, "Expected indented definition")
        type_expr = None
        if self.check_type_start():
            type_expr = self.parse_type()
        name_token = self.consume(TokenType.IDENTIFIER, "Expected identifier in #define")
        name = name_token.lexeme
        value = None
        if self.match(TokenType.OP_ASSIGN):
            value = self.parse_expression()
        self.consume(TokenType.NEWLINE, "Expected newline after define")
        self.consume(TokenType.DEDENT, "Expected dedent after define")
        return DefineDirective(name=name, type_expr=type_expr, value=value)

    def parse_statement(self) -> Statement:
        if self.match(TokenType.KEYWORD_IF):
            return self.parse_if_stmt()
        elif self.match(TokenType.KEYWORD_SWITCH):
            return self.parse_switch_stmt()
        elif self.match(TokenType.KEYWORD_FOR):
            return self.parse_for_stmt()
        elif self.match(TokenType.KEYWORD_WHILE):
            return self.parse_while_stmt()
        elif self.match(TokenType.KEYWORD_REPEAT):
            return self.parse_repeat_until_stmt()
        elif self.match(TokenType.KEYWORD_FUN):
            return self.parse_function_decl(is_method=False)
        elif self.match(TokenType.KEYWORD_CLASS):
            return self.parse_class_decl()
        elif self.match(TokenType.KEYWORD_TRAIT):
            return self.parse_trait_decl()
        elif self.match(TokenType.KEYWORD_IMPL):
            return self.parse_impl_decl()
        elif self.match(TokenType.KEYWORD_CONST) or self.check_type_start():
            return self.parse_variable_decl()
        elif self.match(TokenType.KEYWORD_DEL):
            return self.parse_del_stmt()
        elif self.match(TokenType.MODIFIER_UNSAFE):
            return self.parse_unsafe_block()
        else:
            expr = self.parse_expression()
            self.consume(TokenType.NEWLINE, "Expected newline after statement")
            return ExprStmt(expr=expr)

    def parse_if_stmt(self) -> IfStmt:
        cond = self.parse_expression()
        self.consume(TokenType.COLON, "Expected ':' after if condition")
        self.consume(TokenType.NEWLINE, "Expected newline after if colon")
        then_branch = self.parse_block()
        elif_branches = []
        while self.match(TokenType.KEYWORD_ELSIF):
            elif_cond = self.parse_expression()
            self.consume(TokenType.COLON, "Expected ':' after elsif condition")
            self.consume(TokenType.NEWLINE, "Expected newline after elsif colon")
            elif_body = self.parse_block()
            elif_branches.append((elif_cond, elif_body))
        else_branch = None
        if self.match(TokenType.KEYWORD_ELSE):
            self.consume(TokenType.COLON, "Expected ':' after else")
            self.consume(TokenType.NEWLINE, "Expected newline after else colon")
            else_branch = self.parse_block()
        return IfStmt(condition=cond, then_branch=then_branch, elif_branches=elif_branches, else_branch=else_branch)

    def parse_switch_stmt(self) -> SwitchStmt:
        expr = self.parse_expression()
        self.consume(TokenType.COLON, "Expected ':' after switch expression")
        self.consume(TokenType.NEWLINE, "Expected newline after switch colon")
        self.consume(TokenType.INDENT, "Expected indented cases")
        cases: List[tuple[Expression, BlockStmt]] = []
        default_case: Optional[BlockStmt] = None
        while not self.check(TokenType.DEDENT) and not self.check(TokenType.EOF):
            if self.match(TokenType.KEYWORD_CASE):
                pattern = self.parse_expression()
                self.consume(TokenType.COLON, "Expected ':' after case pattern")
                self.consume(TokenType.NEWLINE, "Expected newline after case colon")
                body = self.parse_block()
                cases.append((pattern, body))
            elif self.match(TokenType.KEYWORD_DEFAULT):
                self.consume(TokenType.COLON, "Expected ':' after default")
                self.consume(TokenType.NEWLINE, "Expected newline after default colon")
                default_case = self.parse_block()
            else:
                break
        self.consume(TokenType.DEDENT, "Expected dedent after switch cases")
        return SwitchStmt(expr=expr, cases=cases, default_case=default_case)

    def parse_for_stmt(self) -> ForStmt:
        var_type = self.parse_type()
        var_token = self.consume(TokenType.IDENTIFIER, "Expected loop variable name")
        var_name = var_token.lexeme
        self.consume_identifier("in", "Expected 'in' after loop variable")
        iterable = self.parse_expression()
        self.consume(TokenType.COLON, "Expected ':' after for iterable")
        self.consume(TokenType.NEWLINE, "Expected newline after for colon")
        body = self.parse_block()
        return ForStmt(var_name=var_name, var_type=var_type, iterable=iterable, body=body)

    def parse_while_stmt(self) -> WhileStmt:
        cond = self.parse_expression()
        self.consume(TokenType.COLON, "Expected ':' after while condition")
        self.consume(TokenType.NEWLINE, "Expected newline after while colon")
        body = self.parse_block()
        return WhileStmt(condition=cond, body=body)

    def parse_repeat_until_stmt(self) -> RepeatUntilStmt:
        self.consume(TokenType.COLON, "Expected ':' after repeat")
        self.consume(TokenType.NEWLINE, "Expected newline after repeat colon")
        body = self.parse_block()
        self.consume(TokenType.KEYWORD_UNTIL, "Expected 'until' after repeat block")
        cond = self.parse_expression()
        self.consume(TokenType.NEWLINE, "Expected newline after until condition")
        return RepeatUntilStmt(body=body, condition=cond)

    def parse_function_decl(self, is_method: bool) -> FunctionDecl:
        modifiers: List[FunctionModifier] = []
        while self.check_any(
            TokenType.MODIFIER_ASYNC, TokenType.MODIFIER_THREAD,
            TokenType.MODIFIER_CONST, TokenType.MODIFIER_GPU,
            TokenType.MODIFIER_CGPU, TokenType.MODIFIER_CORO,
            TokenType.MODIFIER_NONRET
        ):
            mod_token = self.advance()
            if mod_token.type == TokenType.MODIFIER_ASYNC:
                modifiers.append(FunctionModifier.ASYNC)
            elif mod_token.type == TokenType.MODIFIER_THREAD:
                modifiers.append(FunctionModifier.THREAD)
            elif mod_token.type == TokenType.MODIFIER_CONST:
                modifiers.append(FunctionModifier.CONST)
            elif mod_token.type == TokenType.MODIFIER_GPU:
                modifiers.append(FunctionModifier.GPU)
            elif mod_token.type == TokenType.MODIFIER_CGPU:
                modifiers.append(FunctionModifier.CGPU)
            elif mod_token.type == TokenType.MODIFIER_CORO:
                modifiers.append(FunctionModifier.CORO)
            elif mod_token.type == TokenType.MODIFIER_NONRET:
                modifiers.append(FunctionModifier.NONRET)
        if self.match(TokenType.MODIFIER_UNSAFE):
            modifiers.append(FunctionModifier.UNSAFE)
        name_token = self.consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.lexeme
        generics: List[GenericType] = []
        if self.match(TokenType.KEYWORD_TEMP):
            self.consume(TokenType.LT, "Expected '<' after temp")
            while not self.check(TokenType.GT) and not self.is_at_end():
                gen_name = self.consume(TokenType.IDENTIFIER, "Expected generic name").lexeme
                generics.append(GenericType(name=gen_name))
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.GT, "Expected '>' after generics")
        self.consume(TokenType.LPAREN, "Expected '(' before parameters")
        params = self.parse_parameters()
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        return_type = None
        if self.match(TokenType.OP_DEREF):
            if self.match(TokenType.GT):
                return_type = self.parse_type()
            else:
                self.current -= 1
                self.column -= 1
        self.consume(TokenType.COLON, "Expected ':' before function body")
        self.consume(TokenType.NEWLINE, "Expected newline after function colon")
        body = self.parse_block() if not self.check(TokenType.DEDENT) else None
        return FunctionDecl(name=name, params=params, return_type=return_type, body=body, modifiers=modifiers, is_method=is_method, generics=generics)

    def parse_parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        if self.check(TokenType.RPAREN):
            return params
        while True:
            if self.check(TokenType.IDENTIFIER):
                next_token = self.peek()
                if next_token.lexeme == "self":
                    self.advance()
                    params.append(Parameter(name="self", type_expr=None, is_self=True))
                    if not self.match(TokenType.COMMA):
                        break
                    continue
            param_type = self.parse_type()
            name_token = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
            name = name_token.lexeme
            default_value = None
            if self.match(TokenType.OP_ASSIGN):
                default_value = self.parse_expression()
            params.append(Parameter(name=name, type_expr=param_type, default_value=default_value, is_self=False))
            if not self.match(TokenType.COMMA):
                break
        return params

    def parse_class_decl(self) -> ClassDecl:
        modifiers: List[ClassModifier] = []
        while self.check_any(
            TokenType.MODIFIER_LINEAR, TokenType.MODIFIER_MODULE, TokenType.MODIFIER_ACTOR
        ):
            mod_token = self.advance()
            if mod_token.type == TokenType.MODIFIER_LINEAR:
                modifiers.append(ClassModifier.LINEAR)
            elif mod_token.type == TokenType.MODIFIER_MODULE:
                modifiers.append(ClassModifier.MODULE)
            elif mod_token.type == TokenType.MODIFIER_ACTOR:
                modifiers.append(ClassModifier.ACTOR)
        name_token = self.consume(TokenType.IDENTIFIER, "Expected class name")
        name = name_token.lexeme
        parent = None
        if self.match(TokenType.LPAREN):
            parent_token = self.consume(TokenType.IDENTIFIER, "Expected parent class name")
            parent = parent_token.lexeme
            self.consume(TokenType.RPAREN, "Expected ')' after parent class")
        self.consume(TokenType.COLON, "Expected ':' before class body")
        self.consume(TokenType.NEWLINE, "Expected newline after class colon")
        self.consume(TokenType.INDENT, "Expected indented class body")
        body: List[Statement] = []
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            if self.match(TokenType.KEYWORD_FUN):
                body.append(self.parse_function_decl(is_method=True))
            elif self.match(TokenType.KEYWORD_CONST):
                body.append(self.parse_variable_decl())
            elif self.check_type_start():
                body.append(self.parse_variable_decl())
            else:
                token = self.peek()
                raise ParseError(f"Unexpected token in class body: {token.lexeme}", token)
        self.consume(TokenType.DEDENT, "Expected dedent after class body")
        decl = ClassDecl(name=name, parent=parent, body=body, modifiers=modifiers)
        violations = decl.validate_constitutional_constraints()
        if violations:
            raise ParseError("; ".join(violations), self.previous())
        return decl

    def parse_trait_decl(self) -> TraitDecl:
        name_token = self.consume(TokenType.IDENTIFIER, "Expected trait name")
        name = name_token.lexeme
        generics: List[GenericType] = []
        if self.match(TokenType.LT):
            while not self.check(TokenType.GT) and not self.is_at_end():
                gen_name = self.consume(TokenType.IDENTIFIER, "Expected generic name").lexeme
                generics.append(GenericType(name=gen_name))
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.GT, "Expected '>' after trait generics")
        self.consume(TokenType.COLON, "Expected ':' before trait body")
        self.consume(TokenType.NEWLINE, "Expected newline after trait colon")
        self.consume(TokenType.INDENT, "Expected indented trait body")
        methods: List[FunctionDecl] = []
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            if self.match(TokenType.KEYWORD_FUN):
                methods.append(self.parse_function_decl(is_method=True))
            else:
                token = self.peek()
                raise ParseError(f"Only methods allowed in trait body", token)
        self.consume(TokenType.DEDENT, "Expected dedent after trait body")
        return TraitDecl(name=name, methods=methods, generics=generics)

    def parse_impl_decl(self) -> ImplDecl:
        trait_token = self.consume(TokenType.IDENTIFIER, "Expected trait name after 'impl'")
        trait_name = trait_token.lexeme
        self.consume_identifier("for", "Expected 'for' after trait name")
        type_token = self.consume(TokenType.IDENTIFIER, "Expected type name after 'for'")
        type_name = type_token.lexeme
        self.consume(TokenType.COLON, "Expected ':' before impl body")
        self.consume(TokenType.NEWLINE, "Expected newline after impl colon")
        self.consume(TokenType.INDENT, "Expected indented impl body")
        methods: List[FunctionDecl] = []
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            if self.match(TokenType.KEYWORD_FUN):
                methods.append(self.parse_function_decl(is_method=True))
            else:
                token = self.peek()
                raise ParseError(f"Only method implementations allowed in impl block", token)
        self.consume(TokenType.DEDENT, "Expected dedent after impl body")
        return ImplDecl(trait_name=trait_name, type_name=type_name, methods=methods)

    def parse_variable_decl(self) -> VariableDecl:
        is_const = False
        if self.match(TokenType.KEYWORD_CONST):
            is_const = True
        type_expr = self.parse_type()
        name_token = self.consume(TokenType.IDENTIFIER, "Expected variable name")
        name = name_token.lexeme
        initializer = None
        if self.match(TokenType.OP_ASSIGN):
            initializer = self.parse_expression()
        self.consume(TokenType.NEWLINE, "Expected newline after variable declaration")
        return VariableDecl(name=name, type_expr=type_expr, initializer=initializer, is_const=is_const)

    def parse_del_stmt(self) -> DelStmt:
        target = self.parse_expression()
        self.consume(TokenType.NEWLINE, "Expected newline after 'del'")
        return DelStmt(target=target)

    def parse_unsafe_block(self) -> UnsafeBlock:
        self.consume(TokenType.COLON, "Expected ':' after #[unsafe]")
        self.consume(TokenType.NEWLINE, "Expected newline after unsafe colon")
        body = self.parse_block()
        return UnsafeBlock(body=body)

    def parse_block(self) -> BlockStmt:
        self.consume(TokenType.INDENT, "Expected indented block")
        statements: List[Statement] = []
        while not self.check(TokenType.DEDENT) and not self.is_at_end():
            if self.match(TokenType.NEWLINE):
                continue
            try:
                statements.append(self.parse_statement())
            except ParseError as e:
                while not self.check_any(TokenType.NEWLINE, TokenType.DEDENT, TokenType.EOF):
                    self.advance()
                if self.match(TokenType.NEWLINE):
                    continue
        self.consume(TokenType.DEDENT, "Expected dedent after block")
        return BlockStmt(statements=statements)

    def parse_expression(self, precedence: int = Precedence.LOWEST.value) -> Expression:
        if self.match(TokenType.OP_SUB) or self.match(TokenType.OP_BIT_NOT):
            op = self.previous().lexeme
            operand = self.parse_expression(self.Precedence.UNARY.value)
            return UnaryExpr(operator=op, operand=operand, token=self.previous())
        if self.match(TokenType.PREDICATE_TILDE):
            return self.parse_predicate_expr()
        if self.match(TokenType.OP_PIPELINE_FWD):
            return self.parse_pipeline_expr(">>>")
        if self.match(TokenType.OP_PIPELINE_BWD):
            return self.parse_pipeline_expr("<<<")
        left = self.parse_primary_expression()
        while precedence < self.current_precedence():
            left = self.parse_infix_expression(left)
        return left

    def parse_primary_expression(self) -> Expression:
        if self.match_any(
            TokenType.LITERAL_INT, TokenType.LITERAL_FLOAT, TokenType.LITERAL_STRING,
            TokenType.LITERAL_CHAR, TokenType.LITERAL_BOOL
        ):
            token = self.previous()
            return LiteralExpr(value=token.value, literal_type=token.type, token=token)
        if self.match(TokenType.IDENTIFIER):
            return IdentifierExpr(name=self.previous().lexeme, token=self.previous())
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        if self.match(TokenType.LBRACKET):
            return self.parse_array_literal()
        if self.match(TokenType.LBRACE):
            return self.parse_collection_literal()
        token = self.peek()
        raise ParseError(f"Expected expression, got '{token.lexeme}'", token)

    def parse_pipeline_expr(self, direction: str) -> PipelineExpr:
        stages: List[Expression] = []
        stages.append(self.parse_expression(self.Precedence.PIPELINE.value - 1))
        while self.match(TokenType.OP_PIPELINE_FWD if direction == ">>>" else TokenType.OP_PIPELINE_BWD):
            stages.append(self.parse_expression(self.Precedence.PIPELINE.value - 1))
        return PipelineExpr(direction=direction, stages=stages, token=self.previous())

    def parse_predicate_expr(self) -> PredicateExpr:
        ident = self.consume(TokenType.IDENTIFIER, "Expected 'value' or 'len' after ~").lexeme
        if ident == "value":
            op = self.parse_comparison_operator()
            rhs = self.parse_expression(self.Precedence.COMPARISON.value)
            return ValuePredicate(operator=op, rhs=rhs, token=self.previous())
        elif ident == "len":
            op = self.parse_comparison_operator()
            rhs = self.parse_expression(self.Precedence.COMPARISON.value)
            return LengthPredicate(operator=op, rhs=rhs, token=self.previous())
        else:
            raise ParseError(f"Unknown predicate: ~{ident} (expected 'value' or 'len')", self.previous())

    def parse_comparison_operator(self) -> str:
        if self.match_any(TokenType.OP_EQ, TokenType.OP_NE, TokenType.OP_LT,
                          TokenType.OP_GT, TokenType.OP_LTE, TokenType.OP_GTE):
            return self.previous().lexeme
        token = self.peek()
        raise ParseError(f"Expected comparison operator, got '{token.lexeme}'", token)

    def parse_infix_expression(self, left: Expression) -> Expression:
        op_token = self.previous()
        precedence = self.token_precedence(op_token).value
        if self.is_assignment_operator(op_token.type):
            right = self.parse_expression(precedence - 1)
            return AssignmentExpr(left=left, operator=op_token.lexeme, right=right, token=op_token)
        if op_token.type == TokenType.OPERATOR_QUESTION:
            return ErrorPropagateExpr(expr=left, token=op_token)
        right = self.parse_expression(precedence)
        return BinaryExpr(left=left, operator=op_token.lexeme, right=right, token=op_token)

    def parse_array_literal(self) -> Expression:
        elements: List[Expression] = []
        if not self.check(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                if self.check(TokenType.RBRACKET):
                    break
                elements.append(self.parse_expression())
        self.consume(TokenType.RBRACKET, "Expected ']' after array elements")
        return CallExpr(callee=IdentifierExpr("array"), args=elements, token=self.previous())

    def parse_collection_literal(self) -> Expression:
        if self.check(TokenType.RBRACE):
            self.advance()
            return CallExpr(callee=IdentifierExpr("set"), args=[], token=self.previous())
        first = self.parse_expression()
        if self.match(TokenType.COLON):
            pairs = [(first, self.parse_expression())]
            while self.match(TokenType.COMMA):
                if self.check(TokenType.RBRACE):
                    break
                key = self.parse_expression()
                self.consume(TokenType.COLON, "Expected ':' in dict literal")
                value = self.parse_expression()
                pairs.append((key, value))
            self.consume(TokenType.RBRACE, "Expected '}' after dict")
            args = []
            for k, v in pairs:
                args.append(BinaryExpr(left=k, operator=":", right=v, token=None))
            return CallExpr(callee=IdentifierExpr("dict"), args=args, token=self.previous())
        else:
            elements = [first]
            while self.match(TokenType.COMMA):
                if self.check(TokenType.RBRACE):
                    break
                elements.append(self.parse_expression())
            self.consume(TokenType.RBRACE, "Expected '}' after set")
            return CallExpr(callee=IdentifierExpr("set"), args=elements, token=self.previous())

    def parse_type(self) -> TypeExpr:
        if self.match(TokenType.PREDICATE_TILDE):
            base = self.parse_base_type()
            predicate = None
            if self.match(TokenType.COLON):
                if not self.match(TokenType.PREDICATE_TILDE):
                    raise ParseError("Expected predicate after ':' in refined type", self.previous())
                predicate = self.parse_predicate_expr()
            return PredicatedType(base_type=base, predicate=predicate, token=self.previous())
        return self.parse_base_type()

    def parse_base_type(self) -> TypeExpr:
        if self.match(TokenType.TYPE_REF):
            mutable = False
            if self.match(TokenType.KEYWORD_MUT):
                mutable = True
            inner = self.parse_base_type()
            return RefType(mutable=mutable, inner=inner, token=self.previous())
        if self.match_identifier("gc"):
            self.consume(TokenType.LT, "Expected '<' after gc")
            inner = self.parse_type()
            self.consume(TokenType.GT, "Expected '>' after gc type")
            return GcType(inner=inner, token=self.previous())
        if self.match_identifier("Result"):
            self.consume(TokenType.LT, "Expected '<' after Result")
            ok_type = self.parse_type()
            self.consume(TokenType.COMMA, "Expected ',' in Result type")
            err_type = self.parse_type()
            self.consume(TokenType.GT, "Expected '>' after Result type")
            return ResultType(ok_type=ok_type, err_type=err_type, token=self.previous())
        if self.match_identifier("Option"):
            self.consume(TokenType.LT, "Expected '<' after Option")
            inner = self.parse_type()
            self.consume(TokenType.GT, "Expected '>' after Option type")
            return OptionType(inner=inner, token=self.previous())
        if self.match_identifier("array") or self.match_identifier("darray"):
            is_dynamic = self.previous().lexeme == "darray"
            self.consume(TokenType.LBRACKET, "Expected '[' after array/darray")
            elem_type = self.parse_type()
            size = None
            if not is_dynamic:
                self.consume(TokenType.COMMA, "Expected ',' after array element type")
                size = self.parse_expression()
            self.consume(TokenType.RBRACKET, "Expected ']' after array type")
            return ArrayType(element_type=elem_type, size=size, token=self.previous())
        if self.match_identifier("dict"):
            self.consume(TokenType.LBRACKET, "Expected '[' after dict")
            key_type = self.parse_type()
            self.consume(TokenType.COMMA, "Expected ',' in dict type")
            value_type = self.parse_type()
            self.consume(TokenType.RBRACKET, "Expected ']' after dict type")
            return DictType(key_type=key_type, value_type=value_type, token=self.previous())
        if self.match_identifier("set"):
            self.consume(TokenType.LBRACKET, "Expected '[' after set")
            elem_type = self.parse_type()
            self.consume(TokenType.RBRACKET, "Expected ']' after set type")
            return SetType(element_type=elem_type, token=self.previous())
        if self.match_identifier("vector"):
            self.consume(TokenType.LBRACKET, "Expected '[' after vector")
            elem_type = self.parse_type()
            self.consume(TokenType.COMMA, "Expected ',' in vector type")
            dims_token = self.consume(TokenType.LITERAL_INT, "Expected dimension count")
            dims = int(dims_token.lexeme)
            self.consume(TokenType.RBRACKET, "Expected ']' after vector type")
            return VectorType(element_type=elem_type, dimensions=dims, token=self.previous())
        if self.check(TokenType.IDENTIFIER):
            name = self.advance().lexeme
            generics: List[TypeExpr] = []
            if self.match(TokenType.LT):
                while not self.check(TokenType.GT) and not self.is_at_end():
                    generics.append(self.parse_type())
                    if not self.match(TokenType.COMMA):
                        break
                self.consume(TokenType.GT, "Expected '>' after generic arguments")
            if generics:
                return NamedType(name=name, generics=generics, token=self.previous())
            else:
                basic_types = {"int", "float", "bool", "string", "char", "void", "Unit"}
                if name in basic_types:
                    return BasicType(name=name, token=self.previous())
                return NamedType(name=name, token=self.previous())
        token = self.peek()
        raise ParseError(f"Expected type, got '{token.lexeme}'", token)

    def check_type_start(self) -> bool:
        return self.check_any(
            TokenType.TYPE_INT, TokenType.TYPE_FLOAT, TokenType.TYPE_BOOL,
            TokenType.TYPE_STRING, TokenType.TYPE_CHAR, TokenType.TYPE_VOID,
            TokenType.TYPE_UNIT, TokenType.TYPE_REF, TokenType.PREDICATE_TILDE,
            TokenType.IDENTIFIER
        ) or (self.check(TokenType.IDENTIFIER) and self.peek().lexeme in {
            "int", "float", "bool", "string", "char", "void", "Unit",
            "array", "darray", "dict", "set", "vector", "gc", "Result", "Option", "&"
        })

    def token_precedence(self, token: Token) -> Precedence:
        return self.PRECEDENCE_MAP.get(token.type, self.Precedence.LOWEST)

    def current_precedence(self) -> int:
        return self.token_precedence(self.peek()).value

    def is_assignment_operator(self, token_type: TokenType) -> bool:
        return token_type in {
            TokenType.OP_ASSIGN, TokenType.OP_ASSIGN_ADD, TokenType.OP_ASSIGN_SUB,
            TokenType.OP_ASSIGN_MUL, TokenType.OP_ASSIGN_DIV, TokenType.OP_ASSIGN_MOD,
            TokenType.OP_ASSIGN_POW, TokenType.OP_ASSIGN_BIT_OR, TokenType.OP_ASSIGN_BIT_AND,
            TokenType.OP_ASSIGN_BIT_XOR, TokenType.OP_ASSIGN_LSHIFT, TokenType.OP_ASSIGN_RSHIFT,
            TokenType.OP_ASSIGN_IF_GT, TokenType.OP_ASSIGN_IF_LT, TokenType.OP_ASSIGN_IF_NE,
            TokenType.OP_ASSIGN_ADDR, TokenType.OP_ASSIGN_SWAP
        }

    def match(self, token_type: TokenType) -> bool:
        if self.check(token_type):
            self.advance()
            return True
        return False

    def match_any(self, *token_types: TokenType) -> bool:
        for tt in token_types:
            if self.check(tt):
                self.advance()
                return True
        return False

    def match_identifier(self, expected: str) -> bool:
        if self.check(TokenType.IDENTIFIER) and self.peek().lexeme == expected:
            self.advance()
            return True
        return False

    def consume(self, token_type: TokenType, message: str) -> Token:
        if self.check(token_type):
            return self.advance()
        token = self.peek()
        raise ParseError(f"{message}. Got '{token.lexeme}' instead.", token)

    def consume_identifier(self, expected: str, message: str) -> Token:
        token = self.consume(TokenType.IDENTIFIER, message)
        if token.lexeme != expected:
            raise ParseError(f"{message}. Expected '{expected}', got '{token.lexeme}'.", token)
        return token

    def check(self, token_type: TokenType) -> bool:
        return not self.is_at_end() and self.peek().type == token_type

    def check_any(self, *token_types: TokenType) -> bool:
        return not self.is_at_end() and self.peek().type in token_types

    def advance(self) -> Token:
        if not self.is_at_end():
            self.current += 1
            return self.previous()
        return self.peek()

    def previous(self) -> Token:
        return self.tokens[self.current - 1]

    def peek(self, offset: int = 0) -> Token:
        if self.current + offset >= len(self.tokens):
            return Token(TokenType.EOF, "", self.line(), 0)
        return self.tokens[self.current + offset]

    def is_at_end(self) -> bool:
        return self.peek().type == TokenType.EOF

    def line(self) -> int:
        return self.peek().line if not self.is_at_end() else -1

    def validate_program_structure(self, program: Program):
        if program.directives and not isinstance(program.directives[0], TargetDirective):
            raise ParseError("Constitutional violation (Article II.2): #target must be first directive", program.directives[0].token if program.directives else None)