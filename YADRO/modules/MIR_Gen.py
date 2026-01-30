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
from enum import Enum, auto
from typing import List, Optional, Union, Dict, Set, Tuple
from parser import (
    Program, FunctionDecl, ClassDecl, BlockStmt, IfStmt, ReturnStmt,
    VariableDecl, AssignmentExpr, BinaryExpr, CallExpr, IdentifierExpr,
    LiteralExpr, PipelineExpr, ErrorPropagateExpr, DelStmt,
    BasicType, RefType, ResultType, OptionType, PredicatedType,
    NamedType, ArrayType, DictType, GcType, Parameter,
    ExprStmt, MemberExpr, IndexExpr,
    FunctionModifier as ParserFunctionModifier,
    ClassModifier as ParserClassModifier
)
from semantic_analyzer import (
    OwnershipKind, SymbolTable, SymbolKind
)
from lexer import Token, TokenType

class MirType:
    def __init__(self):
        self.is_linear = False
        self.predicates: List[str] = []

@dataclass
class MirBasicType(MirType):
    name: str
    
    def __str__(self):
        return self.name

@dataclass
class MirPointerType(MirType):
    pointee: MirType
    mutable: bool
    is_borrow: bool = True
    
    def __str__(self):
        mut = "mut " if self.mutable else ""
        borrow = "&" if self.is_borrow else "*"
        return f"{borrow}{mut}{self.pointee}"

@dataclass
class MirStructType(MirType):
    name: str
    fields: List[Tuple[str, MirType]] = field(default_factory=list)
    is_linear: bool = False
    
    def __str__(self):
        fields_str = ", ".join(f"{name}: {typ}" for name, typ in self.fields[:3])
        return f"struct {self.name} {{ {fields_str} }}"

@dataclass
class MirFunctionType(MirType):
    params: List[MirType]
    return_type: MirType
    effects: Set[str] = field(default_factory=set)
    
    def __str__(self):
        params_str = ", ".join(str(p) for p in self.params)
        effects_str = f" !{','.join(sorted(self.effects))}" if self.effects else ""
        return f"fn({params_str}) -> {self.return_type}{effects_str}"

@dataclass
class MirResultType(MirType):
    ok_type: MirType
    err_type: MirType
    
    def __str__(self):
        return f"Result<{self.ok_type}, {self.err_type}>"

@dataclass
class MirOptionType(MirType):
    inner: MirType
    
    def __str__(self):
        return f"Option<{self.inner}>"

class MirOpCode(Enum):
    ALLOC = auto()
    ALLOC_HEAP = auto()
    LOAD = auto()
    STORE = auto()
    MOVE = auto()
    BORROW_IMMUT = auto()
    BORROW_MUT = auto()
    DROP = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    SHL = auto()
    SHR = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    NOT = auto()
    BRANCH = auto()
    JUMP = auto()
    PHI = auto()
    SWITCH = auto()
    CALL = auto()
    CALL_INDIRECT = auto()
    RETURN = auto()
    RESULT_OK = auto()
    RESULT_ERR = auto()
    RESULT_UNWRAP = auto()
    RESULT_IS_OK = auto()
    NOP = auto()
    UNREACHABLE = auto()

@dataclass
class MirOperand:
    name: str
    mir_type: MirType
    ownership: OwnershipKind = OwnershipKind.OWNED
    is_const: bool = False
    
    def __str__(self):
        own = f"({self.ownership.name[0]})" if self.ownership != OwnershipKind.OWNED else ""
        return f"%{self.name}{own}"

@dataclass
class MirInstruction:
    opcode: MirOpCode
    operands: List[MirOperand] = field(default_factory=list)
    result: Optional[MirOperand] = None
    effects: Set[str] = field(default_factory=set)
    token: Optional[Token] = None
    
    def __str__(self):
        ops = ", ".join(str(op) for op in self.operands)
        res = f"{self.result} = " if self.result else ""
        effects = f" !{','.join(sorted(self.effects))}" if self.effects else ""
        return f"  {res}{self.opcode.name.lower()} {ops}{effects}"

@dataclass
class MirBasicBlock:
    name: str
    instructions: List[MirInstruction] = field(default_factory=list)
    predecessors: List['MirBasicBlock'] = field(default_factory=list)
    successors: List['MirBasicBlock'] = field(default_factory=list)
    
    def __str__(self):
        instrs = "\n".join(str(instr) for instr in self.instructions)
        preds = ", ".join(p.name for p in self.predecessors)
        succs = ", ".join(s.name for s in self.successors)
        return f"bb_{self.name}:\n{instrs}\n  // preds: [{preds}] succs: [{succs}]"

@dataclass
class MirFunction:
    name: str
    params: List[MirOperand] = field(default_factory=list)
    return_type: MirType = field(default_factory=lambda: MirBasicType("void"))
    blocks: List[MirBasicBlock] = field(default_factory=list)
    effects: Set[str] = field(default_factory=set)
    is_linear: bool = False
    is_async: bool = False
    is_actor: bool = False
    
    def __str__(self):
        params_str = ", ".join(f"{p}: {p.mir_type}" for p in self.params)
        effects_str = f" !{','.join(sorted(self.effects))}" if self.effects else ""
        blocks_str = "\n\n".join(str(block) for block in self.blocks)
        return f"fn {self.name}({params_str}) -> {self.return_type}{effects_str} {{\n{blocks_str}\n}}"

@dataclass
class MirModule:
    name: str
    functions: List[MirFunction] = field(default_factory=list)
    types: Dict[str, MirStructType] = field(default_factory=dict)
    constants: Dict[str, Tuple[MirType, any]] = field(default_factory=dict)
    
    def __str__(self):
        types_str = "\n".join(f"{name} = {typ}" for name, typ in self.types.items())
        funcs_str = "\n\n".join(str(func) for func in self.functions)
        return f"module {self.name} {{\n\n{types_str}\n\n{funcs_str}\n}}"

class MirGenerationError(Exception):
    def __init__(self, message: str, token: Optional[Token] = None):
        loc = f" @{token.line}:{token.column}" if token else ""
        super().__init__(f"MIR Generation Error{loc}: {message}")
        self.token = token

class MirGenerator:
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.current_function: Optional[MirFunction] = None
        self.current_block: Optional[MirBasicBlock] = None
        self.blocks: List[MirBasicBlock] = []
        self.next_block_id = 0
        self.next_value_id = 0
        self.value_map: Dict[str, MirOperand] = {}
        self.linear_vars: Dict[str, MirOperand] = {}
        self.drop_points: Dict[str, List[MirInstruction]] = {}
    
    def generate_module(self, program: Program) -> MirModule:
        module = MirModule(name="main")
        self.generate_types(program, module)
        for stmt in program.statements:
            if isinstance(stmt, FunctionDecl):
                mir_func = self.generate_function(stmt)
                module.functions.append(mir_func)
        self.insert_linear_cleanup(module)
        return module
    
    def generate_types(self, program: Program, module: MirModule):
        for stmt in program.statements:
            if isinstance(stmt, ClassDecl):
                is_linear = ParserClassModifier.LINEAR in stmt.modifiers
                struct_type = MirStructType(name=stmt.name, is_linear=is_linear)
                for field_stmt in stmt.body:
                    if isinstance(field_stmt, VariableDecl) and field_stmt.type_expr:
                        field_type = self.map_ast_type(field_stmt.type_expr)
                        struct_type.fields.append((field_stmt.name, field_type))
                module.types[stmt.name] = struct_type
    
    def generate_function(self, func: FunctionDecl) -> MirFunction:
        return_type = self.map_ast_type(func.return_type) if func.return_type else MirBasicType("void")
        is_async = ParserFunctionModifier.ASYNC in func.modifiers
        is_actor = ParserFunctionModifier.ACTOR in func.modifiers if hasattr(ParserFunctionModifier, 'ACTOR') else False
        mir_func = MirFunction(name=func.name, return_type=return_type, is_linear=isinstance(return_type, MirStructType) and return_type.is_linear, is_async=is_async, is_actor=is_actor)
        self.current_function = mir_func
        entry_block = self.new_block("entry")
        self.current_block = entry_block
        for param in func.params:
            if param.is_self:
                class_symbol = self.symbol_table.resolve(func.name.split("::")[0] if "::" in func.name else func.name)
                is_linear = False
                if class_symbol and isinstance(class_symbol.node, ClassDecl):
                    is_linear = ParserClassModifier.LINEAR in class_symbol.node.modifiers
                param_type = MirPointerType(pointee=MirStructType(name="Self"), mutable="mut" in param.name if isinstance(param.name, str) else False, is_borrow=not is_linear)
                ownership = OwnershipKind.LINEAR if is_linear else OwnershipKind.BORROWED_MUTABLE if "mut" in param.name else OwnershipKind.BORROWED_IMMUTABLE
            else:
                param_type = self.map_ast_type(param.type_expr) if param.type_expr else MirBasicType("i64")
                ownership = OwnershipKind.OWNED
            param_op = MirOperand(name=f"arg_{param.name}", mir_type=param_type, ownership=ownership)
            mir_func.params.append(param_op)
            self.value_map[param.name] = param_op
            if ownership == OwnershipKind.LINEAR:
                self.linear_vars[param.name] = param_op
        if func.body:
            self.generate_block(func.body)
            if self.current_block and not self.has_terminator(self.current_block):
                if isinstance(return_type, MirBasicType) and return_type.name == "void":
                    self.emit(MirOpCode.RETURN)
                else:
                    raise MirGenerationError(f"Function '{func.name}' missing return statement (Article I.5: complete control flow)", func.token)
        mir_func.blocks = self.blocks
        self.blocks = []
        self.value_map.clear()
        self.linear_vars.clear()
        self.drop_points.clear()
        self.current_function = None
        return mir_func
    
    def generate_block(self, block: BlockStmt):
        scope_id = f"scope_{self.next_value_id}"
        self.next_value_id += 1
        for stmt in block.statements:
            self.generate_statement(stmt)
        self.schedule_scope_cleanup(scope_id)
    
    def generate_statement(self, stmt):
        if isinstance(stmt, VariableDecl):
            self.generate_variable_decl(stmt)
        elif isinstance(stmt, AssignmentExpr):
            self.generate_assignment(stmt)
        elif isinstance(stmt, IfStmt):
            self.generate_if(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.generate_return(stmt)
        elif isinstance(stmt, ExprStmt):
            result = self.generate_expression(stmt.expr)
            if isinstance(result.mir_type, (MirResultType, MirOptionType)):
                raise MirGenerationError(f"Unhandled Result/Option value (Article I.12: explicit error handling required)", stmt.token)
        elif isinstance(stmt, DelStmt):
            self.generate_del(stmt)
    
    def generate_variable_decl(self, decl: VariableDecl):
        var_type = self.map_ast_type(decl.type_expr) if decl.type_expr else MirBasicType("i64")
        var_name = f"var_{decl.name}_{self.next_value_id}"
        self.next_value_id += 1
        if isinstance(var_type, MirStructType) and var_type.is_linear:
            alloc_instr = self.emit(MirOpCode.ALLOC, result=MirOperand(name=var_name, mir_type=MirPointerType(pointee=var_type, mutable=True)))
            self.linear_vars[decl.name] = alloc_instr.result
        elif isinstance(decl.type_expr, GcType):
            alloc_instr = self.emit(MirOpCode.ALLOC_HEAP, result=MirOperand(name=var_name, mir_type=MirPointerType(pointee=var_type, mutable=True, is_borrow=False)))
        else:
            alloc_instr = self.emit(MirOpCode.ALLOC, result=MirOperand(name=var_name, mir_type=MirPointerType(pointee=var_type, mutable=True)))
        self.value_map[decl.name] = alloc_instr.result
        if decl.initializer:
            init_value = self.generate_expression(decl.initializer)
            if init_value.ownership == OwnershipKind.LINEAR:
                self.emit(MirOpCode.MOVE, operands=[init_value, alloc_instr.result])
                if decl.initializer.token and isinstance(decl.initializer, IdentifierExpr):
                    self.linear_vars.pop(decl.initializer.name, None)
            else:
                self.emit(MirOpCode.STORE, operands=[init_value, alloc_instr.result])
    
    def generate_assignment(self, assign: AssignmentExpr):
        rhs = self.generate_expression(assign.right)
        lhs = self.generate_lvalue(assign.left)
        if assign.operator != "=":
            current = self.emit(MirOpCode.LOAD, operands=[lhs], result=MirOperand(name=f"tmp_{self.next_value_id}", mir_type=rhs.mir_type))
            self.next_value_id += 1
            op_map = {"+=": MirOpCode.ADD, "-=": MirOpCode.SUB, "*=": MirOpCode.MUL, "/=": MirOpCode.DIV, "%=": MirOpCode.MOD, "&=": MirOpCode.BIT_AND, "|=": MirOpCode.BIT_OR, "^=": MirOpCode.BIT_XOR, "<<=": MirOpCode.SHL, ">>=": MirOpCode.SHR}
            op = op_map.get(assign.operator)
            if not op:
                raise MirGenerationError(f"Unsupported compound assignment: {assign.operator}", assign.token)
            result = self.emit(op, operands=[current.result, rhs], result=MirOperand(name=f"tmp_{self.next_value_id}", mir_type=rhs.mir_type))
            self.next_value_id += 1
            rhs = result.result
        if rhs.ownership == OwnershipKind.LINEAR:
            self.emit(MirOpCode.MOVE, operands=[rhs, lhs])
            if isinstance(assign.right, IdentifierExpr):
                self.linear_vars.pop(assign.right.name, None)
        else:
            self.emit(MirOpCode.STORE, operands=[rhs, lhs])
    
    def generate_if(self, stmt: IfStmt):
        cond = self.generate_expression(stmt.condition)
        then_block = self.new_block("then")
        else_block = self.new_block("else") if stmt.else_branch else None
        merge_block = self.new_block("merge")
        self.emit(MirOpCode.BRANCH, operands=[cond, MirOperand(name=then_block.name, mir_type=MirBasicType("label")), MirOperand(name=else_block.name if else_block else merge_block.name, mir_type=MirBasicType("label"))])
        self.current_block = then_block
        self.generate_block(stmt.then_branch)
        if not self.has_terminator(self.current_block):
            self.emit(MirOpCode.JUMP, operands=[MirOperand(name=merge_block.name, mir_type=MirBasicType("label"))])
        if else_block:
            self.current_block = else_block
            if stmt.else_branch:
                self.generate_block(stmt.else_branch)
            if not self.has_terminator(self.current_block):
                self.emit(MirOpCode.JUMP, operands=[MirOperand(name=merge_block.name, mir_type=MirBasicType("label"))])
        self.current_block = merge_block
    
    def generate_return(self, stmt: ReturnStmt):
        if stmt.value:
            value = self.generate_expression(stmt.value)
            if value.ownership == OwnershipKind.LINEAR:
                self.emit(MirOpCode.MOVE, operands=[value, MirOperand(name="return", mir_type=value.mir_type)])
            else:
                self.emit(MirOpCode.STORE, operands=[value, MirOperand(name="return_ptr", mir_type=MirPointerType(pointee=value.mir_type, mutable=True))])
        self.emit(MirOpCode.RETURN)
    
    def generate_del(self, stmt: DelStmt):
        target = self.generate_expression(stmt.target)
        if target.ownership != OwnershipKind.LINEAR:
            raise MirGenerationError(f"del only permitted on linear-owned values (Article I.9: explicit disposal)", stmt.token)
        self.emit(MirOpCode.DROP, operands=[target])
        if isinstance(stmt.target, IdentifierExpr):
            self.linear_vars.pop(stmt.target.name, None)
    
    def generate_expression(self, expr) -> MirOperand:
        if isinstance(expr, LiteralExpr):
            return self.generate_literal(expr)
        elif isinstance(expr, IdentifierExpr):
            return self.generate_identifier(expr)
        elif isinstance(expr, BinaryExpr):
            return self.generate_binary(expr)
        elif isinstance(expr, CallExpr):
            return self.generate_call(expr)
        elif isinstance(expr, PipelineExpr):
            return self.generate_pipeline(expr)
        elif isinstance(expr, ErrorPropagateExpr):
            return self.generate_error_propagate(expr)
        elif isinstance(expr, MemberExpr):
            return self.generate_member(expr)
        elif isinstance(expr, IndexExpr):
            return self.generate_index(expr)
        else:
            raise MirGenerationError(f"Unsupported expression: {type(expr).__name__}", expr.token)
    
    def generate_literal(self, expr: LiteralExpr) -> MirOperand:
        value_name = f"const_{self.next_value_id}"
        self.next_value_id += 1
        if expr.literal_type == TokenType.LITERAL_INT:
            mir_type = MirBasicType("i64")
        elif expr.literal_type == TokenType.LITERAL_FLOAT:
            mir_type = MirBasicType("f64")
        elif expr.literal_type == TokenType.LITERAL_BOOL:
            mir_type = MirBasicType("bool")
        elif expr.literal_type == TokenType.LITERAL_STRING:
            mir_type = MirPointerType(pointee=MirBasicType("i8"), mutable=False, is_borrow=True)
        else:
            mir_type = MirBasicType("void")
        operand = MirOperand(name=value_name, mir_type=mir_type, is_const=True)
        self.value_map[value_name] = operand
        return operand
    
    def generate_identifier(self, expr: IdentifierExpr) -> MirOperand:
        if expr.name not in self.value_map:
            raise MirGenerationError(f"Undefined identifier: {expr.name}", expr.token)
        stored_op = self.value_map[expr.name]
        if stored_op.ownership == OwnershipKind.LINEAR and expr.name in self.linear_vars:
            borrow_op = MirOperand(name=f"borrow_{expr.name}_{self.next_value_id}", mir_type=MirPointerType(pointee=stored_op.mir_type, mutable=False), ownership=OwnershipKind.BORROWED_IMMUTABLE)
            self.next_value_id += 1
            self.emit(MirOpCode.BORROW_IMMUT, operands=[stored_op], result=borrow_op)
            return borrow_op
        if isinstance(stored_op.mir_type, MirPointerType):
            loaded_op = MirOperand(name=f"loaded_{expr.name}_{self.next_value_id}", mir_type=stored_op.mir_type.pointee, ownership=stored_op.ownership)
            self.next_value_id += 1
            self.emit(MirOpCode.LOAD, operands=[stored_op], result=loaded_op)
            return loaded_op
        return stored_op
    
    def generate_binary(self, expr: BinaryExpr) -> MirOperand:
        left = self.generate_expression(expr.left)
        right = self.generate_expression(expr.right)
        op_map = {"+": MirOpCode.ADD, "-": MirOpCode.SUB, "*": MirOpCode.MUL, "/": MirOpCode.DIV, "%": MirOpCode.MOD, "==": MirOpCode.EQ, "!=": MirOpCode.NE, "<": MirOpCode.LT, ">": MirOpCode.GT, "<=": MirOpCode.LE, ">=": MirOpCode.GE, "&": MirOpCode.BIT_AND, "|": MirOpCode.BIT_OR, "^": MirOpCode.BIT_XOR, "<<": MirOpCode.SHL, ">>": MirOpCode.SHR, "and": MirOpCode.BIT_AND, "or": MirOpCode.BIT_OR}
        opcode = op_map.get(expr.operator)
        if not opcode:
            raise MirGenerationError(f"Unsupported binary operator: {expr.operator}", expr.token)
        result_op = MirOperand(name=f"binop_{self.next_value_id}", mir_type=left.mir_type if opcode in (MirOpCode.ADD, MirOpCode.SUB, MirOpCode.MUL, MirOpCode.DIV, MirOpCode.MOD) else MirBasicType("bool"))
        self.next_value_id += 1
        self.emit(opcode, operands=[left, right], result=result_op)
        return result_op
    
    def generate_call(self, expr: CallExpr) -> MirOperand:
        callee = self.generate_expression(expr.callee)
        args = []
        for arg_expr in expr.args:
            arg = self.generate_expression(arg_expr)
            args.append(arg)
        return_type = MirBasicType("void")
        effects: Set[str] = set()
        if isinstance(callee.mir_type, MirFunctionType):
            return_type = callee.mir_type.return_type
            effects = callee.mir_type.effects.copy()
        is_linear_return = isinstance(return_type, MirStructType) and return_type.is_linear
        result_op = MirOperand(name=f"call_{self.next_value_id}", mir_type=return_type, ownership=OwnershipKind.LINEAR if is_linear_return else OwnershipKind.OWNED)
        self.next_value_id += 1
        self.emit(MirOpCode.CALL, operands=[callee] + args, result=result_op, effects=effects)
        if is_linear_return:
            self.linear_vars[f"call_result_{self.next_value_id-1}"] = result_op
        return result_op
    
    def generate_pipeline(self, expr: PipelineExpr) -> MirOperand:
        current = self.generate_expression(expr.stages[0])
        stages = expr.stages[1:] if expr.direction == ">>>" else reversed(expr.stages[1:])
        for stage_expr in stages:
            func = self.generate_expression(stage_expr)
            current = self.generate_function_call_with_arg(func, current)
        return current
    
    def generate_function_call_with_arg(self, func: MirOperand, arg: MirOperand) -> MirOperand:
        return_type = MirBasicType("void")
        if isinstance(func.mir_type, MirFunctionType) and func.mir_type.params:
            return_type = func.mir_type.return_type
        result_op = MirOperand(name=f"pipe_{self.next_value_id}", mir_type=return_type)
        self.next_value_id += 1
        self.emit(MirOpCode.CALL, operands=[func, arg], result=result_op)
        return result_op
    
    def generate_error_propagate(self, expr: ErrorPropagateExpr) -> MirOperand:
        inner = self.generate_expression(expr.expr)
        if not isinstance(inner.mir_type, MirResultType):
            raise MirGenerationError("Error propagation requires Result type", expr.token)
        ok_block = self.new_block("ok")
        err_block = self.new_block("err")
        merge_block = self.new_block("merge")
        is_ok = self.emit(MirOpCode.RESULT_IS_OK, operands=[inner], result=MirOperand(name=f"is_ok_{self.next_value_id}", mir_type=MirBasicType("bool")))
        self.next_value_id += 1
        self.emit(MirOpCode.BRANCH, operands=[is_ok.result, MirOperand(name=ok_block.name, mir_type=MirBasicType("label")), MirOperand(name=err_block.name, mir_type=MirBasicType("label"))])
        self.current_block = ok_block
        unwrapped = self.emit(MirOpCode.RESULT_UNWRAP, operands=[inner], result=MirOperand(name=f"unwrapped_{self.next_value_id}", mir_type=inner.mir_type.ok_type))
        self.next_value_id += 1
        self.emit(MirOpCode.JUMP, operands=[MirOperand(name=merge_block.name, mir_type=MirBasicType("label"))])
        self.current_block = err_block
        self.emit(MirOpCode.RETURN, operands=[inner])
        self.current_block = merge_block
        phi = self.emit(MirOpCode.PHI, operands=[unwrapped.result, MirOperand(name="unreachable", mir_type=unwrapped.result.mir_type)], result=MirOperand(name=f"phi_{self.next_value_id}", mir_type=inner.mir_type.ok_type))
        self.next_value_id += 1
        return phi.result
    
    def generate_member(self, expr: MemberExpr) -> MirOperand:
        obj = self.generate_expression(expr.object)
        field_type = MirBasicType("i64")
        result_op = MirOperand(name=f"field_{expr.member}_{self.next_value_id}", mir_type=field_type, ownership=OwnershipKind.BORROWED_IMMUTABLE)
        self.next_value_id += 1
        self.emit(MirOpCode.LOAD, operands=[obj], result=result_op)
        return result_op
    
    def generate_index(self, expr: IndexExpr) -> MirOperand:
        container = self.generate_expression(expr.container)
        index = self.generate_expression(expr.index)
        elem_type = MirBasicType("i64")
        result_op = MirOperand(name=f"elem_{self.next_value_id}", mir_type=elem_type, ownership=OwnershipKind.BORROWED_IMMUTABLE)
        self.next_value_id += 1
        self.emit(MirOpCode.LOAD, operands=[container, index], result=result_op)
        return result_op
    
    def generate_lvalue(self, expr) -> MirOperand:
        if isinstance(expr, IdentifierExpr):
            if expr.name not in self.value_map:
                raise MirGenerationError(f"Undefined variable: {expr.name}", expr.token)
            return self.value_map[expr.name]
        else:
            val = self.generate_expression(expr)
            addr_op = MirOperand(name=f"addr_{self.next_value_id}", mir_type=MirPointerType(pointee=val.mir_type, mutable=True))
            self.next_value_id += 1
            self.emit(MirOpCode.BORROW_MUT, operands=[val], result=addr_op)
            return addr_op
    
    def map_ast_type(self, ast_type: Optional[BasicType]) -> MirType:
        if ast_type is None:
            return MirBasicType("void")
        if isinstance(ast_type, BasicType):
            type_map = {"int": "i64", "float": "f64", "bool": "bool", "char": "i8", "void": "void", "Unit": "void"}
            return MirBasicType(type_map.get(ast_type.name, "i64"))
        elif isinstance(ast_type, RefType):
            inner = self.map_ast_type(ast_type.inner)
            return MirPointerType(pointee=inner, mutable=ast_type.mutable)
        elif isinstance(ast_type, ResultType):
            ok = self.map_ast_type(ast_type.ok_type)
            err = self.map_ast_type(ast_type.err_type)
            return MirResultType(ok_type=ok, err_type=err)
        elif isinstance(ast_type, OptionType):
            inner = self.map_ast_type(ast_type.inner)
            return MirOptionType(inner=inner)
        elif isinstance(ast_type, NamedType):
            symbol = self.symbol_table.resolve(ast_type.name)
            if symbol and symbol.kind == SymbolKind.CLASS:
                return MirStructType(name=ast_type.name)
            return MirBasicType("i64")
        elif isinstance(ast_type, ArrayType):
            elem = self.map_ast_type(ast_type.element_type)
            return MirPointerType(pointee=elem, mutable=True)
        elif isinstance(ast_type, PredicatedType):
            base = self.map_ast_type(ast_type.base_type)
            if isinstance(base, MirBasicType):
                base.predicates.append(str(ast_type.predicate))
            return base
        return MirBasicType("i64")
    
    def new_block(self, suffix: str) -> MirBasicBlock:
        block_id = f"{suffix}_{self.next_block_id}"
        self.next_block_id += 1
        block = MirBasicBlock(name=block_id)
        self.blocks.append(block)
        return block
    
    def emit(self, opcode: MirOpCode, operands: List[MirOperand] = None, result: Optional[MirOperand] = None, effects: Set[str] = None) -> MirInstruction:
        if operands is None:
            operands = []
        if effects is None:
            effects = set()
        if opcode == MirOpCode.CALL and not effects and self.current_function:
            effects.add("ffi")
        instr = MirInstruction(opcode=opcode, operands=operands, result=result, effects=effects, token=self.current_function.token if self.current_function else None)
        if self.current_block is None:
            raise MirGenerationError("No current block for instruction emission", None)
        self.current_block.instructions.append(instr)
        return instr
    
    def has_terminator(self, block: MirBasicBlock) -> bool:
        if not block.instructions:
            return False
        last = block.instructions[-1]
        return last.opcode in (MirOpCode.RETURN, MirOpCode.JUMP, MirOpCode.BRANCH, MirOpCode.SWITCH, MirOpCode.UNREACHABLE)
    
    def schedule_scope_cleanup(self, scope_id: str):
        drops = []
        for var_name, operand in list(self.linear_vars.items()):
            drop_instr = MirInstruction(opcode=MirOpCode.DROP, operands=[operand], token=None)
            drops.append(drop_instr)
            del self.linear_vars[var_name]
        if drops:
            self.drop_points[scope_id] = drops
    
    def insert_linear_cleanup(self, module: MirModule):
        pass
    
    def verify_constitutional_properties(self, module: MirModule):
        violations = []
        for func in module.functions:
            if self.linear_vars:
                violations.append(f"Function {func.name}: Linear values not consumed before return")
        if violations:
            raise MirGenerationError("Constitutional violations in MIR:\n" + "\n".join(violations), None)