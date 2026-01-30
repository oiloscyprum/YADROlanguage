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
from llvmlite import ir
from typing import Dict, Optional, List, Union
from MIR_Gen import (
    MirModule, MirFunction, MirBasicBlock, MirInstruction, MirOpCode,
    MirOperand, MirType, MirBasicType, MirPointerType, MirStructType,
    MirResultType, MirOptionType, OwnershipKind
)
from mir_optimiser import OptimizationStats

class LLVMGenerationError(Exception):
    def __init__(self, message: str):
        super().__init__(f"LLVM Generation Error: {message}")

class LLVMTypeCache:
    def __init__(self):
        self.type_map: Dict[str, ir.Type] = {}
        self.struct_types: Dict[str, ir.IdentifiedStructType] = {}
    
    def get_llvm_type(self, mir_type: MirType) -> ir.Type:
        type_key = self._type_key(mir_type)
        if type_key in self.type_map:
            return self.type_map[type_key]
        
        llvm_type = self._create_llvm_type(mir_type)
        self.type_map[type_key] = llvm_type
        return llvm_type
    
    def _type_key(self, mir_type: MirType) -> str:
        if isinstance(mir_type, MirBasicType):
            return f"basic:{mir_type.name}"
        elif isinstance(mir_type, MirPointerType):
            return f"ptr:{self._type_key(mir_type.pointee)}:{mir_type.mutable}:{mir_type.is_borrow}"
        elif isinstance(mir_type, MirStructType):
            return f"struct:{mir_type.name}"
        elif isinstance(mir_type, MirResultType):
            return f"result:{self._type_key(mir_type.ok_type)}:{self._type_key(mir_type.err_type)}"
        elif isinstance(mir_type, MirOptionType):
            return f"option:{self._type_key(mir_type.inner)}"
        return f"unknown:{type(mir_type).__name__}"
    
    def _create_llvm_type(self, mir_type: MirType) -> ir.Type:
        if isinstance(mir_type, MirBasicType):
            if mir_type.name == "i64":
                return ir.IntType(64)
            elif mir_type.name == "f64":
                return ir.DoubleType()
            elif mir_type.name == "bool":
                return ir.IntType(1)
            elif mir_type.name == "i8":
                return ir.IntType(8)
            elif mir_type.name == "void":
                return ir.VoidType()
            else:
                raise LLVMGenerationError(f"Unsupported basic type: {mir_type.name}")
        
        elif isinstance(mir_type, MirPointerType):
            pointee = self.get_llvm_type(mir_type.pointee)
            return ir.PointerType(pointee)
        
        elif isinstance(mir_type, MirStructType):
            if mir_type.name in self.struct_types:
                return self.struct_types[mir_type.name]
            
            struct_type = ir.global_context.get_identified_type(mir_type.name)
            self.struct_types[mir_type.name] = struct_type
            
            if mir_type.fields:
                field_types = [self.get_llvm_type(field_type) for _, field_type in mir_type.fields]
                struct_type.set_body(*field_types)
            else:
                struct_type.set_body(ir.IntType(8))
            
            return struct_type
        
        elif isinstance(mir_type, MirResultType):
            ok_type = self.get_llvm_type(mir_type.ok_type)
            err_type = self.get_llvm_type(mir_type.err_type)
            tag_type = ir.IntType(1)
            max_size = max(self._type_size(ok_type), self._type_size(err_type))
            union_type = ir.ArrayType(ir.IntType(8), max_size)
            return ir.global_context.get_identified_type(f"Result_{ok_type}_{err_type}")
        
        elif isinstance(mir_type, MirOptionType):
            inner_type = self.get_llvm_type(mir_type.inner)
            tag_type = ir.IntType(1)
            return ir.LiteralStructType([tag_type, inner_type])
        
        else:
            raise LLVMGenerationError(f"Unsupported type: {type(mir_type).__name__}")
    
    def _type_size(self, llvm_type: ir.Type) -> int:
        if isinstance(llvm_type, ir.IntType):
            return llvm_type.width // 8
        elif isinstance(llvm_type, ir.DoubleType):
            return 8
        elif isinstance(llvm_type, ir.PointerType):
            return 8
        elif isinstance(llvm_type, ir.LiteralStructType):
            return sum(self._type_size(t) for t in llvm_type.elements)
        return 1

class LLVMGenerator:
    def __init__(self):
        self.module = ir.Module(name="yadro_module")
        self.module.triple = "x86_64-unknown-linux-gnu"
        self.type_cache = LLVMTypeCache()
        self.func_decls: Dict[str, ir.Function] = {}
        self.current_func: Optional[ir.Function] = None
        self.current_builder: Optional[ir.IRBuilder] = None
        self.block_map: Dict[str, ir.Block] = {}
        self.value_map: Dict[str, ir.Value] = {}
        self.stats = OptimizationStats()
    
    def generate(self, mir_module: MirModule) -> ir.Module:
        self._declare_runtime_functions()
        self._declare_struct_types(mir_module)
        self._declare_functions(mir_module)
        self._define_functions(mir_module)
        return self.module
    
    def _declare_runtime_functions(self):
        void_ptr = ir.PointerType(ir.IntType(8))
        
        panic_fn = ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="yadro_panic")
        self.func_decls["yadro_panic"] = panic_fn
        
        alloc_fn = ir.Function(self.module, ir.FunctionType(void_ptr, [ir.IntType(64)]), name="yadro_alloc")
        self.func_decls["yadro_alloc"] = alloc_fn
        
        dealloc_fn = ir.Function(self.module, ir.FunctionType(ir.VoidType(), [void_ptr]), name="yadro_dealloc")
        self.func_decls["yadro_dealloc"] = dealloc_fn
    
    def _declare_struct_types(self, mir_module: MirModule):
        for name, mir_struct in mir_module.types.items():
            self.type_cache.get_llvm_type(mir_struct)
    
    def _declare_functions(self, mir_module: MirModule):
        for mir_func in mir_module.functions:
            ret_type = self.type_cache.get_llvm_type(mir_func.return_type)
            param_types = [self.type_cache.get_llvm_type(p.mir_type) for p in mir_func.params]
            func_type = ir.FunctionType(ret_type, param_types)
            llvm_func = ir.Function(self.module, func_type, name=mir_func.name)
            self.func_decls[mir_func.name] = llvm_func
    
    def _define_functions(self, mir_module: MirModule):
        for mir_func in mir_module.functions:
            self._define_function(mir_func)
    
    def _define_function(self, mir_func: MirFunction):
        llvm_func = self.func_decls[mir_func.name]
        self.current_func = llvm_func
        self.block_map.clear()
        self.value_map.clear()
        
        entry_block = llvm_func.append_basic_block(name="entry")
        self.current_builder = ir.IRBuilder(entry_block)
        
        for i, param in enumerate(llvm_func.args):
            param.name = mir_func.params[i].name
            self.value_map[mir_func.params[i].name] = param
        
        for mir_block in mir_func.blocks:
            llvm_block = llvm_func.append_basic_block(name=mir_block.name)
            self.block_map[mir_block.name] = llvm_block
        
        for mir_block in mir_func.blocks:
            self.current_builder.position_at_end(self.block_map[mir_block.name])
            for instr in mir_block.instructions:
                self._generate_instruction(instr)
        
        if not entry_block.is_terminated:
            ret_type = llvm_func.type.return_type
            if isinstance(ret_type, ir.VoidType):
                self.current_builder.ret_void()
            else:
                zero = ir.Constant(ret_type, 0)
                self.current_builder.ret(zero)
    
    def _generate_instruction(self, instr: MirInstruction):
        handlers = {
            MirOpCode.ALLOC: self._gen_alloc,
            MirOpCode.ALLOC_HEAP: self._gen_alloc_heap,
            MirOpCode.LOAD: self._gen_load,
            MirOpCode.STORE: self._gen_store,
            MirOpCode.MOVE: self._gen_move,
            MirOpCode.BORROW_IMMUT: self._gen_borrow_immut,
            MirOpCode.BORROW_MUT: self._gen_borrow_mut,
            MirOpCode.DROP: self._gen_drop,
            MirOpCode.ADD: self._gen_add,
            MirOpCode.SUB: self._gen_sub,
            MirOpCode.MUL: self._gen_mul,
            MirOpCode.DIV: self._gen_div,
            MirOpCode.MOD: self._gen_mod,
            MirOpCode.BIT_AND: self._gen_bit_and,
            MirOpCode.BIT_OR: self._gen_bit_or,
            MirOpCode.BIT_XOR: self._gen_bit_xor,
            MirOpCode.SHL: self._gen_shl,
            MirOpCode.SHR: self._gen_shr,
            MirOpCode.EQ: self._gen_eq,
            MirOpCode.NE: self._gen_ne,
            MirOpCode.LT: self._gen_lt,
            MirOpCode.GT: self._gen_gt,
            MirOpCode.LE: self._gen_le,
            MirOpCode.GE: self._gen_ge,
            MirOpCode.BRANCH: self._gen_branch,
            MirOpCode.JUMP: self._gen_jump,
            MirOpCode.CALL: self._gen_call,
            MirOpCode.RETURN: self._gen_return,
            MirOpCode.RESULT_OK: self._gen_result_ok,
            MirOpCode.RESULT_ERR: self._gen_result_err,
            MirOpCode.RESULT_UNWRAP: self._gen_result_unwrap,
            MirOpCode.RESULT_IS_OK: self._gen_result_is_ok,
            MirOpCode.NOP: self._gen_nop,
            MirOpCode.UNREACHABLE: self._gen_unreachable,
        }
        
        handler = handlers.get(instr.opcode)
        if not handler:
            raise LLVMGenerationError(f"Unsupported opcode: {instr.opcode.name}")
        
        handler(instr)
    
    def _get_value(self, operand: MirOperand) -> ir.Value:
        if operand.name in self.value_map:
            return self.value_map[operand.name]
        
        if operand.is_const:
            llvm_type = self.type_cache.get_llvm_type(operand.mir_type)
            if isinstance(llvm_type, ir.IntType):
                try:
                    value = int(operand.name)
                    return ir.Constant(llvm_type, value)
                except ValueError:
                    if operand.name == "true":
                        return ir.Constant(llvm_type, 1)
                    elif operand.name == "false":
                        return ir.Constant(llvm_type, 0)
            elif isinstance(llvm_type, ir.DoubleType):
                try:
                    value = float(operand.name)
                    return ir.Constant(llvm_type, value)
                except ValueError:
                    pass
        
        raise LLVMGenerationError(f"Undefined operand: {operand.name}")
    
    def _set_value(self, operand: Optional[MirOperand], value: ir.Value):
        if operand and operand.name:
            self.value_map[operand.name] = value
    
    def _gen_alloc(self, instr: MirInstruction):
        assert instr.result
        llvm_type = self.type_cache.get_llvm_type(instr.result.mir_type)
        alloca = self.current_builder.alloca(llvm_type, name=instr.result.name)
        self._set_value(instr.result, alloca)
    
    def _gen_alloc_heap(self, instr: MirInstruction):
        assert instr.result
        llvm_type = self.type_cache.get_llvm_type(instr.result.mir_type)
        size = ir.Constant(ir.IntType(64), self._get_type_size(llvm_type))
        alloc_fn = self.func_decls["yadro_alloc"]
        ptr = self.current_builder.call(alloc_fn, [size], name=f"{instr.result.name}_raw")
        bitcast = self.current_builder.bitcast(ptr, ir.PointerType(llvm_type), name=instr.result.name)
        self._set_value(instr.result, bitcast)
    
    def _gen_load(self, instr: MirInstruction):
        assert instr.result and len(instr.operands) >= 1
        ptr = self._get_value(instr.operands[0])
        loaded = self.current_builder.load(ptr.type.pointee, ptr, name=instr.result.name)
        self._set_value(instr.result, loaded)
    
    def _gen_store(self, instr: MirInstruction):
        assert len(instr.operands) >= 2
        value = self._get_value(instr.operands[0])
        ptr = self._get_value(instr.operands[1])
        self.current_builder.store(value, ptr)
    
    def _gen_move(self, instr: MirInstruction):
        assert instr.result and len(instr.operands) >= 1
        src = self._get_value(instr.operands[0])
        self._set_value(instr.result, src)
    
    def _gen_borrow_immut(self, instr: MirInstruction):
        self._gen_borrow(instr, mutable=False)
    
    def _gen_borrow_mut(self, instr: MirInstruction):
        self._gen_borrow(instr, mutable=True)
    
    def _gen_borrow(self, instr: MirInstruction, mutable: bool):
        assert instr.result and len(instr.operands) >= 1
        src = self._get_value(instr.operands[0])
        if not isinstance(src.type, ir.PointerType):
            alloca = self.current_builder.alloca(src.type, name=f"{instr.result.name}_temp")
            self.current_builder.store(src, alloca)
            src = alloca
        self._set_value(instr.result, src)
    
    def _gen_drop(self, instr: MirInstruction):
        assert len(instr.operands) >= 1
        ptr = self._get_value(instr.operands[0])
        
        if isinstance(ptr.type, ir.PointerType):
            pointee_type = ptr.type.pointee
            if isinstance(pointee_type, ir.IdentifiedStructType):
                drop_fn_name = f"drop_{pointee_type.name}"
                if drop_fn_name in self.func_decls:
                    drop_fn = self.func_decls[drop_fn_name]
                    self.current_builder.call(drop_fn, [ptr])
            
            if "heap" in str(ptr):
                dealloc_fn = self.func_decls["yadro_dealloc"]
                void_ptr = self.current_builder.bitcast(ptr, ir.PointerType(ir.IntType(8)))
                self.current_builder.call(dealloc_fn, [void_ptr])
    
    def _gen_add(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.add)
    
    def _gen_sub(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.sub)
    
    def _gen_mul(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.mul)
    
    def _gen_div(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.sdiv)
    
    def _gen_mod(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.srem)
    
    def _gen_bit_and(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.and_)
    
    def _gen_bit_or(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.or_)
    
    def _gen_bit_xor(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.xor_)
    
    def _gen_shl(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.shl)
    
    def _gen_shr(self, instr: MirInstruction):
        self._gen_binop(instr, self.current_builder.ashr)
    
    def _gen_eq(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '==')
    
    def _gen_ne(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '!=')
    
    def _gen_lt(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '<')
    
    def _gen_gt(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '>')
    
    def _gen_le(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '<=')
    
    def _gen_ge(self, instr: MirInstruction):
        self._gen_cmp(instr, self.current_builder.icmp_signed, '>=')
    
    def _gen_binop(self, instr: MirInstruction, op_func):
        assert instr.result and len(instr.operands) >= 2
        left = self._get_value(instr.operands[0])
        right = self._get_value(instr.operands[1])
        result = op_func(left, right, name=instr.result.name)
        self._set_value(instr.result, result)
    
    def _gen_cmp(self, instr: MirInstruction, cmp_func, predicate):
        assert instr.result and len(instr.operands) >= 2
        left = self._get_value(instr.operands[0])
        right = self._get_value(instr.operands[1])
        result = cmp_func(predicate, left, right, name=instr.result.name)
        self._set_value(instr.result, result)
    
    def _gen_branch(self, instr: MirInstruction):
        assert len(instr.operands) >= 3
        cond = self._get_value(instr.operands[0])
        true_block = self.block_map[instr.operands[1].name]
        false_block = self.block_map[instr.operands[2].name]
        self.current_builder.cbranch(cond, true_block, false_block)
    
    def _gen_jump(self, instr: MirInstruction):
        assert len(instr.operands) >= 1
        target_block = self.block_map[instr.operands[0].name]
        self.current_builder.branch(target_block)
    
    def _gen_call(self, instr: MirInstruction):
        assert instr.result
        callee = self._get_callee(instr.operands[0])
        args = [self._get_value(op) for op in instr.operands[1:]]
        result = self.current_builder.call(callee, args, name=instr.result.name)
        self._set_value(instr.result, result)
    
    def _get_callee(self, operand: MirOperand) -> ir.Value:
        if operand.name in self.func_decls:
            return self.func_decls[operand.name]
        return self._get_value(operand)
    
    def _gen_return(self, instr: MirInstruction):
        if instr.operands:
            value = self._get_value(instr.operands[0])
            self.current_builder.ret(value)
        else:
            self.current_builder.ret_void()
    
    def _gen_result_ok(self, instr: MirInstruction):
        self._gen_result_ctor(instr, is_ok=True)
    
    def _gen_result_err(self, instr: MirInstruction):
        self._gen_result_ctor(instr, is_ok=False)
    
    def _gen_result_ctor(self, instr: MirInstruction, is_ok: bool):
        assert instr.result and len(instr.operands) >= 1
        value = self._get_value(instr.operands[0])
        llvm_type = self.type_cache.get_llvm_type(instr.result.mir_type)
        
        if isinstance(llvm_type, ir.LiteralStructType):
            result = ir.Constant(llvm_type, None)
            result = self.current_builder.insert_value(result, ir.Constant(ir.IntType(1), int(is_ok)), 0, name=f"{instr.result.name}_tag")
            result = self.current_builder.insert_value(result, value, 1, name=instr.result.name)
            self._set_value(instr.result, result)
    
    def _gen_result_unwrap(self, instr: MirInstruction):
        assert instr.result and len(instr.operands) >= 1
        result_val = self._get_value(instr.operands[0])
        llvm_type = self.type_cache.get_llvm_type(instr.result.mir_type)
        
        if isinstance(result_val.type, ir.LiteralStructType):
            tag = self.current_builder.extract_value(result_val, 0, name="tag")
            is_ok = self.current_builder.icmp_signed('==', tag, ir.Constant(ir.IntType(1), 1))
            
            then_block = self.current_func.append_basic_block(name="unwrap_ok")
            else_block = self.current_func.append_basic_block(name="unwrap_err")
            merge_block = self.current_func.append_basic_block(name="unwrap_merge")
            
            self.current_builder.cbranch(is_ok, then_block, else_block)
            
            self.current_builder.position_at_end(then_block)
            ok_val = self.current_builder.extract_value(result_val, 1, name="ok_val")
            self.current_builder.branch(merge_block)
            
            self.current_builder.position_at_end(else_block)
            panic_fn = self.func_decls["yadro_panic"]
            msg = self.current_builder.bitcast(ir.Constant(ir.ArrayType(ir.IntType(8), 15), bytearray(b"unwrap on Err\0")), ir.PointerType(ir.IntType(8)))
            self.current_builder.call(panic_fn, [msg])
            self.current_builder.unreachable()
            
            self.current_builder.position_at_end(merge_block)
            phi = self.current_builder.phi(llvm_type, name=instr.result.name)
            phi.add_incoming(ok_val, then_block)
            phi.add_incoming(ir.Constant(llvm_type, None), else_block)
            self._set_value(instr.result, phi)
    
    def _gen_result_is_ok(self, instr: MirInstruction):
        assert instr.result and len(instr.operands) >= 1
        result_val = self._get_value(instr.operands[0])
        
        if isinstance(result_val.type, ir.LiteralStructType):
            tag = self.current_builder.extract_value(result_val, 0, name=instr.result.name)
            self._set_value(instr.result, tag)
    
    def _gen_nop(self, instr: MirInstruction):
        pass
    
    def _gen_unreachable(self, instr: MirInstruction):
        self.current_builder.unreachable()
    
    def _get_type_size(self, llvm_type: ir.Type) -> int:
        if isinstance(llvm_type, ir.IntType):
            return (llvm_type.width + 7) // 8
        elif isinstance(llvm_type, ir.DoubleType):
            return 8
        elif isinstance(llvm_type, ir.PointerType):
            return 8
        elif isinstance(llvm_type, ir.LiteralStructType):
            return sum(self._get_type_size(t) for t in llvm_type.elements)
        return 1

def generate_llvm_ir(mir_module: MirModule) -> str:
    generator = LLVMGenerator()
    llvm_module = generator.generate(mir_module)
    return str(llvm_module)