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
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple
from MIR_Gen import (
    MirModule, MirFunction, MirBasicBlock, MirInstruction, MirOpCode,
    MirOperand, MirType, MirBasicType, MirPointerType, MirStructType,
    MirResultType, MirOptionType, OwnershipKind
)

class MirOptimizationError(Exception):
    def __init__(self, message: str):
        super().__init__(f"MIR Optimization Error: {message}")

@dataclass
class OptimizationStats:
    instructions_removed: int = 0
    blocks_removed: int = 0
    constants_folded: int = 0
    pipelines_fused: int = 0
    drops_elided: int = 0
    ownership_errors: int = 0

class OwnershipVerifier:
    def __init__(self):
        self.errors: List[str] = []
        self.linear_vars: Dict[str, Tuple[MirOperand, str]] = {}
        self.borrowed_vars: Dict[str, List[Tuple[MirOperand, bool]]] = {}
    
    def verify_function(self, func: MirFunction) -> List[str]:
        self.errors = []
        self.linear_vars.clear()
        self.borrowed_vars.clear()
        for block in func.blocks:
            for instr in block.instructions:
                self.verify_instruction(instr)
        self.check_unconsumed_linears()
        return self.errors
    
    def verify_instruction(self, instr: MirInstruction):
        if instr.opcode == MirOpCode.MOVE:
            if len(instr.operands) >= 1:
                src = instr.operands[0]
                if src.ownership == OwnershipKind.LINEAR:
                    if src.name in self.linear_vars:
                        self.errors.append(f"Use-after-move: {src.name}")
                    else:
                        self.linear_vars[src.name] = (src, "moved")
        elif instr.opcode == MirOpCode.DROP:
            if len(instr.operands) >= 1:
                target = instr.operands[0]
                if target.name in self.linear_vars:
                    del self.linear_vars[target.name]
                else:
                    self.errors.append(f"Double-free or invalid drop: {target.name}")
        elif instr.opcode in (MirOpCode.BORROW_IMMUT, MirOpCode.BORROW_MUT):
            if len(instr.operands) >= 1 and instr.result:
                src = instr.operands[0]
                is_mut = instr.opcode == MirOpCode.BORROW_MUT
                if src.name not in self.borrowed_vars:
                    self.borrowed_vars[src.name] = []
                self.borrowed_vars[src.name].append((instr.result, is_mut))
                if src.ownership == OwnershipKind.LINEAR:
                    if src.name in self.linear_vars:
                        del self.linear_vars[src.name]
        elif instr.opcode == MirOpCode.LOAD:
            if len(instr.operands) >= 1:
                ptr = instr.operands[0]
                if isinstance(ptr.mir_type, MirPointerType):
                    base_name = ptr.name.split('_')[0] if '_' in ptr.name else ptr.name
                    if base_name in self.borrowed_vars:
                        for borrow, is_mut in self.borrowed_vars[base_name]:
                            if is_mut and instr.opcode == MirOpCode.STORE:
                                self.errors.append(f"Aliasing violation: mutable borrow while borrowed")
    
    def check_unconsumed_linears(self):
        for name, (operand, state) in self.linear_vars.items():
            self.errors.append(f"Linear value not consumed: {name} (state: {state})")

class PredicateSolver:
    def __init__(self):
        self.known_constraints: Dict[str, Set[Tuple[str, str, str]]] = {}
    
    def solve_function(self, func: MirFunction) -> OptimizationStats:
        stats = OptimizationStats()
        self.known_constraints.clear()
        new_blocks = []
        for block in func.blocks:
            new_instrs = []
            for instr in block.instructions:
                optimized = self.optimize_instruction(instr)
                if optimized:
                    new_instrs.extend(optimized)
                    if len(optimized) < len([instr]):
                        stats.instructions_removed += 1
                else:
                    new_instrs.append(instr)
            block.instructions = new_instrs
            new_blocks.append(block)
        func.blocks = new_blocks
        return stats
    
    def optimize_instruction(self, instr: MirInstruction) -> Optional[List[MirInstruction]]:
        if instr.opcode in (MirOpCode.EQ, MirOpCode.NE, MirOpCode.LT, MirOpCode.GT, MirOpCode.LE, MirOpCode.GE):
            if len(instr.operands) >= 2 and instr.operands[0].is_const and instr.operands[1].is_const:
                left = instr.operands[0].name
                right = instr.operands[1].name
                op = instr.opcode
                result = self.evaluate_const_comparison(left, right, op)
                if result is not None:
                    return [MirInstruction(
                        opcode=MirOpCode.LOAD,
                        result=instr.result,
                        operands=[MirOperand(name="true" if result else "false", mir_type=MirBasicType("bool"), is_const=True)]
                    )]
        return None
    
    def evaluate_const_comparison(self, left: str, right: str, op: MirOpCode) -> Optional[bool]:
        try:
            lval = int(left) if left.isdigit() else float(left)
            rval = int(right) if right.isdigit() else float(right)
            if op == MirOpCode.EQ: return lval == rval
            if op == MirOpCode.NE: return lval != rval
            if op == MirOpCode.LT: return lval < rval
            if op == MirOpCode.GT: return lval > rval
            if op == MirOpCode.LE: return lval <= rval
            if op == MirOpCode.GE: return lval >= rval
        except:
            pass
        return None

class PipelineFuser:
    def fuse_pipelines(self, func: MirFunction) -> OptimizationStats:
        stats = OptimizationStats()
        for block in func.blocks:
            i = 0
            while i < len(block.instructions) - 2:
                if self.is_pipeline_sequence(block.instructions[i:i+3]):
                    fused = self.create_fused_call(block.instructions[i:i+3])
                    block.instructions[i:i+3] = [fused]
                    stats.pipelines_fused += 1
                    stats.instructions_removed += 2
                else:
                    i += 1
        return stats
    
    def is_pipeline_sequence(self, instrs: List[MirInstruction]) -> bool:
        if len(instrs) < 3:
            return False
        return (instrs[0].opcode == MirOpCode.CALL and 
                instrs[1].opcode == MirOpCode.CALL and 
                instrs[2].opcode == MirOpCode.CALL and
                instrs[1].operands and instrs[1].operands[0] == instrs[0].result and
                instrs[2].operands and instrs[2].operands[0] == instrs[1].result)
    
    def create_fused_call(self, instrs: List[MirInstruction]) -> MirInstruction:
        first = instrs[0]
        last = instrs[2]
        return MirInstruction(
            opcode=MirOpCode.CALL,
            operands=first.operands,
            result=last.result,
            effects=first.effects | last.effects
        )

class DropElider:
    def elide_drops(self, func: MirFunction) -> OptimizationStats:
        stats = OptimizationStats()
        linear_vars: Set[str] = set()
        consumed_vars: Set[str] = set()
        drop_positions: Dict[str, List[int]] = {}
        
        for block_idx, block in enumerate(func.blocks):
            for instr_idx, instr in enumerate(block.instructions):
                if instr.opcode == MirOpCode.ALLOC and instr.result and instr.result.ownership == OwnershipKind.LINEAR:
                    linear_vars.add(instr.result.name)
                    drop_positions[instr.result.name] = []
                elif instr.opcode == MirOpCode.MOVE and instr.operands:
                    src = instr.operands[0]
                    if src.name in linear_vars:
                        consumed_vars.add(src.name)
                elif instr.opcode == MirOpCode.DROP and instr.operands:
                    target = instr.operands[0]
                    if target.name in linear_vars:
                        drop_positions[target.name].append(instr_idx)
        
        for block in func.blocks:
            new_instrs = []
            for instr in block.instructions:
                if instr.opcode == MirOpCode.DROP and instr.operands:
                    target = instr.operands[0]
                    if target.name in consumed_vars:
                        stats.drops_elided += 1
                        continue
                new_instrs.append(instr)
            block.instructions = new_instrs
        
        return stats

class DeadCodeEliminator:
    def eliminate_dead_code(self, func: MirFunction) -> OptimizationStats:
        stats = OptimizationStats()
        reachable = self.find_reachable_blocks(func)
        unreachable_blocks = [b for b in func.blocks if b.name not in reachable]
        stats.blocks_removed = len(unreachable_blocks)
        func.blocks = [b for b in func.blocks if b.name in reachable]
        
        for block in func.blocks:
            new_instrs = []
            defined = set()
            used = self.collect_uses(block)
            for instr in reversed(block.instructions):
                if instr.result and instr.result.name in used:
                    new_instrs.append(instr)
                    defined.add(instr.result.name)
                    used.update(self.get_operand_names(instr))
                elif instr.opcode in (MirOpCode.RETURN, MirOpCode.BRANCH, MirOpCode.JUMP, MirOpCode.SWITCH):
                    new_instrs.append(instr)
            block.instructions = list(reversed(new_instrs))
            stats.instructions_removed += len(block.instructions) - len(new_instrs)
        
        return stats
    
    def find_reachable_blocks(self, func: MirFunction) -> Set[str]:
        reachable = set()
        worklist = [func.blocks[0].name] if func.blocks else []
        while worklist:
            name = worklist.pop()
            if name in reachable:
                continue
            reachable.add(name)
            block = next((b for b in func.blocks if b.name == name), None)
            if block:
                for succ in block.successors:
                    if succ.name not in reachable:
                        worklist.append(succ.name)
        return reachable
    
    def collect_uses(self, block: MirBasicBlock) -> Set[str]:
        uses = set()
        for instr in block.instructions:
            if instr.opcode == MirOpCode.RETURN and instr.operands:
                uses.add(instr.operands[0].name)
            elif instr.opcode in (MirOpCode.BRANCH, MirOpCode.SWITCH) and instr.operands:
                uses.update(op.name for op in instr.operands if op.name)
        return uses
    
    def get_operand_names(self, instr: MirInstruction) -> Set[str]:
        return {op.name for op in instr.operands if op.name}

class ConstantFolder:
    def fold_constants(self, func: MirFunction) -> OptimizationStats:
        stats = OptimizationStats()
        for block in func.blocks:
            for instr in block.instructions:
                if instr.opcode in (MirOpCode.ADD, MirOpCode.SUB, MirOpCode.MUL, MirOpCode.DIV) and instr.result:
                    if len(instr.operands) >= 2 and instr.operands[0].is_const and instr.operands[1].is_const:
                        try:
                            left = int(instr.operands[0].name)
                            right = int(instr.operands[1].name)
                            result = self.compute(left, right, instr.opcode)
                            instr.opcode = MirOpCode.LOAD
                            instr.operands = [MirOperand(name=str(result), mir_type=instr.result.mir_type, is_const=True)]
                            stats.constants_folded += 1
                        except:
                            pass
        return stats
    
    def compute(self, left: int, right: int, opcode: MirOpCode) -> int:
        if opcode == MirOpCode.ADD: return left + right
        if opcode == MirOpCode.SUB: return left - right
        if opcode == MirOpCode.MUL: return left * right
        if opcode == MirOpCode.DIV: return left // right if right != 0 else 0
        return 0

class MirOptimizer:
    def __init__(self):
        self.stats = OptimizationStats()
        self.passes = [
            ("OwnershipVerification", self.run_ownership_verification),
            ("DeadCodeElimination", self.run_dead_code_elimination),
            ("ConstantFolding", self.run_constant_folding),
            ("PredicateSolving", self.run_predicate_solving),
            ("PipelineFusion", self.run_pipeline_fusion),
            ("DropElision", self.run_drop_elision),
            ("DeadCodeElimination2", self.run_dead_code_elimination)
        ]
    
    def optimize(self, module: MirModule) -> MirModule:
        self.stats = OptimizationStats()
        for func in module.functions:
            for pass_name, pass_func in self.passes:
                try:
                    pass_stats = pass_func(func)
                    self.accumulate_stats(pass_stats)
                except Exception as e:
                    raise MirOptimizationError(f"Optimization pass '{pass_name}' failed: {str(e)}")
        return module
    
    def run_ownership_verification(self, func: MirFunction) -> OptimizationStats:
        verifier = OwnershipVerifier()
        errors = verifier.verify_function(func)
        if errors:
            raise MirOptimizationError("Ownership verification failed:\n" + "\n".join(errors))
        return OptimizationStats(ownership_errors=len(errors))
    
    def run_dead_code_elimination(self, func: MirFunction) -> OptimizationStats:
        eliminator = DeadCodeEliminator()
        return eliminator.eliminate_dead_code(func)
    
    def run_constant_folding(self, func: MirFunction) -> OptimizationStats:
        folder = ConstantFolder()
        return folder.fold_constants(func)
    
    def run_predicate_solving(self, func: MirFunction) -> OptimizationStats:
        solver = PredicateSolver()
        return solver.solve_function(func)
    
    def run_pipeline_fusion(self, func: MirFunction) -> OptimizationStats:
        fuser = PipelineFuser()
        return fuser.fuse_pipelines(func)
    
    def run_drop_elision(self, func: MirFunction) -> OptimizationStats:
        elider = DropElider()
        return elider.elide_drops(func)
    
    def accumulate_stats(self, stats: OptimizationStats):
        self.stats.instructions_removed += stats.instructions_removed
        self.stats.blocks_removed += stats.blocks_removed
        self.stats.constants_folded += stats.constants_folded
        self.stats.pipelines_fused += stats.pipelines_fused
        self.stats.drops_elided += stats.drops_elided
        self.stats.ownership_errors += stats.ownership_errors
    
    def get_stats(self) -> OptimizationStats:
        return self.stats

def optimize_mir(module: MirModule) -> MirModule:
    optimizer = MirOptimizer()
    optimized = optimizer.optimize(module)
    return optimized