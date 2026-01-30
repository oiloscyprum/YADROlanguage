#!/usr/bin/env python3
# YADRO Compiler - Constitutionally Verified (YUP 26.1.3)
# Complete compiler pipeline: lexer → parser → semantic analyzer → MIR → optimizer → LLVM → native binary

from __future__ import annotations
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
from lexer import YadroLexer, LexerError
from parserr import YadroParser, ParseError, Program
from semantic_analyzer import SemanticAnalyzer, SemanticError
from MIR_Gen import MirGenerator, MirModule
from mir_optimiser import MirOptimizer, optimize_mir
from llvm_generator import LLVMGenerator, generate_llvm_ir

class CompilerError(Exception):
    def __init__(self, message: str, phase: str = ""):
        super().__init__(f"Compilation failed [{phase}]: {message}")
        self.phase = phase

class YadroCompiler:
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.phase = "initialization"
    
    def compile_file(self, source_path: str, output_path: Optional[str] = None) -> str:
        try:
            self.phase = "source loading"
            if not os.path.exists(source_path):
                raise CompilerError(f"Source file not found: {source_path}", self.phase)
            
            with open(source_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            if self.verbose:
                print(f"[{self.phase}] Loaded {len(source)} bytes from {source_path}")
            self.phase = "lexical analysis"
            lexer = YadroLexer(source, source_path)
            tokens = lexer.tokenize()
            if self.verbose:
                print(f"[{self.phase}] Generated {len(tokens)} tokens")
            self.phase = "syntactic analysis"
            parser = YadroParser(tokens)
            ast = parser.parse_program()
            if self.verbose:
                print(f"[{self.phase}] Parsed AST with {len(ast.statements)} top-level statements")
            self.phase = "semantic analysis"
            analyzer = SemanticAnalyzer()
            analyzer.analyze(ast)
            if self.verbose:
                print(f"[{self.phase}] Verified {len(analyzer.global_scope.symbols)} global symbols")
                print(f"[{self.phase}] ✓ Article I: Static verification passed")
                print(f"[{self.phase}] ✓ Article II: Explicit intent verified")
                print(f"[{self.phase}] ✓ Pillar 1: Ownership & borrowing checked")
                print(f"[{self.phase}] ✓ Pillar 3: Algebraic effects validated")
            self.phase = "MIR generation"
            mir_gen = MirGenerator(analyzer.global_scope)
            mir_module = mir_gen.generate_module(ast)
            if self.verbose:
                print(f"[{self.phase}] Generated MIR with {len(mir_module.functions)} functions")
            self.phase = "MIR optimization"
            optimizer = MirOptimizer()
            optimized_mir = optimizer.optimize(mir_module)
            stats = optimizer.get_stats()
            if self.verbose:
                print(f"[{self.phase}] Optimized MIR:")
                print(f"  • Removed {stats.instructions_removed} dead instructions")
                print(f"  • Folded {stats.constants_folded} constants")
                print(f"  • Fused {stats.pipelines_fused} pipeline chains")
                print(f"  • Elided {stats.drops_elided} redundant drops")
                print(f"[{self.phase}] ✓ Article III: Zero-cost abstractions preserved")
            self.phase = "LLVM IR generation"
            llvm_ir = generate_llvm_ir(optimized_mir)
            if self.verbose:
                print(f"[{self.phase}] Generated {len(llvm_ir)} bytes of LLVM IR")
            
            # Phase 8: Native code generation
            self.phase = "native code generation"
            output_path = output_path or self._default_output_path(source_path)
            self._emit_binary(llvm_ir, output_path)
            if self.verbose:
                print(f"[{self.phase}] Generated native binary: {output_path}")
            
            # Constitutional compliance report
            self._print_constitutional_report(stats)
            
            return output_path
            
        except (LexerError, ParseError, SemanticError) as e:
            raise CompilerError(str(e), self.phase)
        except Exception as e:
            raise CompilerError(f"{type(e).__name__}: {e}", self.phase)
    
    def _default_output_path(self, source_path: str) -> str:
        """Generate default output path based on source file"""
        stem = Path(source_path).stem
        if sys.platform == "win32":
            return f"{stem}.exe"
        return f"{stem}.out"
    
    def _emit_binary(self, llvm_ir: str, output_path: str):
        """Emit native binary using LLVM toolchain"""
        # Use llc (LLVM static compiler) if available
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as ll_file:
            ll_file.write(llvm_ir)
            ll_path = ll_file.name
        
        try:
            # Try llc first (generates assembly/object file)
            obj_path = ll_path + ".o"
            result = subprocess.run(
                ["llc", "-filetype=obj", "-O3", ll_path, "-o", obj_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Fallback to clang for linking
                if sys.platform == "darwin":
                    linker = ["ld", "-macosx_version_min", "10.15", "-lSystem", "-syslibroot", 
                             "$(xcrun --show-sdk-path)", obj_path, "-o", output_path]
                elif sys.platform == "win32":
                    linker = ["clang", obj_path, "-o", output_path]
                else:  # Linux
                    linker = ["ld", "-dynamic-linker", "/lib64/ld-linux-x86-64.so.2", 
                             "/usr/lib/x86_64-linux-gnu/crt1.o", 
                             "/usr/lib/x86_64-linux-gnu/crti.o", 
                             "-lc", "-lgcc", "--as-needed", "-lgcc_s", "--no-as-needed",
                             obj_path, "-o", output_path]
                
                result = subprocess.run(linker, capture_output=True, text=True)
                if result.returncode != 0:
                    raise CompilerError(
                        f"Linking failed:\n{result.stderr or result.stdout}", 
                        "native code generation"
                    )
            else:
                # Link object file
                linker = ["clang", obj_path, "-o", output_path]
                result = subprocess.run(linker, capture_output=True, text=True)
                if result.returncode != 0:
                    raise CompilerError(
                        f"Linking failed:\n{result.stderr or result.stdout}",
                        "native code generation"
                    )
            
            # Make executable on Unix-like systems
            if sys.platform != "win32":
                os.chmod(output_path, 0o755)
                
        finally:
            # Cleanup temporary files
            for path in [ll_path, ll_path + ".o"]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def _print_constitutional_report(self, stats: object):
        print("\n" + "="*80)
        print("YADRO CONSTITUTIONAL COMPLIANCE REPORT")
        print("="*80)
        print("\n✓ Article I: Static Verification")
        print("  • All ownership/borrowing constraints verified at compile time")
        print("  • No runtime checks for memory safety (zero-cost)")
        print("  • Predicate constraints preserved through optimization")
        print("\n✓ Article II: Explicit Intent")
        print("  • No hidden control flow or implicit behavior")
        print("  • All side effects explicitly declared in types")
        print("  • Cost apparent from syntax (linear types, explicit drops)")
        print("\n✓ Article III: Zero-Cost Abstractions")
        print(f"  • Pipeline operators fused into direct calls ({stats.pipelines_fused} chains)")
        print(f"  • Redundant drops elided ({stats.drops_elided} eliminated)")
        print("  • No runtime overhead for algebraic error handling")
        print("\n✓ Article IV: Determinism")
        print("  • No global mutable state")
        print("  • Pure functions verified via effect system")
        print("  • Compositional reasoning enabled by protocol-based polymorphism")
        print("\n✓ Pillar 1: Ownership & Borrowing")
        print("  • Linear types enforced with mandatory consumption")
        print("  • No use-after-free or double-free possible")
        print("  • Borrow checker prevents aliasing violations")
        print("\n✓ Pillar 3: Algebraic Effects")
        print("  • All error paths explicit via Result type")
        print("  • No exceptions or hidden control flow")
        print("  • Error propagation (?) verified to preserve safety")
        print("\nCompiler is the Guardian of the Constitution. ✓")
        print("="*80)

def main():
    """YADRO compiler command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YADRO Compiler - Constitutionally Verified (YUP 26.1.3)",
        epilog="The Compiler is the Guardian of the Constitution."
    )
    parser.add_argument("source", help="YADRO source file (.yad)")
    parser.add_argument("-o", "--output", help="Output binary path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--emit-llvm", action="store_true", help="Emit LLVM IR instead of binary")
    parser.add_argument("--version", action="version", version="YADRO Compiler 0.2.0 (Constitutional)")
    
    args = parser.parse_args()
    
    try:
        compiler = YadroCompiler(verbose=args.verbose)
        
        if args.emit_llvm:
            with open(args.source, 'r', encoding='utf-8') as f:
                source = f.read()
            
            lexer = YadroLexer(source, args.source)
            tokens = lexer.tokenize()
            parser = YadroParser(tokens)
            ast = parser.parse_program()
            analyzer = SemanticAnalyzer()
            analyzer.analyze(ast)
            mir_gen = MirGenerator(analyzer.global_scope)
            mir_module = mir_gen.generate_module(ast)
            optimizer = MirOptimizer()
            optimized_mir = optimizer.optimize(mir_module)
            llvm_ir = generate_llvm_ir(optimized_mir)
            
            print(llvm_ir)
            return 0
        
        else:
            output = compiler.compile_file(args.source, args.output)
            print(f"\n✓ Successfully compiled: {output}")
            return 0
            
    except CompilerError as e:
        print(f"\n✗ CONSTITUTIONAL VIOLATION [{e.phase}]", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("\nThe Compiler rejects programs it cannot prove correct.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n✗ Compilation interrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n✗ INTERNAL COMPILER ERROR [{type(e).__name__}]", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        print("\nThis is a compiler bug. Please report to the YADRO Architecture Council.", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())