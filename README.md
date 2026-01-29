# YADRO Language Specification (Revised)
## Overview
YADRO (Your Architecture's Definitive Runtime Optimizer)
 made by CyrOil
   - version 0.2.0 alpha
   - date: 2026-01-29
   - description: a language for operating systems development and low-level programming with high-level syntax, static safety, and zero-cost abstractions.
   - Governed by: YUP 26.1.1 using updates from YUP 26.1.2 and YUP 26.1.3

## Core Philosophy & Memory Model
YADRO employs a dual-level memory model: **static ownership for safety/performance** and a **managed heap (`gc`) for convenience**.

*   **Ownership & Borrowing:** All types are unique by default. Assignment or passing to a function **moves** ownership. Access is via immutable (`&T`) or mutable (`&mut T`) references with compile-time lifetime checking.
*   **Managed Heap:** The `gc<T>` type is a reference-counted pointer to a value in a separate heap. Use for data structures with unclear ownership or dynamic lifetimes. **Cycles must be broken manually using `gc_weak<T>`**.
*   **Explicit Copying:** Only types implementing the `#Copy` intrinsic (e.g., `int`, `float`) can be copied implicitly. Use the `copy()` method for others.
*   **Immediate Destruction:** The `del` keyword immediately calls a value's destructor and releases its memory.

## Compiler Directives
### #target (formerly #direct)
Declares the compilation environment. Does not change language semantics.
```
#target
os = "linux"
arch = "x86-64"
```
### #plugin
```
#plugin
safety-opt
linter==1.2
```
### #requires
Declares external native dependencies (DLLs/SOs). Functions from these are considered `#[unsafe]`.
```
#requires
"kernel32.dll"
"libc.so.6"
```
### #import
Imports YADRO modules.
```
#import
std::core::cli
std::os::fs as fs
```
### #start / #end
Marks main program body boundaries.

## Control Structures
### Branching
```
if condition:
    ...
elsif condition:
    ...
else:
    ...

switch expression:
case pattern1:
    ...
case pattern2:
    ...
default:
    ...
```
### Loops
```
for Type var in range(start, end, step):
    ...

while condition:
    ...

repeat:
    ...
until condition
```

## Functions & Error Handling
### Function Declarations
```
// Basic function
fun name(Type param, Type param = default) -> ReturnType:
    ...

// Specialized functions
fun[async] name() -> Task[Type]
fun[thread] name() -> ThreadId
fun[const] name() -> Type  // Compile-time execution
fun[ffi("lib.so", abi="cdecl")] extern_func() -> &void

// Generic function
temp<Type> fun identity(Type val) -> Type:
    return val

// Method (within class)
fun[class] method(Class self) -> ReturnType:
    ...
```

### Result Type & Error Handling
The `or` operator for errors is removed. Use the `Result[T, E]` algebraic type.
```
// Function that may fail
fun read_file(string path) -> Result[string, IOError]:
    if !file_exists(path):
        return Err(IOError("Not found"))
    return Ok(file_content)

// Explicit handling with `switch`
Result[string, IOError] res = read_file("log.txt")
switch res:
    case Ok(contents):
        cli.print(contents)
    case Err(err):
        cli.print("Error: " + err.message)

// Concise propagation with `?` operator
fun process() -> Result[int, IOError]:
    string data = read_file("data.bin")?  // Early return on Err
    int num = parse_binary(data)?
    return Ok(num)

// `Option[T]` for optional values (without error)
Option[string] maybe_name = find_name()
string name = maybe_name or "Default"  // `or` allowed only for Option
```

## Types & Variables
### Basic Types
```
int, float, bool, string, char
array[Type, size]        // Fixed-size, owning
darray[Type]             // Dynamic-size, owning
gc[darray[Type]]         // Dynamic-size, in managed heap
dict[Key, Val], set[Type]
vector[Type, dims]       // SIMD vector
&Type, &mut Type         // References
gc<Type>                 // Managed heap pointer
Result[Type, Error]      // Success or Error
Option[Type]             // Some(Type) or None
```

### Variable Declarations
```
int x = 10                     // Owned value
&int ref = &x                  // Immutable reference
gc<string> s = gc<string>("")  // In managed heap
const u32 ID = 0xDEADBEEF      // Compile-time constant

// Unpacking (destructuring)
darray[int] vals = [1, 2, 3]
int a, b, c = *vals

dict[int, string] map = {1: "one"}
darray[int] keys :: darray[string] vals = *map
```

### Predicates (Refined Types)
```
~int positive = ~value > 0
~string non_empty = ~len > 0

fun safe_divide(~int numerator, ~int denominator : ~value != 0) -> int:
    return numerator / denominator  // Compiler knows denominator != 0
```

## Classes & Traits
### Class Declaration
```
// Standard class
class Name(Parent):
    int field
    fun[class] new() -> Name:
        ...
    fun method(&self) -> ReturnType:
        ...

// Linear class (owns resources, non-copyable)
class[linear] File:
    int descriptor
    sfun drop(&mut self):
        os::close(self.descriptor)

// Actor class (message-passing concurrency)
class[actor] BankAccount:
    int balance
    fun receive(&mut self, msg Message):
        switch msg:
            case Deposit(amt):
                self.balance += amt
            case Withdraw(amt):
                self.balance -= amt
```

### Traits (Static Polymorphism)
```
// Trait definition
trait Stringifiable:
    fun to_string(&self) -> string

trait Comparable[Other]:
    fun compare(&self, other: &Other) -> int

// Trait implementation
impl Stringifiable for int:
    fun to_string(&self) -> string:
        return int_to_str(*self)

// Generic function with trait bounds
temp<Type> fun print_all(array[Type] items) where Type: Stringifiable:
    for item in items:
        cli.print(item.to_string())

// Trait objects (dynamic dispatch)
gc[Stringifiable] obj = gc[MyType]()
```

## Operators
### Arithmetic & Logic
```
+ - * / % ^        // Basic arithmetic
|                  // Integer division (floor)
<< >> & | ^ ~      // Bitwise operations
and or xor nand    // Logical
== != < > <= >=    // Comparison
```

### Assignment
```
=                  // Move or copy
+= -= *= /= %=     // Compound arithmetic
<<= >>= &= |= ^=   // Compound bitwise
@=                 // Get address (returns &T)
$=                 // Swap values
```

### Pipeline
```
result = value >>> func1 >>> func2  // func2(func1(value))
result = func2 <<< func1 <<< value  // Same as above
```

## Standard Library Structure
```
std::core           // Core types, traits, intrinsics
std::alloc          // Memory allocators (global, arena)
std::collections    // Data structures (vector, map, set)
std::os             // OS interaction (fs, process, bios*)
std::async          // Async/await, actors, channels
std::libc_safe      // Safe bindings to libc
std::math           // Mathematical functions
std::time           // Timekeeping
std::crypto         // Cryptography
// Note: GUI, decompilation, and Python translation are separate projects.
```

## Advanced Features
### Compile-Time Function Execution (CTFE)
```
fun[const] u32 crc32(string data) -> u32:
    // Pure computation, no side effects
    ...

const u32 CHECKSUM = crc32("YADRO")  // Computed at compile time
```

### Arena Allocation (Deterministic GC)
```
#import std::alloc::arena

fun parse() -> Result[Ast, Error]:
    arena: Arena = Arena.new()
    gc[Node] root = arena.alloc(Node{...})  // Freed when arena drops
    gc[darray[Token]] tokens = arena.alloc(darray[Token][])
    return Ok(parse_tree)
```

### Unsafe Blocks & Inline Assembly
```
#[unsafe]
fun port_in(u16 port) -> u8:
    u8 result
    asm("in %dx, %al"):
        output("al") result
        input("dx") port
    return result

// Unsafe block within safe function
fun wrapper():
    #[unsafe]:
        let ptr = #requires("kernel32::GetProcAddress")
        // ... use raw pointer
```

## Compiler Pipeline
```
1. Directive & Metadata Collection (#target, #requires)
2. Lexical Analysis (Tokenizer)
3. Syntax Analysis (Parser) → AST
4. Semantic Analysis & Borrow Checking → Typed AST
   - Type Inference
   - Lifetime Verification
   - Predicate & Trait Checking
5. Mid-Level IR Generation (MIR)
6. Optimization Passes
7. Code Generation (LLVM IR / Cranelift)
8. Linking (#requires dependencies)
```

## Community (YUPPI) YUP 26.1.2
YUPPI packages follow this structure:
```
package.ymd/
├── main.toml          // Metadata, dependencies
├── setup.yad          // Build script
├── checksum.sha256    // Integrity check
├── src/main.yad       // Main module
├── bin/               // Optional binaries
└── docs/index.html    // Documentation
```

## Syntax Notes
- Indentation is significant (like Python).
- Multiple statements per line require `;`.
- Single statements can span multiple lines.
- Comments use `//` for line and `/* */` for block.
