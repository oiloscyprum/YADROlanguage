# The YADRO Constitution
## YUP 26.1.3

### **Preamble**

This Constitution establishes the foundational, immutable principles of the YADRO programming language. It serves as the supreme authority for all design decisions, implementation work, and evolution of the language, its compiler, and its standard library. Any feature, syntax, or tool that contravenes these principles is incompatible with YADRO.

---

### **Article I: The Primacy of Static Verification**
> **The correctness and safety of a program must be provable at compile time.**

*   **1.1. The Compiler as a Proof Assistant:** The compiler is not merely a translator but a verification engine. Its primary duty is to reject programs it cannot prove are correct within the defined model.
*   **1.2. Runtime is for Dynamism, not for Correctness:** Dynamic checks (`panic`, exception handling) are mechanisms for responding to unpredictable external states (e.g., hardware failure, network partition), not for papering over static logical errors.
*   **1.3. Types Describe Behavior:** The type system must encode not just data structure, but also behavioral contracts—lifetimes, ownership, state predicates, and concurrency contexts.

### **Article II: The Principle of Explicit Intent**
> **Code must unambiguously express programmer intent and its requirements on the execution environment.**

*   **2.1. No Magic:** Implicit global behavior, hidden control flow, and "smart" compiler magic that significantly alters semantics are forbidden. Side effects must be traceable from the source code.
*   **2.2. Declarative Configuration:** All environmental concerns—optimization mode (`safety`/`speed`), concurrency model (`async`/`actor`), execution target (`cpu`/`gpu`)—must be explicitly declared, not inferred.
*   **2.3. Cost must be Apparent:** The resource implications (memory, time, side-channels) of an operation should be discernible from its syntax and type signature.

### **Article III: Zero-Cost Abstraction Hierarchy**
> **Programmers must be able to descend from high-level abstractions to low-level operations without losing performance or control.**

*   **3.1. Abstraction without Overhead:** Using a high-level construct (e.g., an iterator, a pipeline) must generate machine code as efficient as its hand-written, low-level equivalent.
*   **3.2. Seamless Interoperation:** The memory model and type system are uniform across all abstraction levels. Data must move freely between "safe" and "unsafe" contexts without opaque marshaling.
*   **3.3. The Right to Descend:** Escape hatches (`asm`, raw FFI) are first-class, legitimate tools. Their use is gated by `#[unsafe]` not to forbid, but to demarcate and require justification.

### **Article IV: Determinism and Composability**
> **System behavior must be predictable and built from independently verifiable components.**

*   **4.1. Against Global Entropy:** Unmanaged global mutable state is prohibited. The behavior of a function must be determined by its inputs and accessible immutable state.
*   **4.2. Modular Reasoning:** The correctness of a composite system must be inferable from the correctness of its parts and the formal contracts between them. Modules and YUPPI packages are units of verifiable capability.
*   **4.3. Side-Effects are part of the Type:** Functions that perform I/O, modify shared state, or may panic must have this reflected in their type signature or explicit attribute.

### **Article V: The Technical Pillars**

These pillars are the direct, non-negotiable implementation of the above principles.

*   **Pillar 1: Ownership & Borrowing.** Memory safety is guaranteed by a static model of single ownership with checked, scoped references (`&T`, `&mut T`). Linear/unique types are the default.
*   **Pillar 2: Protocol-Based Polymorphism.** Generics are constrained by explicit, compile-time-checked protocols (`protocol[Comparable]`). This enables zero-cost abstraction.
*   **Pillar 3: Algebraic Effects for Errors.** Failure is modeled using the `Result[T, E]` type. The `?` operator provides ergonomics; `match` provides completeness. Exceptions are banned.
*   **Pillar 4: Region-Based Resource Management.** The `arena` construct provides deterministic, fast lifetime management for objects, complementing ownership for complex workflows.

### **Article VI: Governance and Evolution**

*   **6.1. Constitutional Supremacy:** Any change to the language, including YUP proposals, must first be evaluated against this Constitution. A change that violates it is invalid, regardless of its utility.
*   **6.2. Backward Compatibility as a Contract:** Backward compatibility may only be broken to rectify a critical violation of Article I (safety) or Article II (explicit intent), and requires a supermajority of the Architecture Council.
*   **6.3. The Standard Library as Exemplar:** The `std` library shall be minimal, idiomatic, and the primary demonstration of correct YADRO style. It is a user of the language, not its master.
*   **6.4. Community as an Actor System:** The development process (YUP) is modeled as a message-passing system where proposals are messages, the Constitution is the receiving protocol, and the compiler is the enforcing runtime.

---

### **Ratification & Enactment**

This Constitution is enacted immediately upon ratification. It is the lens through which the existing v0.3 Specification is viewed and the framework upon which all future YADRO development is built.

**The Compiler is the Guardian of this Constitution.**