# YADRO Language Specification
## Overview
YADRO (Your Architecture's Definitive Runtime Optimizer)
 made by CyrOil
   - version 0.0.1 alpha
   - date: 2026-01-27
   - description: a language for operating systems development and low level programming with high level syntax and optimizations
   - YUP: 26.1.1 updated by YUP 26.1.2 of 2026-01-28


## Compiler Directives
### #direct
Specifies compilation environment and optimization mode
```
#direct
windows //os (Windows, Linux, Core_mode, Android)
x86-64 //architecture (x86-64 / arm / risk-v/ wasm)
safety //optimization features (safety or speed)
```
### #plugin
```
#plugin
safety-opt
linter
memory_watcher==0.5
user_plugin.yad
```
### #requires
Lists DLLs to import
```
#requires
kernel32.dll
user32.dll
```

### #import
Imports YADRO modules
```
#import
cli
uix as ui
os.system as sys
```

### #define
Predefines variables
```
#define
int a
float b=0.0
int size=10
array int[size] // array of static size
darray int // array of dynamic size
set int //set
dict int,string //dictionary
```

### #start / #end
Marks main program body boundaries

## Control Structures
### Branching
```
// if-else
if rule:
    ...
elsif rule:
    ...
else:
    ...

// switch-case
switch var:
case val1:
    ...
case val2:
    ...
default:
    ...

// try-catch
try:
    ...
catch Exception as e1:
    ...
catch Exception as e2:
    ...
finally:
    ...
```

### Loops
```
// for loop
for vartype varname from start to end step step:
    ...
// alternative
for vartype varname in (1,100,3):
    ...

// while
while rule:
    ...

// repeat-until
    repeat
    ...
until rule
```

## Functions
### Function Types
```
// classic function
fun name (type var, type var, type var=value) -> type:
    ...
    return value

// asynchronous function
fun[async] name (type var, type var, type var=value) -> type:
    ...
    return value

// coroutined function
fun[coro] name (type var, type var, type var=value) -> type:
    ...
    return value

// threaded function
fun[thread] name (type var, type var, type var=value) -> type:
    ...
    return value

// only GPU computed function
fun[gpu] name (GPUBuffer input) -> GPUBuffer:
    ...
    return value
// CPU with using GPU computed function
fun[cgpu] name (type var, type var, type var=value) -> type:
    GPUBuffer gpu_data ...
    return value
// nonreturn function
fun[nonret] name (type var, type var, type var=value):
    ...

// lambda function
a = lambda(type var, type var) => (operations)

// function templating
temp<type as Type> fun name(Type var) -> Type:
    ...
    return value
```
```
//also we can do this
//carry
fun abc(int a)->(int b)->int:
    return lambda(int b)=>(a+b)

int abc2=abc(2)
cli.print(abc2(4)) //=6

//partial Use

fun avc(int a, int b)-> int:
    return a+b

fun tavc=avc(1)
cli.print(tavc(32)) //=33
```
## Error handling
at functions
```
fun read_file(string path) throws IOError -> string:
    if !file_exists(path):
        throw IOError(f"Missing: {path}")
    return load_contents(path)
// if you need define multiple errors 
fun read_file(string path) throws IOError,TypeError,... -> string:
```
fallback
```
string val = read_file("file.txt") or "File is empty" //on error gets second value
string val = read_file("file.txt") or return Error(IOSError()) // early return with custom error
string val = read_file("file.txt") or throw CustomError("IDK") // convert error
string val = scan_dir("C:/Windows/System32") or panic("Unresolvable system crash") // on critic situations

```

## Classes
```
// Standard class
class Name(Parent):
fun[class] init (Name me, type vars) -> Name:
    type $.var = vars

fun nname (Name me, type lol) -> None:
    ...
fun[class] add (Name me, int value) -> Name:
    ...

// Templated class
temp<type as Type> class name(Parent):
    ...

```
### Classes types
#### linear
```
class[linear] Reader://Reader linear class
    fun open(string a,string mode)->$
    fun read()->string
    sfun drop()

Reader file=Reader("file.txt","r")
string contents = file.read()
// if file length remaining to read is 0 class drops(drop rules could be written by user)
file.read() // causes Exception -> File is already closed
```
#### module
```
class[module] Reader://Reader linear class
    fun open(string a,string mode)->$
    fun read()->string
    sfun close()
```
```
module file=import("approve.yad.Reader")  //file.moduleName
```
#### actor
```
class[actor] Reader:
    // defenition of accessed commands to actor
    # Inc(int)
    # Dec(int)
    # Get(sender)
    # send(command)
    int balance = 0
    fun receive(message MSG):
        switch MSG:
            case Inc(int val):
                balance+=val
            case Dec(int val):
                balance-=val
            case Get(sender Actor):
                Actor.send(Value(balance))
Reader wallet = Reader()
wallet.send(Inc(100))
walled.send(Get(wallet))
//returns 100
//this model could be released between two classes
```

### Special Class Operators
```
op= // assignation
op== // equals
op>= // not less
op<= // not more
op!= // not equals
op> // more
op< // less
op# // length
op* // multiply
op^ // pow
op/ // divide
op| // divide without remainder
op% // get remainder
op- // substract
op+ // summ
op<< // binary move by n to left
op>> // binary move by n to right
op=> // assign if more
op=< // assign if less
op=>= // assign if not less
op=<= // assign if not more
op=!= // assign if not equals
op->
```

## Operators
### Pipeline-operators
```
result= operation(1)>>>operation(2)>>>operation(3)>>>operation(4)
// equals
result= operation(operation(operation(operation(1),2),3),4)
// equals
result= operation(4)<<<operation(3)<<<operation(2)<<<operation(1)
```
### Assignment & Comparison
```
a = b // assignation
a == b // equals
a >= b // not less
a <= b // not more
a != b // not equals
a > b // more
a < b // less
```

### Arithmetic
```
a + b // summ
a - b // substract
a * b // multiply
a / b // divide
a | b // divide without remainder
a % b // get remainder
a ^ b // pow
#a    // length
|a|   // absolute
```

### Bitwise
```
a << n // binary move by n to left
a >> n // binary move by n to right
```

### Compound Assignment
```
a -= b // a = a - b
a += b // a = a + b
a ^= b // a = a ^ b
a *= b // a = a * b
a \= b // a = a \ b
a |= b // a = a | b
a %= b // a = a % b
a <<= b // a = a << b
a >>= b // a = a >> b
```

### Special Assignment
```
a => b // assign if more
a =< b // assign if less
a =>= b // assign if not less
a =<= b // assign if not more
a =!= b // assign if not equals
a @= b // get memory address
a $= b // swap variables
```

### Logical
```
a and b
a or b
a xor b
a nand b
any a
all a
a -> b // implication
```

## Variable Types
```
int        // integer
float      // float
string     // string
array      // array
darray     // dynamic array
dict       // dictionary
set        // set
var        // dynamic typisation
bool       // boolean
vector     // vector
```
## unpackage
```
darray int a = [10,20,30]
int b, c, d = *a
//b:10 c:20 d:30

dict int, string dd = {1:"lol", 2:"331",3: "malware"}
darray int arr :: darray string barr = *dd 
int a1,a2,a3 :: string b1,b2,b3 = **dd
```

## Predicates
```
~int name = ~value>10 //predicates that int name is higher than 10
~string str = ~len>0 // predicates that string length is more 0
cli.print(10/name)  // compiler knows that name!=0 and won't raise the error
```
## Memory Management
- Automatic memory allocation for dynamic types
- Garbage collected or manually cleared with `del var`

## Standard Library Modules
```
cli           - console operations
os            - OS operations
network       - networking
math          - complex math operations
iterative     - iterative operations
time          - timing
bios_tools    - BIOS interaction
crypto        - cryptography (SHA256/512, RSA, AES256/512, etc.)
gui_tools     - GUI using OS tools
pyTranslate   - Python translation
decompile     - assembly to YADRO
asm           - inline assembly
hw_tools      - hardware communication
emails        - email protocols
http          - HTTP/HTTPS requests
```

## Data Formats
### Strings
```
"" - classic
'' - classic
'''''' - multiline strings
f"{variable} text {another variable}" - f-strings
```

### Numeric
```
0           - int
0.0         - float
0.0+2i      - complex
0x2a        - hex
0b10110     - bin
0o2137      - octal
1.2e+08     - exponential
```

### Collections
```
[1,2,3,4]           - classic array
[1 to 20 step 2]    - generative array
[1:20:2]            - synonym generative
{1,2,4,3}           - classic set
{a:12, b:14, c:16}  - classic dict
```

## Standard Library Functions
```
strToInt(string var) -> int
intToStr(int var) -> string
floatToStr(float var) -> string
strToFloat(string var) -> float
complexToStr(complex var) -> string
strToComplex(string var) -> complex
arrayToStr(array var) -> string
strToArray(string var, sep="") -> array
call(coroutine_function)
callAsm('''assembly code''')
```

## Compiler Pipeline
```
usage_collection -> importing -> lexer -> parser -> ast_tree -> IR -> assembly -> compilation -> linking
```

### Debugger Features
- Safety mode with virtual machine
- Generates ISO for VirtualBox/VMware
- Virtual process isolation
- Memory slice viewing
- Variable value inspection

## Community & Development
### YUP (Yadro Update Projects)
- Y.U.P numbering system: Year.UserId.ProjectId (e.g., 26.1.1)
- Monthly community voting
- Selected projects enter development pipeline
### YUPPI(Yadro user project package index)
- YUPPI modules should have structure:
 ```
 library.ymd
  |__bin    //if binaries are required(.dll,.elf and etc)
  |__module 
       |___main.yad  // structurizer
       |___........
  |__main.toml        // info
  |__setup.yad        // setup script
  |__checksum.md5     // checksum for safety
  |__docs
       |___index.html // docs index
       |___.........
 ```
 - Configuration and requirements should be in main.toml of structure
 ```toml
 package_name="Example module"
 version=0.1
 [checksums]
 md5="Hash_key"

 [requires]
 [[dll]]

 [dll_package_name]
 name="example"
 version="example"
 description="example"

 [[yad]]

 [module_name]
 name="example"
 version="<=1.0"
 ```
 - it should be called using command 
 ```bat
 yadro yuppi install package-name
  ```
## Macros system
```
#macro macroname(act,need):
        if !(act==need):
            panic(act,need)
macroname!(compute(),42)
```

## Syntax Notes
- Multiple statements per line separated by `;`
- Single statements can span multiple lines
- Line breaks or semicolons required between statement