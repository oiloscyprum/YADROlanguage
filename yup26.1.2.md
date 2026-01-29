# YUP 26.1.2: Yadro User Project Package Index (YUPPI) Specification

## Metadata
- **YUP ID:** 26.1.2
- **Title:** Yadro User Project Package Index (YUPPI) Specification
- **Author:** CyrOil 
- **Status:** Proposed
- **Created:** 2026-01-27
- **Governs:** YADRO Language Specification v0.2.0+

## 1. Motivation & Overview

The **Yadro User Project Package Index (YUPPI)** is the official package manager and distribution system for YADRO modules, libraries, and tools. It provides:

1. **Dependency Resolution** - Automatic handling of module dependencies
2. **Version Management** - Semantic versioning with compatibility checking
3. **Security** - Cryptographic verification of package integrity
4. **Cross-Platform** - Consistent package format across all supported targets
5. **Ecosystem Integration** - Seamless integration with YADRO compiler toolchain

YUPPI enables the YADRO community to share, distribute, and reuse code while maintaining the language's safety and performance guarantees.

## 2. Core Concepts

### 2.1 Package
A **YUPPI Package** is a distributable unit containing:
- YADRO source code modules
- Optional native binaries (platform-specific)
- Metadata and configuration files
- Documentation
- Integrity verification files

### 2.2 Repository
A **YUPPI Repository** is a collection of packages with:
- Package metadata database
- Version history
- Dependency graphs
- Digital signatures

### 2.3 Registry
The **YUPPI Registry** is the official central repository maintained by the YADRO project, but users can configure multiple repositories.

## 3. Package Structure Specification

### 3.1 Directory Layout
```
package-name.ymd/
├── main.toml                 # REQUIRED: Package manifest
├── setup.yad                 # REQUIRED: Build/install script
├── checksum.sha256           # REQUIRED: Package integrity
├── LICENSE                   # OPTIONAL: License file
├── README.md                 # OPTIONAL: Documentation
│
├── src/                      # OPTIONAL: Source code root
│   ├── main.yad              # Main module entry point
│   ├── module1.yad           # Additional modules
│   └── subdir/
│       └── module2.yad
│
├── include/                  # OPTIONAL: C/C++ headers (for FFI)
│   ├── wrapper.h
│   └── ffi_interfaces.h
│
├── bin/                      # OPTIONAL: Pre-compiled binaries
│   ├── linux-x86_64/
│   │   └── libmodule.so
│   ├── windows-x86_64/
│   │   └── module.dll
│   └── macos-arm64/
│       └── libmodule.dylib
│
├── lib/                      # OPTIONAL: Static libraries
│   └── libmodule.a
│
└── docs/                     # OPTIONAL: Documentation
    ├── index.html
    ├── api-reference.md
    └── examples/
        └── basic_usage.yad
```

### 3.2 Package Manifest (`main.toml`)
```toml
# REQUIRED SECTION: Package Identification
[package]
name = "example-module"           # Lowercase, hyphens allowed
version = "1.2.3"                 # Semantic Versioning (major.minor.patch)
description = "Example module for YADRO"
authors = ["Author Name <email@example.com>"]
license = "MIT OR Apache-2.0"     # SPDX license identifier
repository = "https://github.com/user/repo"
documentation = "https://docs.example.com"
readme = "README.md"              # Path to readme file (relative)
keywords = ["networking", "async", "web"]
categories = ["network", "utility"]

# OPTIONAL SECTION: Compilation Requirements
[features]
default = ["ssl", "compression"]  # Default enabled features
ssl = []                          # Feature without dependencies
compression = ["zlib", "lz4"]     # Feature with sub-dependencies
experimental = []                 # Optional unstable features

[build]
targets = ["x86_64-linux", "x86_64-windows", "aarch64-macos"]
min_yadro_version = "0.2.0"
max_yadro_version = "1.0.0"
build_script = "setup.yad"        # Default build script

# OPTIONAL SECTION: Dependencies
[dependencies]
# YADRO package dependencies
"std-async" = ">=1.0.0, <2.0.0"
"crypto-library" = { version = "0.5.2", features = ["aes-256"] }

# System library dependencies (via #requires)
[requires.system]
libc = ">=2.31"
openssl = ">=3.0.0"

# Native DLL/SO dependencies
[requires.native]
linux = ["libssl.so.3", "libcrypto.so.3"]
windows = ["ws2_32.dll", "advapi32.dll"]

# Development dependencies (for tests, examples, benchmarks)
[dev-dependencies]
"test-framework" = "0.3.1"
"benchmark-suite" = "0.2.0"

# OPTIONAL SECTION: Module Exports
[exports]
modules = ["src/main.yad", "src/networking.yad"]
binaries = ["tools/cli-tool"]     # Compiled executables to install
includes = ["include/"]           # C headers to expose

# OPTIONAL SECTION: Compatibility
[compatibility]
yadro = ">=0.2.0"
os = ["linux", "windows", "macos"]
arch = ["x86_64", "aarch64"]
libc = ["glibc>=2.31", "musl"]

# OPTIONAL SECTION: Safety & Security
[security]
checksum = "sha256:abc123..."     # Package checksum
signature = "pgp:..."             # PGP/GPG signature
audited = true                    # Security audit status
unsafe_operations = ["ffi", "asm"] # Used unsafe features
```

### 3.3 Build Script (`setup.yad`)
```yadro
#!/usr/bin/env yadro
// YADRO build script for YUPPI packages

#import
std::core::cli
std::os::fs
std::process

// Build configuration
const BUILD_MODE = "release" // or "debug"
const TARGET = #target

// Main build function
fun build() -> Result[Unit, BuildError]:
    cli.print("Building ${package.name} v${package.version}")
    
    // 1. Create build directory
    fs::create_dir("build")?
    
    // 2. Compile YADRO sources
    compile_yadro_sources()?
    
    // 3. Compile native components (if any)
    if fs::exists("native/"):
        compile_native_components()?
    
    // 4. Run tests
    run_tests()?
    
    cli.print("Build completed successfully!")
    return Ok(())

// Helper: Compile all YADRO source files
fun compile_yadro_sources() -> Result[Unit, BuildError]:
    sources = fs::find_files("src/", "*.yad")
    for source in sources:
        result = process::exec("yadro", ["compile", "-O", BUILD_MODE, source, "-o", "build/"])?
        if result.exit_code != 0:
            return Err(BuildError("Compilation failed: " + result.stderr))
    return Ok(())

// Entry point
fun main() -> int:
    switch build():
        case Ok(_):
            return 0
        case Err(e):
            cli.print_error("Build failed: " + e.message)
            return 1
```

### 3.4 Integrity File (`checksum.sha256`)
```
# SHA256 checksums for package verification
# Generated: 2026-01-29T12:00:00Z
# Algorithm: SHA256

src/main.yad: abc123def456...789abc123def456789abc123def456789abc123def456789
src/module.yad: def456abc123...789def456abc123789def456abc123789def456abc123
main.toml: 123abc456def...789123abc456def789123abc456def789123abc456def
setup.yad: 456def123abc...789456def123abc789456def123abc789456def123abc

# TOTAL PACKAGE CHECKSUM:
package.ymd: a1b2c3d4e5f6...789a1b2c3d4e5f6789a1b2c3d4e5f6789a1b2c3d4e5f6
```

## 4. YUPPI Command Line Interface

### 4.1 Basic Commands
```bash
# Package Management
yadro yuppi install <package>           # Install package
yadro yuppi install <package>@<version> # Install specific version
yadro yuppi install -g <package>        # Install globally
yadro yuppi uninstall <package>         # Remove package
yadro yuppi update <package>            # Update to latest version
yadro yuppi update --all                # Update all packages

# Dependency Management
yadro yuppi add <package>               # Add to project dependencies
yadro yuppi add -D <package>            # Add dev dependency
yadro yuppi remove <package>            # Remove dependency
yadro yuppi list                        # List installed packages
yadro yuppi tree                        # Show dependency tree

# Project Operations
yadro yuppi init                        # Initialize new package
yadro yuppi build                       # Build current package
yadro yuppi test                        # Run package tests
yadro yuppi publish                     # Publish to registry
yadro yuppi audit                       # Security audit

# Query Operations
yadro yuppi search <query>              # Search packages
yadro yuppi info <package>              # Show package info
yadro yuppi versions <package>          # List available versions
yadro yuppi outdated                    # Show outdated packages

# Repository Management
yadro yuppi repo add <name> <url>       # Add repository
yadro yuppi repo list                   # List repositories
yadro yuppi repo remove <name>          # Remove repository
yadro yuppi repo update                 # Update repository indexes
```

### 4.2 Advanced Usage Examples
```bash
# Install with specific features
yadro yuppi install network-lib --features ssl,compression

# Install from specific repository
yadro yuppi install private-module --repo company-internal

# Force reinstallation
yadro yuppi install --force-reinstall broken-package

# Dry run (simulate installation)
yadro yuppi install large-package --dry-run

# Install from local path
yadro yuppi install ./path/to/local/package

# Install from Git repository
yadro yuppi install https://github.com/user/repo.git
yadro yuppi install git@github.com:user/repo.git#branch

# Install with version constraints
yadro yuppi install "package>=1.0.0,<2.0.0"
```

## 5. Integration with YADRO Compiler

### 5.1 Compiler Directive Support
YUPPI packages integrate seamlessly with YADRO's compiler directives:

```yadro
// In user code - YUPPI automatically resolves these
#import
network::http        // From installed YUPPI package
crypto::aes256       // From installed YUPPI package

// YUPPI automatically provides these to the compiler
#requires
libssl.so.3          // From package's [requires.native] section
```

### 5.2 Dependency Resolution Algorithm
1. **Parse** `main.toml` from all dependencies
2. **Construct** dependency graph with version constraints
3. **Resolve** conflicts using semantic versioning rules
4. **Select** compatible versions (latest satisfying all constraints)
5. **Download** packages from configured repositories
6. **Verify** integrity using checksums
7. **Compile** in topological order

### 5.3 Cache Management
YUPPI maintains:
- **Source cache**: Downloaded package sources
- **Build cache**: Compiled artifacts (per configuration)
- **Metadata cache**: Repository indexes and package metadata

## 6. Repository Server Specification

### 6.1 API Endpoints
```
GET /index.json                    # Repository index
GET /packages/<name>/index.json    # Package versions
GET /packages/<name>/<version>.ymd # Package download
GET /packages/<name>/<version>.toml # Package metadata
GET /search?q=<query>              # Package search
```

### 6.2 Repository Index Format
```json
{
  "repository": "official",
  "timestamp": "2026-01-29T12:00:00Z",
  "packages": {
    "package-name": {
      "latest": "1.2.3",
      "versions": ["1.2.3", "1.2.2", "1.2.1"],
      "metadata": {
        "1.2.3": {
          "dependencies": {...},
          "checksum": "sha256:...",
          "size": 1048576,
          "downloads": 1500
        }
      }
    }
  }
}
```

## 7. Security Considerations

### 7.1 Integrity Verification
- SHA256 checksums for all package files
- Optional PGP/GPG signatures for packages
- Repository signing with hierarchical keys
- Certificate pinning for repository connections

### 7.2 Sandboxing
- Build scripts run in restricted environment
- Network access disabled during build by default
- Filesystem access limited to package directory
- System call filtering for native compilation

### 7.3 Audit Trail
- All installations logged with package hashes
- Version history maintained for rollback capability
- Dependency provenance tracking
- Security vulnerability database integration


## 9. Reference Implementation

The reference implementation consists of:
- `yuppi-core`: Package management logic (YADRO)
- `yuppi-client`: Command-line interface (YADRO)
- `yuppi-server`: Repository server (YADRO + optional)
- `yuppi-registry`: Official central registry

## 10. Future Extensions (Planned for YUP 26.1.4 or 26.1.6)

1. **Binary Package Distribution**: Pre-compiled packages for common platforms
2. **Plugin System**: Extensible YUPPI with custom commands
3. **Workspace Support**: Multiple related packages in one repository
4. **Offline Mode**: Full functionality without network access
5. **Dependency Vulnerability Scanning**: Automatic security updates

---
*This YUP document formalizes the YUPPI system as the official package management solution for YADRO, ensuring a consistent, secure, and efficient ecosystem for language development and distribution.*

