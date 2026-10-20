## Installation

Canonical (PEP 517) build from source::

    cd python
    python3 -m pip install .

or build a wheel without installing::

    python3 -m pip wheel --no-deps -w dist .

or run `./build.sh`, which invokes `python3 -m build`.

The build requires a C++17 compiler. On macOS, if the environment `CXX`
points at a broken/missing-headers clang (e.g. Homebrew LLVM without a
sysroot), the build automatically substitutes `/usr/local/bin/g++-16` or
`g++-15` when present.

This should build easily on x86-64.

On M1, you need to add `ARCHFLAGS="-arch arm64"` to your environment or on the function call `ARCHFLAGS="-arch arm64" python3 -m pip install .`.

