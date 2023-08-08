## Installation

Run `./build.sh`, which should call `python3 -m build`.

We currently support `python3 setup.py install`, but this may be deprecated.

Use `python3 setup.py install` or pip to build and install.

This should build easily on x86-64.

On Apple silicon, you may need to add `ARCHFLAGS="-arch arm64"` to your environment or on the function call.
