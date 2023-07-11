## Installation

Use `python3 setup.py install` or pip to build and install.

This should build easily on x86-64.

On M1, you need to add `ARCHFLAGS="-arch arm64"` to your environment or on the function call `ARCHFLAGS="-arch arm64" python3 setup.py install`.
