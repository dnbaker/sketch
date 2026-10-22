from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import setuptools


# On macOS, the environment CXX may point at a broken Homebrew LLVM
# clang (missing system headers). Substitute a Homebrew gcc (g++-16
# preferred, then g++-15) BEFORE any compiler object is created; the
# build_ext substitution below is belt-and-suspenders for compilers that
# ignore CC/CXX.
if sys.platform == 'darwin' and 'clang' in os.environ.get('CXX', ''):
    for _candidate in ('/usr/local/bin/g++-16', '/usr/local/bin/g++-15'):
        if os.path.exists(_candidate):
            os.environ['CXX'] = _candidate
            os.environ['CC'] = _candidate.replace('g++', 'gcc')
            break

is_on_arm = "arm" in subprocess.check_output(
    "uname -p", shell=True).decode('utf-8').strip()

CWD = os.environ["SKETCH_DS"]

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


march_flag = "-march=native" if not is_on_arm else "-mcpu=apple-m1"

gomplink_flag = "-lgomp" if "clang" not in os.environ.get(
    'CC', '') else "-lomp"

openmp_flag = []
if os.environ.get("NO_OMP"):
    openmp_flag += ["-fopenmp", gomplink_flag]

extra_compile_args = [march_flag,
                      '-Wno-char-subscripts', '-Wno-unused-function',
                      '-Wno-strict-aliasing', '-Wno-ignored-attributes', '-fno-wrapv',
                      '-lz', '-DNDEBUG',
                      '-DBLAZE_SHARED_MEMORY_PARALLELIZATION=0', "-O2"] + openmp_flag

include_dirs = [
    # Path to pybind11 headers
    get_pybind_include(),
    get_pybind_include(user=True),
    CWD + "/../",
    CWD + "/../libpopcnt",
    CWD + "/../include",
    CWD + "/./include",
    CWD + "/../include/vec",
    CWD + "/../include/blaze",
    CWD + "/../..",
    CWD + "/../pybind11/include",
    CWD
]

__version__ = "0.20.2"


def make_namepair(name):
    return ('sketch_' + name, name + ".cpp")


def make_module(namecppf):
    name, cppf = namecppf
    return Extension(
        name,
        [cppf],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args
    )


ext_modules = list(map(make_module, map(
    make_namepair, ('hll', 'bbmh', 'util', 'bf', 'hmh', 'ss', 'lsh', 'ull', 'ell', 'fmh', 'dmh', 'pmh', 'cqf', 'bcqf'))))

'''
ext_modules = [
    Extension(
        'sketch_hll',
        ['hll.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'sketch_bbmh',
        ['bbmh.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args
    ),
    Extension(
        'sketch_util',
        ['util.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args
    ),
]
'''


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++20', '-std=c++2a', '-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


extra_link_opts = [gomplink_flag, "-lz"] if openmp_flag else ["-lz"]


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        # darwin_opts = ['-mmacosx-version-min=10.7']# , '-libstd=libc++']
        darwin_opts = []
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        print(os.getcwd(), " is PWD during building", file=sys.stderr)
        ct = self.compiler.compiler_type
        # On macOS, if the environment CXX resolves to a broken/missing-
        # headers clang (e.g. Homebrew LLVM without a sysroot), fall back
        # to a Homebrew gcc when available (g++-16 preferred, then g++-15).
        if sys.platform == 'darwin':
            # The environment CXX may resolve to a broken Homebrew LLVM
            # clang (missing system headers on this machine). Substitute a
            # Homebrew gcc (g++-16 preferred, then g++-15): swap the
            # executable and strip clang/llvm tokens (including the
            # llvm -I that drags in its libc++).
            for candidate in ('/usr/local/bin/g++-16', '/usr/local/bin/g++-15'):
                if not os.path.exists(candidate):
                    continue
                cc = candidate.replace('g++', 'gcc')
                try:
                    if self.compiler.compiler_cxx and 'clang' in self.compiler.compiler_cxx[0]:
                        self.compiler.compiler_cxx = [
                            e for e in self.compiler.compiler_cxx
                            if 'clang' not in e and '/llvm/' not in e]
                        self.compiler.compiler_cxx.insert(0, candidate)
                    if self.compiler.compiler_so and 'clang' in self.compiler.compiler_so[0]:
                        self.compiler.compiler_so = [
                            e for e in self.compiler.compiler_so
                            if 'clang' not in e and '/llvm/' not in e]
                        self.compiler.compiler_so.insert(0, cc)
                except (AttributeError, IndexError):
                    pass
                break
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_compile_args += extra_compile_args
            ext.extra_link_args = link_opts + extra_link_opts
        build_ext.build_extensions(self)

print(os.getcwd(), " is PWD", file=sys.stderr)


# Static metadata (name, version, description, ...) lives in
# pyproject.toml ([project]); this file only supplies the extension modules
# and the custom build_ext (which substitutes a working gcc on macOS --
# see BuildExt.build_extensions).
setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
