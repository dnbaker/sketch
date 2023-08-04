from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import setuptools


is_on_arm = "arm" in subprocess.check_output(
    "uname -p", shell=True).decode('utf-8').strip()


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
    "../",
    "../libpopcnt",
    "../include",
    "../include/vec",
    "../include/blaze",
    "../..",
    "../pybind11/include"
]

__version__ = "0.19.0"


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
    make_namepair, ('hll', 'bbmh', 'util', 'bf', 'hmh', 'ss', 'lsh'))))

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
        ct = self.compiler.compiler_type
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


setup(
    name='sketch_ds',
    version=__version__,
    author='Daniel Baker',
    author_email='dnb@cs.jhu.edu',
    url='https://github.com/dnbaker/sketch',
    description='A python module for constructing and comparing sketch data structures',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    packages=find_packages()
)
