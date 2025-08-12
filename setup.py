from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build SYCL extensions")
        
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_extension(ext)
        
        super().run()
    
    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]
        
        # Add Intel oneAPI compiler if available
        if 'ONEAPI_ROOT' in os.environ:
            cmake_args.append(f'-DCMAKE_CXX_COMPILER=icpx')
        
        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        
        # Set build parallel level
        build_args += ['--', '-j4']
        
        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DNDEBUG'
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Run CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, 
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, 
                              cwd=self.build_temp)

# Check if SYCL build is requested
build_sycl = os.environ.get('BUILD_SYCL', '0') == '1'

ext_modules = []
cmdclass = {}

if build_sycl:
    ext_modules.append(
        CMakeExtension('sp_aurora.sycl_flash_attn', 'sp_aurora/sycl')
    )
    cmdclass['build_ext'] = CMakeBuild

setup(
    name="sp_aurora",
    version="0.1",
    author="zhuzilin",
    url="https://github.com/zhuzilin/ring-flash-attention",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "torch>=2.0.0",
        "intel-extension-for-pytorch>=2.0.0",
    ],
    extras_require={
        'sycl': [
            'pybind11>=2.6.0',
            'cmake>=3.20.0',
        ]
    },
    python_requires=">=3.7",
)