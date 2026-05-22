from setuptools import setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("WARNING: Cython not found. The package will be installed as pure Python.")
    print("To protect IP via binary compilation, run 'pip install Cython' before building.")

ext = '.pyx' if USE_CYTHON else '.py'

extensions = [
    Extension("kalpana.core", ["kalpana/core" + ext]),
    Extension("kalpana.integrations", ["kalpana/integrations" + ext])
]

if USE_CYTHON:
    # Set compiler directives for maximum performance and IP protection
    extensions = cythonize(
        extensions, 
        compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False}
    )

setup(
    name="kalpana-sdk-enterprise",
    version="1.0.0",
    author="Vijñāna AI",
    description="Kalpanā O(1) Holographic Memory Architecture SDK",
    packages=find_packages(),
    ext_modules=extensions,
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
