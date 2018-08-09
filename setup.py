from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#setup(name='Hello world app', ext_modules=cythonize("hello.pyx"), requires=['Cython'])
#setup(name='Cython NumPy Tutorial', ext_modules=cythonize("convolve.pyx"), requires=['Cython'])
#setup(name='Cython Functions', ext_modules=cythonize("cython_functions.pyx"), requires=['Cython'])

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=[],
        library_dirs=[]),
]
setup(
    name="Cython Functions",
    ext_modules=cythonize(extensions),
)
