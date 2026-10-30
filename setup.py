from setuptools import setup, Extension
import numpy

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()



comblicorn = Extension("pyhlicorn.cpylibs.comblicorn",
                   sources=["pyhlicorn/cpylibs/comblicorn.c"],
                   include_dirs=[numpy.get_include()])

setup(
    ext_modules=[comblicorn],
)


