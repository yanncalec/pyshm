"""Setup file for pyshm package.
"""

# # Automatically install setuptools
# from ez_setup import use_setuptools
# use_setuptools()

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from setuptools.extension import Extension

# from Cython.Distutils import build_ext
from Cython.Build import cythonize

# To use a consistent encoding
from codecs import open
import os, sys
import glob

import numpy

# Tell Distutils to use C++ compiler
# os.environ["CC"] = "gcc"
# os.environ["CXX"] = "g++"
# os.environ["CC"] = "gcc-6"
# os.environ["CXX"] = "g++-6"

current_path = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(current_path, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# For cython
# must add the line in setup(...):
#    ext_modules = cythonize(extentions),
# in order to compile pyx files

# Strange behavior of Cython:
# if a full list of pyx filenames are given instead of the wildcard *, Cython will build a module using the keyword name (if not given, using the first filename) which includes all .so
# It seems impossible to combine two wildcards: ["pyshm/*.pyx", "pyshm/OSMOS_pkg/*.pyx"] is not accepted
# Note that the name of the extension must be the same as sources
# pyshm_ext = [Extension(name="pyshm",  # the full name of the extension
#                     # sources=["pyshm/*.py"],
#                     sources=["pyshm/Tools.py", "pyshm/Stat.py", "pyshm/Models.py"],...)]

# 2. or explicitly construct the list of extensions like
pyshm_ext = []
for fname in ['Tools.py', 'Models.py', 'Stat.py', 'OSMOS.py']:
    pyshm_ext.append(Extension(name="pyshm."+fname[:-3],  # the full name of the package must be given in order that the compiled library is correctly placed in the folder
                                sources=[os.path.join("pyshm", fname)],
                                include_dirs=[numpy.get_include()],
                                libraries=[],
                                extra_compile_args=["-w"]  # turn off warning
    ))

# Cythonization only for binary build
if sys.argv[1] in ['sdist', 'develop'] :
    ext = []
    # clear all binary files in the source folder
    filelist = glob.glob(os.path.join("pyshm", "*.so"))
    filelist += glob.glob(os.path.join("pyshm", "*.pyd"))
    filelist += glob.glob(os.path.join("pyshm", "*.c"))
    for f in filelist:
        os.remove(f)
else:
    ext = cythonize(pyshm_ext)


setup(
    name='pyshm',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='A Python package for SHM in construction engineering',
    long_description=long_description,

    # The project's main homepage.
    # url='https://github.com/sivienn/pyshm',

    # Author details
    author='Han Wang',
    author_email='han@sivienn.com',

    # Choose your license
    license='Copyright',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        # 'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        # 'Intended Audience :: Developers',
        # 'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        # 'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        # 'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='structure health monitoring, construction engineering',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    # packages=['pyshm'],
    # packages = ['script'] + find_packages(exclude=[], include=['script', 'src']), # "include" works only for  folders containing a __init__.py

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        '':['*.tex', '*.txt', '*.rst'],
        'pyshm': ['*'],
    },

    include_package_data = True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('data', ['data/data_file'])],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
    'numpy>=1.11.0',
    'scipy>=0.18.0',
    'pandas>=0.19.0',
    'matplotlib>=1.5.1',
    'mpld3>=0.2',
    # 'statsmodels>=0.6.1'
    # 'cython>=0.25',
    # 'bokeh>=0.12'
    # 'setuptools>=26.1.1'  # not necessary
    ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        # 'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': [
            'osmos_download_data = pyshm.script.Download_data:main',
            'osmos_preprocess_static_data = pyshm.script.Preprocess_static_data:main',
            'osmos_deconv_static_data = pyshm.script.Deconv_static_data:main',
            'osmos_deconv_static_data_plot = pyshm.script.Deconv_static_data_plot:main',
            'osmos_thermal_static_data = pyshm.script.Thermal_static_data:main',
            'osmos_thermal_static_data_plot = pyshm.script.Thermal_static_data_plot:main'
        ],
    },

    # ext_package='pyshm',  # this will put all compiled libraries in a subfolder named pyshm
    ext_modules = ext
    # ext_modules = cythonize(pyshm_ext)  # setuptools_cython module contain bugs
    # cmdclass = {'build_ext': build_ext}
)