#!/bin/bash
# script for packaging and distribution of pyshm library.

rm -rf build/ dist/
pname=$(python -c "import platform; print(platform.platform())")  # platform name
python setup.py bdist_dumb --plat-name $pname --relative --format=zip

# Manually modify the content of the package:
# remove those .py and .c files that we want to protect the source
cd dist
unzip pyshm-0.1.0.${pname}.zip
rm pyshm-0.1.0.${pname}.zip
rm ./lib/python3.5/site-packages/pyshm/*.c
rm ./lib/python3.5/site-packages/pyshm/*.py
cp ../pyshm/__init__.py ./lib/python3.5/site-packages/pyshm/  # copy back __init__.py

zip -r pyshm-0.1.0-${pname}.zip bin lib
rm -rf bin/ lib/
