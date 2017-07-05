#!/bin/bash
# script for packaging and distribution of pyshm library.

#rm -rf build/ dist/
pname=$(python -c "import platform; print(platform.platform())")  # platform name
echo ${pname}
# python setup.py bdist_dumb --plat-name ${pname} --relative --format=zip  # --relative fails on Windows
# python setup.py bdist --plat-name ${pname} --format=zip  # --relative fails on Windows

# Manually modify the content of the package:
# remove those .py and .c files that we want to protect the source
fname=pyshm-0.1.0.${pname}.zip
echo $fname
# cd ./dist
# unzip pyshm-0.1.0.${pname}.zip
# rm pyshm-0.1.0.${pname}.zip

# # for f in *.c; do
# #     echo rm ./lib/python3.5/site-packages/pyshm/$f
# # done

# for f in {'Kalman','Models','OSMOS','Stat','Tools'}; do
#     rm ./lib/python3.5/site-packages/pyshm/${f}.c
#     rm ./lib/python3.5/site-packages/pyshm/${f}.py
# done
# # cp ../pyshm/__init__.py ./lib/python3.5/site-packages/pyshm/  # copy back __init__.py

# # Compress again
# zip -r pyshm-0.1.0-${pname}.zip bin lib
# rm -rf bin/ lib/
