Python Structural Health Monitoring (SHM) Package
=================================================

This package provides tools of analysis for data of structural health monitoring (SHM) in construction engineering.

=====================
For build & packaging
=====================
On Linux, run packaging.sh in the project root folder.
This script does the following things:
- build the package (binarisation via Cython)
- uncompress the package
- delete the source files that we want to protect
- compress again

The output of packaging.sh is a zip-file in the subfolder dist. It constains two subfolders bin and lib.

On Windows, run 

$python setup.py bdist


================
For distribution
================
For installation simply unzip the dist/ in the root directory of the Anaconda distribution, e.g. /Applications/anaconda. This amounts to copy the content of the subfolder bin/ into /Applications/anaconda/bin and lib/ into /Applications/anaconda/lib.

============================
For open-source distribution
============================
The procedures above are not needed for a open-source distribution.