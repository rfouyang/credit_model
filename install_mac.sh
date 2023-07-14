#!/bin/sh

conda install -c conda-forge toad
conda install -c anaconda statsmodels
conda install -c conda-forge matplotlib
conda install -c conda-forge lightgbm
pip install -r requirements.txt
pip uninstall pillow
pip install pillow==9.4.0
pip uninstall numpy
pip install numpy==1.23