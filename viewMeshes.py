"""
This script will get any *.bms (pyBERT/gImli mesh files) in the current directory and display them all, one-by-one.
"""

import pygimli as pg
import glob
import os

# Show all the files
#path =r'F:\WinPython-64bit-3.6.3.0Qt5\settings\New folder\Stable_025mday\dd\5m'
path = os.getcwd()
os.chdir(path)
all_files = glob.glob(os.path.join(path, "*\*.bms"), recursive = True)

for i in all_files:
    mesh = pg.load(i)
    print(i,mesh)
    ax = pg.show(mesh)
    ax[0].set_title(i)