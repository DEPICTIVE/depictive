import os
from setuptools import setup, find_packages

NAME='depictive'
ROOT = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = "A tool for uncovering sources of cell-to-cell variability from single cell data."

try:
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as fid:
        LONG_DESCRIPT = fid.read()
except FileNotFoundError:
    LONG_DESCRIPT = ''
try:
    with open(os.path.join(ROOT, NAME,'__version__'), 'r') as fid:
        VERSION=fid.read().strip()
except FileNotFoundError:
    VERSION='0.0.0error'

setup(name=NAME,
    version=VERSION,
    url='https://github.com/jeriscience',
    python_requires='>=3.6.0',
    long_description=LONG_DESCRIPT,
    long_description_content_type='text/markdown',
    author='Pablo Meyer and Robert Vogel',
    author_email='pmeyerr@us.ibm.com',
    description = DESCRIPTION,
    packages=find_packages(exclude=('examples',)),
    package_data={
        '':['__version__']
    },
    setup_requires=[
        'numpy',
        'scipy',
        'matplotlib'],
    install_requires=[
        'matplotlib',
        'scipy',
        'numpy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Developer',
        'Intended Audience :: End Users',
        'Topic :: biological data analysis'
    ]
)
# 'Development Status :: Alpha',
