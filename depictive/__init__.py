from os import path

PKG_DIR = path.abspath(path.dirname(__file__))

with open(path.join(PKG_DIR, '__version__'), 'r') as fid:
    __version__ = fid.read().strip()

from .depictive import inference
from .depictive import load
