from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize([
    'document/loader/docx_section_loader.py',
    'document/loader/docx_loader.py',
    'document/loader/data_clean.py'
]))
