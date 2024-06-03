# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import datetime
import glob
import os
import shutil
import stat
from pathlib import Path

import yaml
from loguru import logger
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
build_folder = ('build/bdist*', 'build/lib')
cache_folder = ('mx_rag.egg-info', '_package_output')
pwd = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(pwd, "build/lib")


def get_ci_version_info():
    """
    Get version information from ci config file
    :return: version number
    """
    src_path = this_directory.parent
    ci_version_file = src_path.joinpath('mindxsdk', 'build', 'conf', 'config.yaml')
    version = '6.0.RC2'
    logger.info(f"get version from {ci_version_file}")
    try:
        with open(ci_version_file, 'r') as f:
            config = yaml.safe_load(f)
            version = config['version']['mindx_sdk']
    except Exception as ex:
        logger.warning(f"get version failed, {str(ex)}")
    return version


def build_dependencies():
    """generate python file"""
    version_file = os.path.join(pkg_dir, 'mx_rag', 'version.py')
    version_file_dir = os.path.join(pkg_dir, 'mx_rag')
    if not os.path.exists(version_file_dir):
        os.makedirs(version_file_dir, exist_ok=True)

    with os.fdopen(os.open(version_file, os.O_WRONLY | os.O_CREAT, mode=stat.S_IRUSR | stat.S_IWUSR), 'w') as f:
        f.write(f"__version__ = '{get_ci_version_info()}'\n")
        f.write(f"__build_time__ = '{datetime.date.today()}'\n")


def clean():
    for folder in cache_folder:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for pattern in build_folder:
        for name in glob.glob(pattern):
            if os.path.exists(name):
                shutil.rmtree(name)


def copy_patches():
    source_folder = this_directory / 'patches'

    target_folder = this_directory / 'mx_rag/patches'

    try:
        shutil.copytree(source_folder, target_folder)
    except Exception as e:
        logger.warning(f"patches folder replication failed, {str(e)}")


clean()

copy_patches()

build_dependencies()

required_package = []

package_data = {'': ['document/loader/*.so', 'patches/*/*']}

excluded = [
    'mx_rag/document/loader/docx_section_loader.py',
    'mx_rag/document/loader/docx_loader.py',
    'mx_rag/document/loader/data_clean.py'
]


class BuildBy(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        res = []
        for pkg, mod, file in modules:
            if file not in excluded:
                res.append((pkg, mod, file))
        return res


setup(
    name='mx_rag',
    version=get_ci_version_info(),
    platforms=['linux', ],
    description='MindX RAG is library to build RAG system',
    python_requires='>= 3.7',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required_package,
    package_data=package_data,
    packages=find_packages(exclude=["*test*"]),
    include_package_data=True,
    cmdclass={
        'build_py': BuildBy,
    },
)

clean()
