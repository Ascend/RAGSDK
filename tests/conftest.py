#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest configuration file for RAGSDK tests.
This file is automatically loaded by pytest before running any tests.
"""

try:
    from paddle.base import libpaddle  # noqa: F401
except ImportError:
    pass