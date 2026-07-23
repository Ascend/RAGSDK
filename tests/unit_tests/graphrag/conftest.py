#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration for the graphrag unit tests.

The ``mx_rag.graphrag`` package ``__init__`` imports two heavyweight modules
(``graphrag_pipeline`` / ``graph_evaluator``) that pull in the full graphrag
stack (langchain_opengauss, storage, reranker, ...). The multimodal subpackage
itself does not depend on them. In a minimal test environment where those
optional deps are absent, inject lightweight stubs so the multimodal module can
be imported. On a fully provisioned CI environment the real modules import
successfully and no stub is ever injected, so other graphrag tests are unaffected.
"""

import importlib
import sys
import types


def _ensure_stub_if_unavailable(fullname, attr_name):
    try:
        importlib.import_module(fullname)
    except (ImportError, ModuleNotFoundError):
        if fullname not in sys.modules:
            stub = types.ModuleType(fullname)
            setattr(stub, attr_name, type(attr_name, (), {}))
            sys.modules[fullname] = stub


_ensure_stub_if_unavailable("mx_rag.graphrag.graphrag_pipeline", "GraphRAGPipeline")
_ensure_stub_if_unavailable("mx_rag.graphrag.graph_evaluator", "GraphEvaluator")
