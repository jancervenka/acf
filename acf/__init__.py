#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

from .core.computation import Engine
from .core import metrics

from .version import __version__

__doc__ = """
Lightweight recommender engine for implicit feedback datasets.

The package implements a collaborative filtering algorithm as described
in "Collaborative Filtering for Implicit Feedback Datasets" paper by
Yifan Hu, Yehuda Koren, Chris Volinsky (https://doi.org/10.1109/ICDM.2008.22).
"""
