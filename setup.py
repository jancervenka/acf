#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import setuptools

VERSION = "0.2.2"


def run_setup():
    """
    Runs the package setup.
    """

    setup_params = {
        "name": "acf",
        "version": VERSION,
        "description": "Lightweight recommender engine",
        "author": "Jan Cervenka",
        "author_email": "jan.cervenka@yahoo.com",
        "packages": ["acf", "acf.core", "acf.tests"],
        "python_requires": ">=3.7",
        "install_requires": ["pandas>=1.0", "numpy>=1.16"],
    }
    setuptools.setup(**setup_params)


if __name__ == "__main__":
    run_setup()
