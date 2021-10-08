#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021, Jan Cervenka

import setuptools
import pathlib

VERSION = "0.2.3"


def run_setup():
    """
    Runs the package setup.
    """
    
    this_directory = pathlib.Path(__file__).parent

    setup_params = {
        "name": "acf",
        "version": VERSION,
        "description": "Lightweight recommender engine",
        "author": "Jan Cervenka",
        "author_email": "jan.cervenka@yahoo.com",
        "long_description": (this_directory / "README.md").read_text(),
        "long_description_content_type": "text/markdown",
        "packages": ["acf", "acf.core", "acf.tests"],
        "python_requires": ">=3.7",
        "install_requires": ["pandas>=1.0", "numpy>=1.16"],
    }
    setuptools.setup(**setup_params)


if __name__ == "__main__":
    run_setup()
