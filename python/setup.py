#!/usr/bin/env python3
"""
Setup script for Weight Codec Python bindings
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("../README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="wcodec",
    version="0.1.0",
    author="Your Name",
    author_email="wcodec-dev@yourorg.com",
    description="A storage codec for LLM weights inspired by video compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/weight-codec",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
)

