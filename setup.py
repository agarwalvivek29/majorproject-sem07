#!/usr/bin/env python3
"""
Setup Script for Deepfake Detection System
==========================================

Installation:
    pip install -e .              # Development install
    pip install -e .[dev]         # With dev dependencies
    pip install -e .[gpu]         # With GPU support
    pip install -e .[all]         # Everything

Build:
    python -m build
    twine upload dist/*
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read version from package
version = {}
with open("src/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

VERSION = version.get("__version__", "0.1.0")

# Read README for long description
README = Path("README.md")
LONG_DESCRIPTION = README.read_text() if README.exists() else ""

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",
    "opencv-python>=4.5.0",
    "mediapipe>=0.10.0",
    "Pillow>=9.0.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "ffmpeg-python>=0.2.0",
    "faiss-cpu>=1.7.0",
    "xgboost>=1.7.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "colorlog>=6.0.0",
    "requests>=2.28.0",
    "tqdm>=4.64.0",
    "click>=8.0.0",
    "typing-extensions>=4.0.0",
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
    ],
    "gpu": [
        "faiss-gpu>=1.7.0",
    ],
    "transformers": [
        "transformers>=4.25.0",
    ],
    "monitoring": [
        "wandb>=0.15.0",
        "mlflow>=2.0.0",
    ],
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

setup(
    name="deepfake-detector",
    version=VERSION,
    author="Deepfake Detection Team",
    author_email="team@example.com",
    description="Audio-Visual Deepfake Detection System using RAD",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/example/deepfake-detector",
    project_urls={
        "Documentation": "https://deepfake-detector.readthedocs.io/",
        "Source": "https://github.com/example/deepfake-detector",
        "Tracker": "https://github.com/example/deepfake-detector/issues",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video :: Analysis",
    ],
    keywords=[
        "deepfake",
        "detection",
        "audio-visual",
        "machine-learning",
        "computer-vision",
        "deep-learning",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    package_dir={"": "."},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "deepfake-train=scripts.train:main",
            "deepfake-detect=scripts.detect:main",
            "deepfake-realtime=scripts.realtime:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
        "config": ["*.yaml"],
    },
    data_files=[
        ("config", ["config/default.yaml", "config/training.yaml"]),
    ],
    zip_safe=False,
)
