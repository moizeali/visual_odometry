"""
Setup script for Enhanced Visual Odometry System
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("backend/requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="visual-odometry-enhanced",
    version="1.0.0",
    author="Syed Moiz Ali",
    author_email="moizeali@gmail.com",
    description="Enhanced Visual Odometry System with Web Interface and Data Collection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moizeali/visual_odometry_enhanced",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vo-server=backend.app:main",
        ],
    },
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
)