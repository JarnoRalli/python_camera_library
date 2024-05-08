from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent

setup(
    name="python_camera_library",
    version="1.0.0",
    packages=find_packages(),
    author="Jarno Ralli",
    author_email="jarno@ralli.fi",
    description="Camera library for different types of cameras",
    license="BSD 3-Clause",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Camera",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["numpy", "pykitti", "open3d", "opencv-python-headless"],
)
