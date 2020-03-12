"""
Install in development mode:
python3 setup.py develop
or
pip install -e .[lib]
"""

import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bengali-utils",
    version="0.0.1",
    author="Artur Klibyshev",
    author_email="sanderson180610@gmail.com",
    description="",
    # license="",
    keywords="kaggle bengali-ai",
    packages=find_packages(exclude=["tests"]),
    # long_description=read('README'),
    classifiers=[
        'Topic :: Utilities',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow',
        'matplotlib',
        'scikit-learn',
        'torch',
        'torchvision',
        'albumentations>=0.4.5',
        'opencv-contrib-python',
        'tqdm>=4.43.0',
        'pretrainedmodels',
        'efficientnet_pytorch',
        'iterative-stratification',
        'kaggle',
    ],
)
