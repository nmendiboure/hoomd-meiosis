#!/usr/bin/env python3
# -*-coding:Utf-8 -*

# To install the package, run :"
# pip install -e ."


import setuptools
import codecs

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Dynamic Molecular",
    "Topic :: Scientific/Engineering :: Meiosis and Recombination",
    "Topic :: Scientific/Engineering :: Polymer Modelling",
    "Operating System :: Unix",
    "Operating System :: MacOS",
]

NAME = "hoomd-meiosis"
MAJOR = 1
MINOR = 0
MAINTENANCE = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MAINTENANCE)


LICENSE = "GPLv3"
AUTHOR = "Nicolas Mendiboure"
AUTHOR_EMAIL = "nicolas.mendiboure@ens-lyon.fr"
URL = "https://github.com/nmendiboure/hoomd-meiosis"


with codecs.open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    description="Stochastic polymer model to study chromosome force challenges in meiosis",
    long_description=LONG_DESCRIPTION,
    url=URL,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    classifiers=CLASSIFIERS,
    python_requires='>=3.11',
    install_requires=REQUIREMENTS,
    # entry_points={
    #     "console_scripts": [
    #     ]
    # },
    zip_safe=False,
)

