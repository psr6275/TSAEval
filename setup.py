from setuptools import setup, find_packages
from pathlib import Path

VERSION = "0.0.1"
DESCRIPTION = "TSAEval"
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "readme.md").read_text()

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="tsaeval",
    version=VERSION,
    author="SafeAI Lab",
    author_email="srompark@unist.ac.kr",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch",
        "tensorboard",
        "opacus",
        "tqdm",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "more-itertools",
        "ipykernel",
        # "gensim==3.8.3",        
        "notebook",
        "ipyplot",
        "jupyterlab",
        "statsmodels",
        "gdown",
        "annoy==1.17.1",                
        "ray",
        "ray[default]",
        "multiprocess",
        "addict",
        "config_io==0.4.0",
        "flask",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "tsaeval"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux"
    ],
)