# https://python-packaging.readthedocs.io
from setuptools import setup

setup(
    name="dataTSAR",
    version="0.1",
    description="dataTSAR will help you process your HAWC2 result files in bulk.",
    author="Jaime Liew",
    author_email="jaimeliew1@gmail.com",
    packages=["TSAR"],
    install_requires=["h5py", "numpy", "scipy", "pandas", "rich", "click", "rust-fatigue"],
    entry_points={"console_scripts": ["tsar-inspect=TSAR.cli:inspect"]},
)
