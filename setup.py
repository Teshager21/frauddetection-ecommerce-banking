# setup.py

from setuptools import setup, find_packages

setup(
    name="fraud_detection_ecommerce_banking",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
