import os
from setuptools import find_packages, setup

def req_file(filename, folder=""):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content]

setup(
    name="feature_transformer_marysyaka",
    packages=find_packages(),
    version="0.1.0",
    description="For graduation project in MADE academy",
    author="arhimisha",
    install_requires=req_file("requirements.txt"),
    license="MIT",
)
