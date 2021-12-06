import os
import os.path

import subprocess
from setuptools import find_packages, setup

def req_file(filename, folder=""):
    with open(os.path.join(folder, filename)) as f:
        content = f.readlines()
    return [x.strip() for x in content]


  
  
if os.path.exists("scripts/download.sh"):
    print(subprocess.run(["sh", "scripts/download.sh"], capture_output=True))
	
setup(
    name="feature_transformer_made_marusya",
    packages=find_packages(),
    version="0.1.0",
    description="For graduation project in MADE academy",
    #extras_require=extras,
    author="arhimisha",
    install_requires=req_file("requirements.txt"),
    license="MIT",
)
