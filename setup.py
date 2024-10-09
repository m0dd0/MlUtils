from setuptools import setup, find_packages

setup(
    name="MlUtils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "hydra-core",
        "omegaconf",
        "opencv-python",
        "matplotlib",
        "torch",
    ],
    author="Moritz Hesche",
    author_email="mo.hesche@gmail.com",
    description="A collection of machine learning utilities",
    url="https://github.com/m0dd0/MlUtils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
