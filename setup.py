"""For pip."""

from setuptools import setup, find_packages

setup(
    name="embedding",
    version="0.0",
    description="compute word embeddings",
    packages=find_packages(),
    install_requires=[
        "torch",
        "scipy",
        "argcomplete"
    ],
    data_files=[
        ("data", ["data"])
    ],
    entry_points={
        "console_scripts": [
            "embedding = embedding.embedding:main",
        ],
    },
)
