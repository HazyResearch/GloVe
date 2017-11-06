"""For pip."""

from setuptools import setup, find_packages

exec(open('embedding/__version__.py').read())

setup(
    name="embedding",
    version=__version__,
    description="compute word embeddings",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numba",
        "scipy",
        "pandas",
        "sparsesvd",
        # "gensim"
    ],
    entry_points={
        "console_scripts": [
            "embedding = embedding.main:main",
        ],
    },
)
