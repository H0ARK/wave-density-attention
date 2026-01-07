from setuptools import setup, find_packages

setup(
    name="wave-dencity",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "scipy",
        "tqdm",
        "bitsandbytes",
    ],
    description="Wave-Density Attention implementation",
    author="GitHub Copilot",
    url="https://github.com/conrad/wave-dencity",
)
