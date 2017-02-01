from setuptools import find_packages, setup

setup(
    name="cogsci17_decide",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'matplotlib < 2.0',
        'nengo >= 2.3, < 3.0',
        'numpy',
        'pandas',
        'pytry',
        'seaborn',
    ]
)
