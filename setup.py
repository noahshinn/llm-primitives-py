from setuptools import find_packages, setup

setup(
    name="llm_primitives",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pydantic>=2.7.2",
    ],
)
