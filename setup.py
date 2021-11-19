import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Dippa",
    version="0.0.1",
    author="Oskari Lehtonen",
    author_email="oskari.lehtonen@helsinki.fi",
    description="Benchmarking dl segmentation methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/sfo/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)
