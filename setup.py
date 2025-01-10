from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt", "r") as req_file:
        return [line.strip() for line in req_file]


def read_description():
    with open("README.md", "r") as desc_file:
        return desc_file.read()


setup(  
    name = "Mneme",
    version = "0.1.0",
    author = "Hadjidoukas Panagiotis, Metaxakis Dimitris, Sofotasios Argyris",
    author_email = "phadjido@gmail.com, dimetaxakis@gmail.com, argsofos.1@gmail.com",
    keywords = "parallel, data preprocessing, scikit-learn, machine learning, big data",
    description = "A high-performance, parallel computing library designed to accelerate preprocessing tasks in ML pipelines.",
    long_description = read_description(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/CEID-HPCLAB/Mneme/tree/main",
    packages = find_packages(where = "src"),
    package_dir = {"": "src"},
    install_requires = read_requirements(),
)