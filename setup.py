import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

with open("requirements.txt", 'r') as f:
    install_requires = [line.strip() for line in f.readlines()]

setuptools.setup(
    name="qualg",
    version="0.1.0",
    author="Axel Dahlberg",
    author_email="axel.dahlberg12@gmail.com",
    description="Symbolic linear algrebra for quantum mechanics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AckslD/QuAlg",
    include_package_data=True,
    packages=setuptools.find_packages(exclude=('tests', 'docs', 'examples')),
    install_requires=install_requires,
    python_requires='>=3.6',
)
