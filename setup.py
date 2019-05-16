import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_DATA = {
    'miner2': ['data/*']
    }
    
setuptools.setup(
    name="miner2",
    version="0.0.7",
    author="Adrian Lopez Garcia de Lomana",
    author_email="alomana@systemsbiology.org",
    description="A newer version of MINER.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adelomana/miner2",
    packages=['miner2'],
    install_requires = setuptools.find_packages(),
    include_package_data=True, package_data=PACKAGE_DATA,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
