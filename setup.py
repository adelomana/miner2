import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGE_DATA = {
    'miner2': ['data/*']
    }

setuptools.setup(
    name="miner2",
    version="0.0.9",
    author="Adrian Lopez Garcia de Lomana",
    author_email="alomana@systemsbiology.org",
    description="A newer version of MINER.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baliga-lab/miner2",
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
    scripts=['bin/miner2-coexpr', 'bin/miner2-mechinf',
             'bin/miner2-bcmembers', 'bin/miner2-subtypes',
             'bin/miner2-survival', 'bin/miner2-causalinf-pre',
             'bin/miner2-causalinf-post', 'bin/miner2-neo', 'bin/miner2-riskclassifier'])
