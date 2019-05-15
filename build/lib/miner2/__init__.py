name = "miner2"

import datetime,pkg_resources

__version__ = pkg_resources.get_distribution(name)

print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t hello from miner2 version {}".format(__version__)))
