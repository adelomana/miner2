import datetime,pkg_resources

name = "miner2"

#__version__ = pkg_resources.get_distribution(name)
try:
    __version__ = pkg_resources.get_distribution(name)
except:
    __version__ = 'development'

message=str(__version__).replace('miner2','miner2 version')
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t hello from {}".format(message)))
