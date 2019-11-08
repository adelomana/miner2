import numpy, scipy, pandas, sklearn, lifelines, matplotlib, seaborn

import multiprocessing, multiprocessing.pool
import sys


def write_dependency_infos(outfile):
    py_version = map(str, list(sys.version_info[0:3]))
    outfile.write('Python %s\n\n' % ('.'.join(py_version)))
    outfile.write('Dependencies:\n')
    outfile.write('-------------\n\n')
    outfile.write('numpy: %s\n' % numpy.__version__)
    outfile.write('scipy: %s\n' % scipy.__version__)
    outfile.write('pandas: %s\n' % scipy.__version__)
    outfile.write('sklearn: %s\n' % sklearn.__version__)
    outfile.write('lifelines: %s\n' % lifelines.__version__)
    outfile.write('matplotlib: %s\n' % matplotlib.__version__)
    outfile.write('seaborn: %s\n' % seaborn.__version__)


def split_for_multiprocessing(vector, cores):
    partition = int(len(vector) / cores)
    remainder = len(vector) - cores * partition
    starts = numpy.arange(0, len(vector) ,partition)[0:cores]

    for i in range(remainder):
        starts[cores - remainder + i] = starts[cores - remainder + i] + i

    stops = starts + partition
    for i in range(remainder):
        stops[cores - remainder + i] = stops[cores - remainder + i] + 1

    return list(zip(starts, stops))


def multiprocess(function, tasks):
    hydra = multiprocessing.pool.Pool(len(tasks))
    output = hydra.map(function, tasks)
    hydra.close()
    hydra.join()
    return output


def condense_output(output):
    """WW: what is this weird construct doing ???? It seems like it does things very complicated"""
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = list(resultsDict.keys())
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results
