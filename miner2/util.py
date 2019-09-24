import numpy
import multiprocessing, multiprocessing.pool


def split_for_multiprocessing(vector, cores):
    partition = int(len(vector) / cores)
    remainder = len(vector) - cores * partition
    starts = numpy.arange(0, len(vector) ,partition)[0:cores]

    for i in range(remainder):
        starts[cores - remainder + i] = starts[cores - remainder + i] + i

    stops = starts + partition
    for i in range(remainder):
        stops[cores - remainder + i] = stops[cores - remainder + i] + 1

    return zip(starts, stops)


def multiprocess(function, tasks):
    hydra = multiprocessing.pool.Pool(len(tasks))
    output = hydra.map(function, tasks)
    hydra.close()
    hydra.join()
    return output


def condense_output(output):
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = resultsDict.keys()
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results
