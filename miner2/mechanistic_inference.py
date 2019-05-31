import datetime,pandas,numpy,os,pickle,sys
import sklearn,sklearn.decomposition
import scipy,scipy.stats
from pkg_resources import Requirement, resource_filename
import miner2.coexpression

### to remove
def splitForMultiprocessing(vector,cores):
    
    partition = int(len(vector)/cores)
    remainder = len(vector) - cores*partition
    starts = np.arange(0,len(vector),partition)[0:cores]
    for i in range(remainder):
        starts[cores-remainder+i] = starts[cores-remainder+i] + i    

    stops = starts+partition
    for i in range(remainder):
        stops[cores-remainder+i] = stops[cores-remainder+i] + 1
        
    return zip(starts,stops)

def splitForMultiprocessing(vector,cores):
    
    partition = int(len(vector)/cores)
    remainder = len(vector) - cores*partition
    starts = numpy.arange(0,len(vector),partition)[0:cores]
    for i in range(remainder):
        starts[cores-remainder+i] = starts[cores-remainder+i] + i    

    stops = starts+partition
    for i in range(remainder):
        stops[cores-remainder+i] = stops[cores-remainder+i] + 1
        
    return zip(starts,stops)

def multiprocess(function,tasks):
    import multiprocessing, multiprocessing.pool
    hydra=multiprocessing.pool.Pool(len(tasks))  
    output=hydra.map(function,tasks)   
    hydra.close()
    hydra.join()
    return output

def condenseOutput(output):
    
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = resultsDict.keys()
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results
### to remove

def axisTfs(axesDf,tfList,expressionData,correlationThreshold=0.3):
    
    axesArray = numpy.array(axesDf.T)
    if correlationThreshold > 0:
        tfArray=numpy.array(expressionData.reindex(tfList)) # ALO Py3
    axes = numpy.array(axesDf.columns)
    tfDict = {}
    
    if type(tfList) is list:
        tfs = numpy.array(tfList)
    elif type(tfList) is not list:
        tfs = tfList
        
    if correlationThreshold == 0:
        for axis in range(axesArray.shape[0]):
            tfDict[axes[axis]] = tfs

        return tfDict
    
    for axis in range(axesArray.shape[0]):
        tfCorrelation = miner2.coexpression.pearson_array(tfArray,axesArray[axis,:])
        ### ALO, fixed warning over nan evaluations
        condition1=numpy.greater_equal(numpy.abs(tfCorrelation),correlationThreshold,where=numpy.isnan(tfCorrelation) == False)
        condition2=numpy.isnan(tfCorrelation)
        tfDict[axes[axis]]=tfs[numpy.where(numpy.bitwise_and(condition1 == True, condition2 == False))[0]]
        ### end ALO
    
    return tfDict

def enrichment(axes,revisedClusters,expressionData,correlationThreshold=0.3,numCores=1,p=0.05,database="tfbsdb_tf_to_genes.pkl"):
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t mechanistic inference"))
    
    tfToGenesPath = resource_filename(Requirement.parse("miner2"), 'miner2/data/{}'.format(database))
    with open(tfToGenesPath, 'rb') as f:
        tfToGenes = pickle.load(f)

    if correlationThreshold <= 0:
        allGenes = [int(len(expressionData.index))]
    elif correlationThreshold > 0:
        allGenes = list(expressionData.index)
        
    tfs = list(tfToGenes.keys())
    tfMap = axisTfs(axes,tfs,expressionData,correlationThreshold=correlationThreshold)

    tasks=[[clusterKey,(allGenes,revisedClusters,tfMap,tfToGenes,p)] for clusterKey in list(revisedClusters.keys())]
    hydra=multiprocessing.pool.Pool(numCores)
    results=hydra.map(tfbsdbEnrichment,tasks)

    mechanisticOutput={}
    for result in results:
        for key in result.keys():
            if key not in mechanisticOutput:
                mechanisticOutput[key]=result[key]
            else:
                print('key twice')
                sys.exit()
    print('completed')

    sys.exit()

        
    return mechanisticOutput

def hyper(population,set1,set2,overlap):
    
    b = max(set1,set2)
    c = min(set1,set2)
    hyp = scipy.stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])
    
    return prb 

def principalDf(dict_,expressionData,regulons=None,subkey='genes',minNumberGenes=8,random_state=12):

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t preparing mechanistic inference"))

    pcDfs = []
    setIndex = set(expressionData.index)
    
    if regulons is not None:
        dict_, df = regulonDictionary(regulons)
    for i in dict_.keys():
        if subkey is not None:
            genes = list(set(dict_[i][subkey])&setIndex)
            if len(genes) < minNumberGenes:
                continue
        elif subkey is None:
            genes = list(set(dict_[i])&setIndex)
            if len(genes) < minNumberGenes:
                continue
            
        pca = sklearn.decomposition.PCA(1,random_state=random_state)
        principalComponents = pca.fit_transform(expressionData.loc[genes,:].T)
        principalDf = pandas.DataFrame(principalComponents)
        principalDf.index = expressionData.columns
        principalDf.columns = [str(i)]
        
        normPC = numpy.linalg.norm(numpy.array(principalDf.iloc[:,0]))
        pearson = scipy.stats.pearsonr(principalDf.iloc[:,0],numpy.median(expressionData.loc[genes,:],axis=0))
        signCorrection = pearson[0]/numpy.abs(pearson[0])
        
        principalDf = signCorrection*principalDf/normPC
        
        pcDfs.append(principalDf)
    
    principalMatrix = pandas.concat(pcDfs,axis=1)
        
    return principalMatrix

def tfbsdbEnrichment(task):
    
    start, stop = task[0]
    allGenes,revisedClusters,tfMap,tfToGenes,p = task[1]
    keys = revisedClusters.keys()[start:stop]

    if len(allGenes) == 1:
    
        population_size = int(allGenes[0])
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:    
                hits0TfTargets = tfToGenes[tf]  
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in clusterTfs.keys():
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

                            
    elif len(allGenes) > 1:
        
        population_size = len(allGenes)
        clusterTfs = {}
        for key in keys:
            for tf in tfMap[str(key)]:    
                hits0TfTargets = list(set(tfToGenes[tf])&set(allGenes))   
                hits0clusterGenes = revisedClusters[key]
                overlapCluster = list(set(hits0TfTargets)&set(hits0clusterGenes))
                if len(overlapCluster) <= 1:
                    continue                
                pHyper = hyper(population_size,len(hits0TfTargets),len(hits0clusterGenes),len(overlapCluster))
                if pHyper < p:
                    if key not in clusterTfs.keys():
                        clusterTfs[key] = {}
                    clusterTfs[key][tf] = [pHyper,overlapCluster]

    return clusterTfs
