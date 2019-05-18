import datetime,pandas,numpy,scipy,os,pickle
import sklearn,sklearn.decomposition,sys
from pkg_resources import Requirement, resource_filename
import miner2.coexpression

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

    print('len(axesArray.shape[0])',axesArray.shape[0])
    
    for axis in range(axesArray.shape[0]):
        
        
        tfCorrelation = miner2.coexpression.pearson_array(tfArray,axesArray[axis,:])
        #print('axis,correlationThreshold',axis,correlationThreshold,tfCorrelation.shape)
        a=numpy.abs(tfCorrelation)
        b=correlationThreshold
        print('a,b',type(a),a.shape,b)

        c=numpy.where(numpy.abs(tfCorrelation) >= correlationThreshold)[0]
        print('c',c,c.shape)

        alternative=numpy.where(numpy.isnan(tfCorrelation))
        alternative2=numpy.where(alternative[0] == False and numpy.abs(tfCorrelation) >= correlationThreshold)
        print('alternative2',alternative2)
        sys.exit()
        
        tfDict[axes[axis]] = tfs[numpy.where(numpy.abs(tfCorrelation) >= correlationThreshold)[0]]
    print('completed')
    
    return tfDict

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

    taskSplit = splitForMultiprocessing(revisedClusters.keys(),numCores)
    print('taskSplit',taskSplit)
    sys.exit()
    
    tasks = [[taskSplit[i],(allGenes,revisedClusters,tfMap,tfToGenes,p)] for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdbEnrichment,tasks)
    mechanisticOutput = condenseOutput(tfbsdbOutput)
        
    return mechanisticOutput

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
