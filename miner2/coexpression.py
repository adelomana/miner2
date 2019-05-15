import datetime,numpy,pandas,time,sys
import sklearn,sklearn.decomposition
import multiprocessing, multiprocessing.pool

def cluster(expressionData,minNumberGenes = 6,minNumberOverExpSamples=4,maxSamplesExcluded=0.50,random_state=12,overExpressionThreshold=80,numCores=1):

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t coexpression"))
    print(numCores)

    df = expressionData.copy()
    
    maxStep = int(numpy.round(10*maxSamplesExcluded))
    allGenesMapped = []
    bestHits = []

    zero = numpy.percentile(expressionData,0)
    expressionThreshold = numpy.mean([numpy.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],overExpressionThreshold) for i in range(expressionData.shape[1])])

    startTimer = time.time()
    trial = -1

    for step in range(maxStep):
        trial+=1
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t working on coexpression step {} out of {}".format(step,maxStep)))
        
        genesMapped = []
        bestMapped = []

        pca = sklearn.decomposition.PCA(10,random_state=random_state)
        principalComponents = pca.fit_transform(df.T)
        principalDf = pandas.DataFrame(principalComponents)
        principalDf.index = df.columns

        #hydra=multiprocessing.pool.Pool(numCores)
        #genesMappedParallel=hydra.map(geneMapper,numpy.arange(10))
        genesMappedParallel=[]
        for i in range(10):
            task=[df,principalDf,i]
            instance=geneMapper(task)
            genesMappedParallel.append(instance)
        for element in genesMappedParallel:
            for gene in element:
                genesMapped.append(gene)

        print(len(genesMapped))
        print('completed')
        sys.exit()

        allGenesMapped.extend(genesMapped)
        try:
            stackGenes = numpy.hstack(genesMapped)
        except:
            stackGenes = []
        residualGenes = list(set(df.index)-set(stackGenes))
        df = df.loc[residualGenes,:]

        # computationally fast surrogate for passing the overexpressed significance test:
        for ix in range(len(genesMapped)):
            
            tmpCluster = expressionData.loc[genesMapped[ix],:]
            tmpCluster[tmpCluster<expressionThreshold] = 0
            tmpCluster[tmpCluster>0] = 1
            sumCluster = numpy.array(np.sum(tmpCluster,axis=0))
            numHits = numpy.where(sumCluster>0.333*len(genesMapped[ix]))[0]
            bestMapped.append(numHits)
            if len(numHits)>minNumberOverExpSamples:
                bestHits.append(genesMapped[ix])

        if len(bestMapped)>0:            
            countHits = Counter(np.hstack(bestMapped))
            ranked = countHits.most_common()
            dominant = [i[0] for i in ranked[0:int(numpy.ceil(0.1*len(ranked)))]]
            remainder = [i for i in numpy.arange(df.shape[1]) if i not in dominant]
            df = df.iloc[:,remainder]

    bestHits.sort(key=lambda s: -len(s))

    stopTimer = time.time()
    print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))
    
    return bestHits

def combineClusters(axes,clusters,threshold=0.925):
    combineAxes = {}
    filterKeys = numpy.array(list(axes.keys())) # ALO: changed to list because of Py3
    axesMatrix = numpy.vstack([axes[i] for i in filterKeys])
    for key in filterKeys:
        axis = axes[key]
        pearson = pearson_array(axesMatrix,axis)
        combine = numpy.where(pearson>threshold)[0]
        combineAxes[key] = filterKeys[combine]
     
    revisedClusters = {}
    combinedKeys = decomposeDictionaryToLists(combineAxes)
    for keyList in combinedKeys:
        genes = list(set(numpy.hstack([clusters[i] for i in keyList])))
        revisedClusters[len(revisedClusters)] = genes    

    return revisedClusters

def geneMapper(task):

    genesMapped=[]
    df=task[0]
    principalDf=task[1]
    i=task[2]

    pearson = pearson_array(numpy.array(df),numpy.array(principalDf[i]))
    highpass = max(numpy.percentile(pearson,95),0.1)
    lowpass = min(numpy.percentile(pearson,5),-0.1)
    cluster1 = numpy.array(df.index[numpy.where(pearson>highpass)[0]])
    cluster2 = numpy.array(df.index[numpy.where(pearson<lowpass)[0]])
    
    for clst in [cluster1,cluster2]:
        pdc = recursiveAlignment(clst,expressionData=df,minNumberGenes=minNumberGenes)
        if len(pdc)==0:
            continue
        elif len(pdc) == 1:
            genesMapped.append(pdc[0])
        elif len(pdc) > 1:
            for j in range(len(pdc)-1):
                if len(pdc[j]) > minNumberGenes:
                    genesMapped.append(pdc[j])

    return genesMapped

def decompose(geneset,expressionData,minNumberGenes=6): 
    fm = FrequencyMatrix(expressionData.loc[geneset,:])
    tst = numpy.multiply(fm,fm.T)
    tst[tst<numpy.percentile(tst,80)]=0
    tst[tst>0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered

def decomposeDictionaryToLists(dict_):
    decomposedSets = []
    for key in dict_.keys():
        newSet = iterativeCombination(dict_,key,iterations=25)
        if newSet not in decomposedSets:
            decomposedSets.append(newSet)
    return decomposedSets

def FrequencyMatrix(matrix,overExpThreshold = 1):
    
    numRows = matrix.shape[0]
    
    if type(matrix) == pandas.core.frame.DataFrame:
        index = matrix.index
        matrix = numpy.array(matrix)
    else:
        index = numpy.arange(numRows)
    
    matrix[matrix<overExpThreshold] = 0
    matrix[matrix>0] = 1
            
    hitsMatrix = pandas.DataFrame(numpy.zeros((numRows,numRows)))
    for column in range(matrix.shape[1]):
        geneset = matrix[:,column]
        hits = numpy.where(geneset>0)[0]
        hitsMatrix.iloc[hits,hits] += 1
        
    frequencyMatrix = numpy.array(hitsMatrix)
    traceFM = numpy.array([frequencyMatrix[i,i] for i in range(frequencyMatrix.shape[0])]).astype(float)
    if numpy.count_nonzero(traceFM)<len(traceFM):
        #subset nonzero. computefm. normFM zeros with input shape[0]. overwrite by slice np.where trace>0
        nonzeroGenes = numpy.where(traceFM>0)[0]
        normFMnonzero = numpy.transpose(numpy.transpose(frequencyMatrix[nonzeroGenes,:][:,nonzeroGenes])/traceFM[nonzeroGenes])
        normDf = pandas.DataFrame(normFMnonzero)
        normDf.index = index[nonzeroGenes]
        normDf.columns = index[nonzeroGenes]          
    else:            
        normFM = numpy.transpose(numpy.transpose(frequencyMatrix)/traceFM)
        normDf = pandas.DataFrame(normFM)
        normDf.index = index
        normDf.columns = index   
    
    return normDf

def getAxes(clusters,expressionData):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = sklearn.decomposition.PCA(1)
        principalComponents = fpc.fit_transform(expressionData.loc[genes,:].T)
        axes[key] = principalComponents.ravel()
    return axes

def iterativeCombination(dict_,key,iterations=25):    
    initial = dict_[key]
    initialLength = len(initial)
    for iteration in range(iterations):
        revised = [i for i in initial]
        for element in initial:
            revised = list(set(revised)|set(dict_[element]))
        revisedLength = len(revised)
        if revisedLength == initialLength:
            return revised
        elif revisedLength > initialLength:
            initial = [i for i in revised]
            initialLength = len(initial)
    return revised

def pearson_array(array,vector):    
    #r = (1/n-1)sum(((x-xbar)/sx)((y-ybar)/sy))

    '''
    Description of function
    '''
    
    ybar = numpy.mean(vector)
    sy = numpy.std(vector,ddof=1)    
    yterms = (vector-ybar)/float(sy)
    
    array_sx = numpy.std(array,axis=1,ddof=1)
    
    if 0 in array_sx:
        passIndex = numpy.where(array_sx>0)[0]
        array = array[passIndex,:]
        array_sx = array_sx[passIndex]
        
    array_xbar = numpy.mean(array,axis=1)            
    product_array = numpy.zeros(array.shape)
    
    for i in range(0,product_array.shape[1]):
        product_array[:,i] = yterms[i]*(array[:,i] - array_xbar)/array_sx
        
    return numpy.sum(product_array,axis=1)/float(product_array.shape[1]-1)

def reconstruction(decomposedList,expressionData,threshold=0.925):
    clusters = {i:decomposedList[i] for i in range(len(decomposedList))}
    axes = getAxes(clusters,expressionData)
    recombine = combineClusters(axes,clusters,threshold)
    return recombine

def recursiveAlignment(geneset,expressionData,minNumberGenes=6):
    recDecomp = recursiveDecomposition(geneset,expressionData,minNumberGenes)
    if len(recDecomp) == 0:
        return []
    reconstructed = reconstruction(recDecomp,expressionData)
    reconstructedList = [reconstructed[i] for i in list(reconstructed.keys()) if len(reconstructed[i])>minNumberGenes] # ALO: changed to list becasue of Py3
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def recursiveDecomposition(geneset,expressionData,minNumberGenes=6):
    unmixedFiltered = decompose(geneset,expressionData,minNumberGenes=minNumberGenes)   
    if len(unmixedFiltered) == 0:
        return []
    shortSets = [i for i in unmixedFiltered if len(i)<50]
    longSets = [i for i in unmixedFiltered if len(i)>=50]    
    if len(longSets)==0:
        return unmixedFiltered
    for ls in longSets:
        unmixedFiltered = decompose(ls,expressionData,minNumberGenes=minNumberGenes)
        if len(unmixedFiltered)==0:
            continue
        shortSets.extend(unmixedFiltered)
    return shortSets

def unmix(df,iterations=25,returnAll=False):    
    frequencyClusters = []
    
    for iteration in range(iterations):
        sumDf1 = df.sum(axis=1)

        maxSum=sumDf1.idxmax() # ALO: changed to idxmax

        hits = numpy.where(df.loc[maxSum]>0)[0]
        hitIndex = list(df.index[hits])
        block = df.loc[hitIndex,hitIndex]
        blockSum = block.sum(axis=1)
        coreBlock = list(blockSum.index[numpy.where(blockSum>=numpy.median(blockSum))[0]])
        remainder = list(set(df.index)-set(coreBlock))
        frequencyClusters.append(coreBlock)
        if len(remainder)==0:
            return frequencyClusters
        if len(coreBlock)==1:
            return frequencyClusters
        df = df.loc[remainder,remainder]
    if returnAll is True:
        frequencyClusters.append(remainder)
    return frequencyClusters
