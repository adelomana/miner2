#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:20:40 2019

@author: MattWall
"""
import numpy as np
import pandas as pd
from scipy import stats
import os
from sklearn.decomposition import PCA
#import multiprocessing, multiprocessing.pool
import matplotlib.pyplot as plt
import time
from collections import Counter
#import seaborn as sns

# =============================================================================
# Functions used for reading and writing files
# =============================================================================

def read_pkl(input_file):
    import pickle
    with open(input_file, 'rb') as f:
        dict_ = pickle.load(f)        
    return dict_

def write_pkl(dictionary,output_file):
    import pickle
    output = open(output_file, 'wb')
    pickle.dump(dictionary, output)
    output.close()
    
    return

def read_json(filename):
    import json    
    with open(filename) as data:
        dict_ = json.load(data) 
    return dict_

def write_json(dict_, output_file):
    import json
    output_file = output_file
    with open(output_file, 'w') as fp:
        json.dump(dict_, fp)
    return

def readFileToDf(filename):
    extension = filename.split(".")[-1]
    if extension == "csv":
        df = pd.read_csv(filename,index_col=0,header=0)
        shape = df.shape
        if shape[1] == 0:
            df = pd.read_csv(filename,index_col=0,header=0,sep="\t")
    elif extension == "txt":
        df = pd.read_csv(filename,index_col=0,header=0,sep="\t")
        shape = df.shape
        if shape[1] == 0:
            df = pd.read_csv(filename,index_col=0,header=0)    
    return df

def fileToReferenceDictionary(filename,dictionaryName,index_col=0):
    read_reference_db = pd.read_csv(filename,index_col=0,header=0)
    
    if list(read_reference_db.iloc[:,0]) == range(len(read_reference_db.iloc[:,0])):
        read_reference_db = read_reference_db.iloc[:,1:]
        print("deleted")
    read_reference_db.index = read_reference_db.iloc[:,0]
    read_reference_db = read_reference_db.iloc[:,1:]
    read_reference_db.head()
    
    reference_dic = {}
    for key in list(set(read_reference_db.index)):
        tmp_df = read_reference_db.loc[key,:]
        if type(tmp_df) is not pd.core.frame.DataFrame:
            tmp_df = pd.DataFrame(tmp_df)
        reference_dic[key] = list(tmp_df.iloc[:,0])
    
    write_pkl(reference_dic,dictionaryName)

    return reference_dic

# =============================================================================
# Functions used for pre-processing data
# =============================================================================

def removeNullRows(df):
    
    minimum = np.percentile(df,0)
    if minimum == 0:
        filteredDf = df.loc[df.sum(axis=1)>0,:]
    else:
        filteredDf = df
        
    return filteredDf

def identifierConversion(expressionData,conversionTable=os.path.join("..","data","identifier_mappings.txt")):    
    idMap = pd.read_table(conversionTable)
    genetypes = list(set(idMap.iloc[:,2]))
    previousIndex = np.array(expressionData.index).astype(str)    
    previousColumns = np.array(expressionData.columns).astype(str)  
    bestMatch = []
    for geneType in genetypes:
        subset = idMap[idMap.iloc[:,2]==geneType]
        subset.index = subset.iloc[:,1]
        mappedGenes = list(set(previousIndex)&set(subset.index))
        mappedSamples = list(set(previousColumns)&set(subset.index))
        if len(mappedGenes)>=max(10,0.01*expressionData.shape[0]):
            if len(mappedGenes)>len(bestMatch):
                bestMatch = mappedGenes
                state = "original"
                gtype = geneType
                continue
        if len(mappedSamples)>=max(10,0.01*expressionData.shape[1]):
            if len(mappedSamples)>len(bestMatch):
                bestMatch = mappedSamples
                state = "transpose"
                gtype = geneType
                continue
    
    mappedGenes = bestMatch
    subset = idMap[idMap.iloc[:,2]==gtype] 
    subset.index = subset.iloc[:,1]

    if len(bestMatch) == 0:
        print("Error: Gene identifiers not recognized")
    
    if state == "transpose":
        expressionData = expressionData.T
        
    try:
        convertedData = expressionData.loc[mappedGenes,:]
    except:
        convertedData = expressionData.loc[np.array(mappedGenes).astype(int),:]
    
    conversionTable = subset.loc[mappedGenes,:]
    conversionTable.index = conversionTable.iloc[:,0]
    conversionTable = conversionTable.iloc[:,1]
    conversionTable.columns = ["Name"]
    
    newIndex = list(subset.loc[mappedGenes,"Preferred_Name"])
    convertedData.index = newIndex
    
    duplicates = [item for item, count in Counter(newIndex).items() if count > 1]
    singles = list(set(convertedData.index)-set(duplicates))
    
    corrections = []
    for duplicate in duplicates:
        dupData = convertedData.loc[duplicate,:]
        firstChoice = pd.DataFrame(dupData.iloc[0,:]).T
        corrections.append(firstChoice)
    
    if len(corrections)  == 0:
        print("completed identifier conversion.\n"+str(convertedData.shape[0])+" genes were converted." )
        return convertedData, conversionTable
        
    correctionsDf = pd.concat(corrections,axis=0)
    uncorrectedData = convertedData.loc[singles,:]
    convertedData = pd.concat([uncorrectedData,correctionsDf],axis=0)    
    
    print("completed identifier conversion.\n"+str(convertedData.shape[0])+" genes were converted." )
    
    return convertedData, conversionTable

def readExpressionFromGZipFiles(directory):

    rootDir = directory
    sample_dfs = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            #print('\t%s' % fname)
            extension = fname.split(".")[-1]
            if extension == 'gz':
                path = os.path.join(rootDir,dirName,fname)
                df = pd.read_csv(path, compression='gzip', index_col=0,header=None, sep='\t', quotechar='"')
                df.columns = [fname.split(".")[0]]
                sample_dfs.append(df)

    expressionData = pd.concat(sample_dfs,axis=1)
    return expressionData

def readCausalFiles(directory):

    rootDir = directory
    sample_dfs = []
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            #print('\t%s' % fname)
            extension = fname.split(".")[-1]
            if extension == 'csv':
                path = os.path.join(rootDir,dirName,fname)
                df = pd.read_csv(path, index_col=0,header=0)
                df.index = np.array(df.index).astype(str)
                sample_dfs.append(df)

    causalData = pd.concat(sample_dfs,axis=0)
    renamed = [("-").join(["R",str(name)]) for name in causalData.Regulon]
    causalData.Regulon = renamed
    return causalData

def entropy(vector):
    
    data = np.array(vector)
    hist = np.histogram(data,bins=50)[0]
    length = len(hist)

    if length <= 1:
        return 0

    counts = np.bincount(hist)
    probs = [float(i)/length for i in counts]
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        if i >0:
            ent -= float(i)*np.log(i)
    return ent

def quantile_norm(df,axis=1):
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata

    if axis == 1:    
        array = np.array(df)
        
        ranked_array = np.zeros(array.shape)    
        for i in range(0,array.shape[0]):
            ranked_array[i,:] = rankdata(array[i,:],method='min') - 1
            
        sorted_array = np.zeros(array.shape)    
        for i in range(0,array.shape[0]):
            sorted_array[i,:] = np.sort(array[i,:])
            
        qn_values = np.nanmedian(sorted_array,axis=0)
        
        quant_norm_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]
        
        quant_norm = pd.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)
    
    if axis == 0:
        array = np.array(df)
        
        ranked_array = np.zeros(array.shape)    
        for i in range(0,array.shape[1]):
            ranked_array[:,i] = rankdata(array[:,i],method='min') - 1
            
        sorted_array = np.zeros(array.shape)    
        for i in range(0,array.shape[1]):
            sorted_array[:,i] = np.sort(array[:,i])
            
        qn_values = np.nanmedian(sorted_array,axis=1)
        
        quant_norm_array = np.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]
        
        quant_norm = pd.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)
                
    return quant_norm

def transformFPKM(expressionData,fpkm_threshold=1,minFractionAboveThreshold=0.5,highlyExpressed=False,quantile_normalize=False):

    median = np.median(np.median(expressionData,axis=1))
    expDataCopy = expressionData.copy()
    expDataCopy[expDataCopy<fpkm_threshold]=0
    expDataCopy[expDataCopy>0]=1
    cnz = np.count_nonzero(expDataCopy,axis=1)
    keepers = np.where(cnz>=int(minFractionAboveThreshold*expDataCopy.shape[1]))[0]
    threshold_genes = expressionData.index[keepers]
    expDataFiltered = expressionData.loc[threshold_genes,:]
    
    if highlyExpressed is True:
        median = np.median(np.median(expDataFiltered,axis=1))
        expDataCopy = expDataFiltered.copy()
        expDataCopy[expDataCopy<median]=0
        expDataCopy[expDataCopy>0]=1
        cnz = np.count_nonzero(expDataCopy,axis=1)
        keepers = np.where(cnz>=int(0.5*expDataCopy.shape[1]))[0]
        median_filtered_genes = expDataFiltered.index[keepers]    
        expDataFiltered = expressionData.loc[median_filtered_genes,:]
    
    if quantile_normalize is True:
        expDataFiltered = quantile_norm(expDataFiltered,axis=0)
        
    finalExpData = pd.DataFrame(np.log2(expDataFiltered+1))
    finalExpData.index = expDataFiltered.index
    finalExpData.columns = expDataFiltered.columns
    
    return finalExpData

def preProcessTPM(tpm):
    
    from scipy import stats
    cutoff = stats.norm.ppf(0.00001)
    
    tmp_array_raw = np.array(tpm)
    keep = []
    keepappend = keep.append
    for i in range(0,tmp_array_raw.shape[0]):
        if np.count_nonzero(tmp_array_raw[i,:]) >= round(float(tpm.shape[1])*0.5):
            keepappend(i)

    tpm_zero_filtered = tmp_array_raw[keep,:]
    tpm_array = np.array(tpm_zero_filtered)
    positive_medians = []

    for i in range(0,tpm_array.shape[1]):
        tmp1 = tpm_array[:,i][tpm_array[:,i]>0]
        positive_medians.append(np.median(tmp1))

    # 2^10 - 1 = 1023
    scale_factors = [float(1023)/positive_medians[i] for i in range(0,len(positive_medians))]

    tpm_scale = np.zeros(tpm_array.shape)
    for i in range(0,tpm_scale.shape[1]):
        tpm_scale[:,i] = tpm_array[:,i]*scale_factors[i]

    tpm_scale_log2 = np.zeros(tpm_scale.shape)
    for i in range(0,tpm_scale_log2.shape[1]):
        tpm_scale_log2[:,i] = np.log2(tpm_scale[:,i]+1)

    tpm_filtered_df = pd.DataFrame(tpm_scale_log2)
    tpm_filtered_df.columns = list(tpm.columns)
    tpm_filtered_df.index = list(np.array(tpm.index)[keep])

    qn_tpm_filtered = quantile_norm(tpm_filtered_df,axis=0)
    qn_tpm = quantile_norm(qn_tpm_filtered,axis=1)

    qn_tpm_array = np.array(qn_tpm)

    tpm_z = np.zeros(qn_tpm_array.shape)
    for i in range(0,tpm_z.shape[0]):
        tmp = qn_tpm_array[i,:][qn_tpm_array[i,:]>0]
        mean = np.mean(tmp)
        std = np.std(tmp)
        for j in range(0,tpm_z.shape[1]):
            tpm_z[i,j] = float(qn_tpm_array[i,j] - mean)/std
            if tpm_z[i,j] < -4:
                tpm_z[i,j] = cutoff

    tpm_entropy = []
    for i in range(0,tpm_z.shape[0]):
        tmp = entropy(tpm_z[i,:])
        tpm_entropy.append(tmp)

    tpmz_df = pd.DataFrame(tpm_z)
    tpmz_df.columns = list(tpm.columns)
    tpmz_df.index = list(np.array(tpm.index)[keep])


    ent = pd.DataFrame(tpm_entropy)
    ent.index = list(tpmz_df.index)
    ent.columns = ['entropy']

    tpm_ent_df = pd.concat([tpmz_df,ent],axis=1)

    tpm_entropy_sorted = tpm_ent_df.sort_values(by='entropy',ascending=False)

    tmp = tpm_entropy_sorted[tpm_entropy_sorted.loc[:,'entropy']>=0]
    tpm_select = tmp.iloc[:,0:-1]

    return tpm_select

def zscore(expressionData):
    zero = np.percentile(expressionData,0)
    meanCheck = np.mean(expressionData[expressionData>zero].mean(axis=1,skipna=True))
    if meanCheck<0.1:
        return expressionData
    means = expressionData.mean(axis=1,skipna=True)
    stds = expressionData.std(axis=1,skipna=True)
    try:
        transform = ((expressionData.T - means)/stds).T
    except:
        passIndex = np.where(stds>0)[0]
        transform = ((expressionData.iloc[passIndex,:].T - means[passIndex])/stds[passIndex]).T
    print("completed z-transformation.")
    return transform

def correctBatchEffects(df): 
    
    zscoredExpression = zscore(df)
    means = []
    stds = []
    for i in range(zscoredExpression.shape[1]):
        mean = np.mean(zscoredExpression.iloc[:,i])
        std = np.std(zscoredExpression.iloc[:,i])
        means.append(mean)
        stds.append(std)

    if np.std(means) >= 0.15:
        zscoredExpression = preProcessTPM(df)
        
    return zscoredExpression

def preprocess(filename, conversionTablePath):
    rawExpression = readFileToDf(filename)
    rawExpressionZeroFiltered = removeNullRows(rawExpression)
    zscoredExpression = correctBatchEffects(rawExpressionZeroFiltered)
    expressionData, conversionTable = identifierConversion(zscoredExpression, conversionTablePath)
    return expressionData, conversionTable

# =============================================================================
# Functions used for clustering 
# =============================================================================

def pearson_array(array,vector):    
    #r = (1/n-1)sum(((x-xbar)/sx)((y-ybar)/sy))
    
    ybar = np.mean(vector)
    sy = np.std(vector,ddof=1)    
    yterms = (vector-ybar)/float(sy)
    
    array_sx = np.std(array,axis=1,ddof=1)
    
    if 0 in array_sx:
        passIndex = np.where(array_sx>0)[0]
        array = array[passIndex,:]
        array_sx = array_sx[passIndex]
        
    array_xbar = np.mean(array,axis=1)            
    product_array = np.zeros(array.shape)
    
    for i in range(0,product_array.shape[1]):
        product_array[:,i] = yterms[i]*(array[:,i] - array_xbar)/array_sx
        
    return np.sum(product_array,axis=1)/float(product_array.shape[1]-1)

def getAxes(clusters,expressionData):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = PCA(1)
        principalComponents = fpc.fit_transform(expressionData.loc[genes,:].T)
        axes[key] = principalComponents.ravel()
    return axes

def FrequencyMatrix(matrix,overExpThreshold = 1):
    
    numRows = matrix.shape[0]
    
    if type(matrix) == pd.core.frame.DataFrame:
        index = matrix.index
        matrix = np.array(matrix)
    else:
        index = np.arange(numRows)
    
    matrix[matrix<overExpThreshold] = 0
    matrix[matrix>0] = 1
            
    hitsMatrix = pd.DataFrame(np.zeros((numRows,numRows)))
    for column in range(matrix.shape[1]):
        geneset = matrix[:,column]
        hits = np.where(geneset>0)[0]
        hitsMatrix.iloc[hits,hits] += 1
        
    frequencyMatrix = np.array(hitsMatrix)
    traceFM = np.array([frequencyMatrix[i,i] for i in range(frequencyMatrix.shape[0])]).astype(float)
    if np.count_nonzero(traceFM)<len(traceFM):
        #subset nonzero. computefm. normFM zeros with input shape[0]. overwrite by slice np.where trace>0
        nonzeroGenes = np.where(traceFM>0)[0]
        normFMnonzero = np.transpose(np.transpose(frequencyMatrix[nonzeroGenes,:][:,nonzeroGenes])/traceFM[nonzeroGenes])
        normDf = pd.DataFrame(normFMnonzero)
        normDf.index = index[nonzeroGenes]
        normDf.columns = index[nonzeroGenes]          
    else:            
        normFM = np.transpose(np.transpose(frequencyMatrix)/traceFM)
        normDf = pd.DataFrame(normFM)
        normDf.index = index
        normDf.columns = index   
    
    return normDf

def f1Binary(similarityMatrix):

    remainingMembers = set(similarityMatrix.index)
    # probeSample is the sample that serves as a seed to identify a cluster in a given iteration
    probeSample = np.argmax(similarityMatrix.sum(axis=1))
    # members are the samples that satisfy the similarity condition with the previous cluster or probeSample
    members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
    # nonMembers are the remaining members not in the current cluster
    nonMembers = remainingMembers-members
    # instantiate list to collect clusters of similar members
    similarityClusters = []
    # instantiate f1 score for optimization
    f1 = 0

    for iteration in range(1500):

        predictedMembers = members
        predictedNonMembers = remainingMembers-predictedMembers

        sumSlice = np.sum(similarityMatrix.loc[:,list(predictedMembers)],axis=1)/float(len(predictedMembers))
        members = set(similarityMatrix.index[np.where(sumSlice>0.8)[0]])

        if members==predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break    
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        nonMembers = remainingMembers-members
        TP = len(members&predictedMembers)
        FN = len(predictedNonMembers&members)
        FP = len(predictedMembers&nonMembers)
        tmpf1 = TP/float(TP+FN+FP)

        if tmpf1 <= f1:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue   

        elif tmpf1 > f1:
            f1 = tmpf1
            continue
    
    similarityClusters.sort(key = lambda s: -len(s))

    return similarityClusters

def unmix(df,iterations=25,returnAll=False):    
    frequencyClusters = []
    
    for iteration in range(iterations):
        sumDf1 = df.sum(axis=1)
        #maxSum = np.argmax(sumDf1)

        maxSum = sumDf1.idxmax()
        
        hits = np.where(df.loc[maxSum]>0)[0]
        hitIndex = list(df.index[hits])
        block = df.loc[hitIndex,hitIndex]
        blockSum = block.sum(axis=1)
        coreBlock = list(blockSum.index[np.where(blockSum>=np.median(blockSum))[0]])
        # WW: sorting for comparability
        remainder = sorted(list(set(df.index) - set(coreBlock)))
        frequencyClusters.append(coreBlock)
        if len(remainder)==0:
            return frequencyClusters
        if len(coreBlock)==1:
            return frequencyClusters
        df = df.loc[remainder,remainder]
    if returnAll is True:
        frequencyClusters.append(remainder)
    return frequencyClusters

def remix(df,frequencyClusters):    
    finalClusters = []
    for cluster in frequencyClusters:
        sliceDf = df.loc[cluster,:]
        sumSlice = sliceDf.sum(axis=0)
        cut = min(0.8,np.percentile(sumSlice.loc[cluster]/float(len(cluster)),90))
        minGenes = max(4,cut*len(cluster))
        keepers = list(sliceDf.columns[np.where(sumSlice>=minGenes)[0]])
        keepers = list(set(keepers)|set(cluster))
        finalClusters.append(keepers)
        finalClusters.sort(key = lambda s: -len(s))
    return finalClusters

def decompose(geneset,expressionData,minNumberGenes=6): 
    fm = FrequencyMatrix(expressionData.loc[geneset,:])
    tst = np.multiply(fm,fm.T)
    tst[tst<np.percentile(tst,80)]=0
    tst[tst>0]=1
    unmix_tst = unmix(tst)
    unmixedFiltered = [i for i in unmix_tst if len(i)>=minNumberGenes]
    return unmixedFiltered

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

def iterativeCombination(dict_,key,iterations=25):    
    initial = dict_[key]
    initialLength = len(initial)
    for iteration in range(iterations):
        revised = [i for i in initial]
        for element in initial:
            # WW: sorting for comparability
            revised = sorted(list(set(revised)|set(dict_[element])))
        revisedLength = len(revised)
        if revisedLength == initialLength:
            return revised
        elif revisedLength > initialLength:
            initial = [i for i in revised]
            initialLength = len(initial)
    return revised

def decomposeDictionaryToLists(dict_):
    decomposedSets = []
    # WW: sorting for comparability
    for key in sorted(dict_.keys()):
        newSet = iterativeCombination(dict_,key,iterations=25)
        if newSet not in decomposedSets:
            decomposedSets.append(newSet)
    return decomposedSets

def combineClusters(axes,clusters,threshold=0.925):
    combineAxes = {}
    filterKeys = np.array(list(axes.keys()))
    axesMatrix = np.vstack([axes[i] for i in filterKeys])
    for key in filterKeys:
        axis = axes[key]
        pearson = pearson_array(axesMatrix,axis)
        combine = np.where(pearson>threshold)[0]
        combineAxes[key] = filterKeys[combine]

    revisedClusters = {}
    combinedKeys = decomposeDictionaryToLists(combineAxes)
    for keyList in combinedKeys:
        genes = list(set(np.hstack([clusters[i] for i in keyList])))
        revisedClusters[len(revisedClusters)] = sorted(genes)

    return revisedClusters

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
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()
                         if len(reconstructed[i])>minNumberGenes]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def cluster(expressionData,minNumberGenes = 6,minNumberOverExpSamples=4,maxSamplesExcluded=0.50,random_state=12,overExpressionThreshold=80):

    try:
        df = expressionData.copy()
        maxStep = int(np.round(10*maxSamplesExcluded))
        allGenesMapped = []
        bestHits = []

        zero = np.percentile(expressionData,0)
        expressionThreshold = np.mean([np.percentile(expressionData.iloc[:,i][expressionData.iloc[:,i]>zero],overExpressionThreshold) for i in range(expressionData.shape[1])])

        startTimer = time.time()
        trial = -1
        for step in range(maxStep):
            trial+=1
            progress = (100./maxStep)*trial
            print('{:.2f} percent complete'.format(progress))
            genesMapped = []
            bestMapped = []

            pca = PCA(10,random_state=random_state)
            principalComponents = pca.fit_transform(df.T)
            principalDf = pd.DataFrame(principalComponents)
            principalDf.index = df.columns

            for i in range(10):
                pearson = pearson_array(np.array(df),np.array(principalDf[i]))
                if len(pearson) == 0:
                    continue
                highpass = max(np.percentile(pearson,95),0.1)
                lowpass = min(np.percentile(pearson,5),-0.1)
                cluster1 = np.array(df.index[np.where(pearson>highpass)[0]])
                cluster2 = np.array(df.index[np.where(pearson<lowpass)[0]])

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

            allGenesMapped.extend(genesMapped)
            try:
                stackGenes = np.hstack(genesMapped)
            except:
                stackGenes = []
            residualGenes = sorted(list(set(df.index)-set(stackGenes)))
            df = df.loc[residualGenes,:]

            # computationally fast surrogate for passing the overexpressed significance test:
            for ix in range(len(genesMapped)):
                tmpCluster = expressionData.loc[genesMapped[ix],:]
                tmpCluster[tmpCluster<expressionThreshold] = 0
                tmpCluster[tmpCluster>0] = 1
                sumCluster = np.array(np.sum(tmpCluster,axis=0))
                numHits = np.where(sumCluster>0.333*len(genesMapped[ix]))[0]
                bestMapped.append(numHits)
                if len(numHits)>minNumberOverExpSamples:
                    bestHits.append(genesMapped[ix])

            if len(bestMapped)>0:            
                countHits = Counter(np.hstack(bestMapped))
                ranked = countHits.most_common()
                dominant = [i[0] for i in ranked[0:int(np.ceil(0.1*len(ranked)))]]
                remainder = [i for i in np.arange(df.shape[1]) if i not in dominant]
                df = df.iloc[:,remainder]

        bestHits.sort(key=lambda s: -len(s))

        stopTimer = time.time()
        print('\ncoexpression clustering completed in {:.2f} minutes'.format((stopTimer-startTimer)/60.))

    except:
        print('\nClustering failed. Ensure that expression data is formatted with genes as rows and samples as columns.')
        print('Consider transposing data (expressionData = expressionData.T) and retrying')

    return bestHits

def backgroundDf(expressionData):

    low = np.percentile(expressionData,100./3,axis=0)
    high = np.percentile(expressionData,200./3,axis=0)
    evenCuts = zip(low,high)
    
    bkgd = expressionData.copy()
    for i in range(bkgd.shape[1]):
        lowCut = evenCuts[i][0]
        highCut = evenCuts[i][1]    
        bkgd.iloc[:,i][bkgd.iloc[:,i]>=highCut]=1
        bkgd.iloc[:,i][bkgd.iloc[:,i]<=lowCut]=-1
        bkgd.iloc[:,i][np.abs(bkgd.iloc[:,i])!=1]=0    

    return bkgd
            
def assignMembership(geneset,background,p=0.05):

    cluster = background.loc[geneset,:]
    classNeg1 = len(geneset)-np.count_nonzero(cluster+1,axis=0)
    class0 = len(geneset)-np.count_nonzero(cluster,axis=0)
    class1 = len(geneset)-np.count_nonzero(cluster-1,axis=0)
    observations = zip(classNeg1,class0,class1)
    
    highpass = stats.binom.ppf(1-p/3,len(geneset),1./3)
    classes = []
    for i in range(len(observations)):
        check = np.where(np.array(observations[i])>=highpass)[0]
        if len(check)>1:
            check = np.array([np.argmax(np.array(observations[i]))])
        classes.append(check)
    
    return classes

def clusterScore(membership,pMembership=0.05):
    hits = len([i for i in membership if len(i)>0])
    N = len(membership)
    return 1-stats.binom.cdf(hits,N,pMembership)

def getClusterScores(regulonModules,background,p=0.05):
    clusterScores = {}
    for key in regulonModules.keys():    
        members = assignMembership(regulonModules[key],background,p)    
        score = clusterScore(members)
        clusterScores[key] = score
    return clusterScores
    
def filterCoexpressionDict(coexpressionDict,clusterScores,threshold=0.01):   
    filterPoorClusters = np.where(clusterScores>threshold)[0]
    for x in filterPoorClusters:
        del coexpressionDict[x]
    keys = coexpressionDict.keys()
    filteredDict = {str(i):coexpressionDict[keys[i]] for i in range(len(coexpressionDict))}
    return filteredDict

def biclusterMembershipDictionary(revisedClusters,background,label=2,p=0.05):

    background_genes = set(background.index)
    if label == "excluded":
        members = {}
        for key in revisedClusters.keys():
            tmp_genes = list(set(revisedClusters[key])&background_genes)
            if len(tmp_genes)>1:
                assignments = assignMembership(tmp_genes,background,p=p)
            else:
                assignments = [np.array([]) for i in range(background.shape[1])]
            nonMembers = np.array([i for i in range(len(assignments)) if len(assignments[i])==0])
            if len(nonMembers) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[nonMembers])
        print("done!")
        return members
    
    if label == "included":
        members = {}
        for key in revisedClusters.keys():
            tmp_genes = list(set(revisedClusters[key])&background_genes)
            if len(tmp_genes)>1:
                assignments = assignMembership(tmp_genes,background,p=p)
            else:
                assignments = [np.array([]) for i in range(background.shape[1])]
            included = np.array([i for i in range(len(assignments)) if len(assignments[i])!=0])
            if len(included) == 0:
                members[key] = []
                continue
            members[key] = list(background.columns[included])
        print("done!")
        return members
    
    members = {}
    for key in revisedClusters.keys():
        tmp_genes = list(set(revisedClusters[key])&background_genes)
        if len(tmp_genes)>1:
            assignments = assignMembership(tmp_genes,background,p=p)
        else:
            assignments = [np.array([]) for i in range(background.shape[1])]
        overExpMembers = np.array([i for i in range(len(assignments)) if label in assignments[i]])
        if len(overExpMembers) ==0:
            members[key] = []
            continue
        members[key] = list(background.columns[overExpMembers])
    print("done!")
    return members

def membershipToIncidence(membershipDictionary,expressionData):
    
    incidence = np.zeros((len(membershipDictionary),expressionData.shape[1]))
    incidence = pd.DataFrame(incidence)
    incidence.index = membershipDictionary.keys()
    incidence.columns = expressionData.columns
    for key in membershipDictionary.keys():
        samples = membershipDictionary[key]
        incidence.loc[key,samples] = 1
    
    try:
        orderIndex = np.array(incidence.index).astype(int)
        orderIndex = np.sort(orderIndex)
    except:
        orderIndex = incidence.index
    try:
        incidence = incidence.loc[orderIndex,:]
    except:
        incidence = incidence.loc[orderIndex.astype(str),:]
        
    return incidence

def processCoexpressionLists(lists,expressionData,threshold=0.925):
    reconstructed = reconstruction(lists,expressionData,threshold)
    reconstructedList = [reconstructed[i] for i in reconstructed.keys()]
    reconstructedList.sort(key = lambda s: -len(s))
    return reconstructedList

def reviseInitialClusters(clusterList,expressionData,threshold=0.925):
    coexpressionLists = processCoexpressionLists(clusterList,expressionData,threshold)

    for iteration in range(5):
        previousLength = len(coexpressionLists)
        coexpressionLists = processCoexpressionLists(coexpressionLists,expressionData,threshold)
        newLength = len(coexpressionLists)
        if newLength == previousLength:
            break

    coexpressionDict = {str(i):list(coexpressionLists[i]) for i in range(len(coexpressionLists))}

    return coexpressionDict

# =============================================================================
# Functions used for mechanistic inference
# =============================================================================

def regulonDictionary(regulons):
    regulonModules = {}
    #str(i):[regulons[key][j]]}
    df_list = []

    for tf in regulons.keys():
        for key in regulons[tf].keys():
            genes = regulons[tf][key]
            id_ = str(len(regulonModules))
            regulonModules[id_] = regulons[tf][key]
            for gene in genes:
                df_list.append([id_,tf,gene])

    array = np.vstack(df_list)
    df = pd.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]
    
    return regulonModules, df

def principalDf(dict_,expressionData,regulons=None,subkey='genes',minNumberGenes=8,random_state=12):

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
            
        pca = PCA(1,random_state=random_state)
        principalComponents = pca.fit_transform(expressionData.loc[genes,:].T)
        principalDf = pd.DataFrame(principalComponents)
        principalDf.index = expressionData.columns
        principalDf.columns = [str(i)]
        
        normPC = np.linalg.norm(np.array(principalDf.iloc[:,0]))
        pearson = stats.pearsonr(principalDf.iloc[:,0],np.median(expressionData.loc[genes,:],axis=0))
        signCorrection = pearson[0]/np.abs(pearson[0])
        
        principalDf = signCorrection*principalDf/normPC
        
        pcDfs.append(principalDf)
    
    principalMatrix = pd.concat(pcDfs,axis=1)
        
    return principalMatrix

def axisTfs(axesDf,tfList,expressionData,correlationThreshold=0.3):
    
    axesArray = np.array(axesDf.T)
    if correlationThreshold > 0:
        tfArray = np.array(expressionData.loc[tfList,:])
    axes = np.array(axesDf.columns)
    tfDict = {}

    if type(tfList) is list:
        tfs = np.array(tfList)
    elif type(tfList) is not list:
        tfs = tfList
        
    if correlationThreshold == 0:
        for axis in range(axesArray.shape[0]):
            tfDict[axes[axis]] = tfs

        return tfDict       
    
    for axis in range(axesArray.shape[0]):
        tfCorrelation = pearson_array(tfArray,axesArray[axis,:])
        tfDict[axes[axis]] = tfs[np.where(np.abs(tfCorrelation)>=correlationThreshold)[0]]
    
    return tfDict

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

def multiprocess(function,tasks):
    import multiprocessing, multiprocessing.pool
    hydra=multiprocessing.pool.Pool(len(tasks))  
    output=hydra.map(function,tasks)   
    hydra.close()
    hydra.join()
    return output

def hyper(population,set1,set2,overlap):
    
    b = max(set1,set2)
    c = min(set1,set2)
    hyp = stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])
    
    return prb 

def condenseOutput(output):
    
    results = {}
    for i in range(len(output)):
        resultsDict = output[i]
        keys = resultsDict.keys()
        for j in range(len(resultsDict)):
            key = keys[j]
            results[key] = resultsDict[key]
    return results

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

def mechanisticInference(axes,revisedClusters,expressionData,correlationThreshold=0.3,numCores=5,p=0.05,database="tfbsdb_tf_to_genes.pkl",dataFolder=os.path.join(os.path.split(os.getcwd())[0],"data"),override_database=None):
    import os
    
    print('Running mechanistic inference')
    
    if override_database is None:
        tfToGenesPath = os.path.join(dataFolder,"network_dictionaries",database)
        tfToGenes = read_pkl(tfToGenesPath)
        
    elif override_database is not None:
        tfToGenes = override_database
     
    if correlationThreshold <= 0:
        allGenes = [int(len(expressionData.index))]
    elif correlationThreshold > 0:
        allGenes = list(expressionData.index)
        
    tfs = tfToGenes.keys()     
    tfMap = axisTfs(axes,tfs,expressionData,correlationThreshold=correlationThreshold) 
    taskSplit = splitForMultiprocessing(revisedClusters.keys(),numCores)
    tasks = [[taskSplit[i],(allGenes,revisedClusters,tfMap,tfToGenes,p)] for i in range(len(taskSplit))]
    tfbsdbOutput = multiprocess(tfbsdbEnrichment,tasks)
    mechanisticOutput = condenseOutput(tfbsdbOutput)
        
    return mechanisticOutput

def coincidenceMatrix(coregulationModules,key,freqThreshold = 0.333):

    tf = coregulationModules.keys()[key]
    subRegulons = coregulationModules[tf]
    srGenes = list(set(np.hstack([subRegulons[i] for i in subRegulons.keys()])))

    template = pd.DataFrame(np.zeros((len(srGenes),len(srGenes))))
    template.index = srGenes
    template.columns = srGenes
    for key in subRegulons.keys():
        genes = subRegulons[key]
        template.loc[genes,genes]+=1
    trace = np.array([template.iloc[i,i] for i in range(template.shape[0])]).astype(float)
    normDf = ((template.T)/trace).T
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1
    
    return normDf

def getCoregulationModules(mechanisticOutput):
    
    coregulationModules = {}
    for i in mechanisticOutput.keys():
        for key in mechanisticOutput[i].keys():
            if key not in coregulationModules.keys():
                coregulationModules[key] = {}
            genes = mechanisticOutput[i][key][1]
            coregulationModules[key][i] = genes
    return coregulationModules

def getRegulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.333):

    regulons = {}
    keys = coregulationModules.keys()
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = 0.333)
        unmixed = unmix(normDf)   
        remixed = remix(normDf,unmixed)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>minNumberGenes:
                    if tf not in regulons.keys():
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster                    
    return regulons

def getCoexpressionModules(mechanisticOutput):
    coexpressionModules = {}
    for i in mechanisticOutput.keys():
        genes = list(set(np.hstack([mechanisticOutput[i][key][1] for key in mechanisticOutput[i].keys()])))
        coexpressionModules[i] = genes
    return coexpressionModules

def f1Regulons(coregulationModules,minNumberGenes=5,freqThreshold = 0.1):

    regulons = {}
    keys = coregulationModules.keys()
    for i in range(len(keys)):
        tf = keys[i]
        normDf = coincidenceMatrix(coregulationModules,key=i,freqThreshold = freqThreshold)
        remixed = f1Binary(normDf)
        if len(remixed)>0:
            for cluster in remixed:
                if len(cluster)>minNumberGenes:
                    if tf not in regulons.keys():
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster                    
    return regulons

# =============================================================================
# Functions used for post-processing mechanistic inference
# =============================================================================

def convertDictionary(dict_,conversionTable):
    converted = {}
    for i in dict_.keys():
        genes = dict_[i]
        conv_genes = conversionTable[genes]
        for j in range(len(conv_genes)):
            if type(conv_genes[j]) is pd.core.series.Series:
                conv_genes[j] = conv_genes[j][0]        
        converted[i] = list(conv_genes)
    return converted

def convertRegulons(df,conversionTable):
    regIds = []
    regs = []
    genes = []
    for i in range(df.shape[0]):
        regIds.append(df.iloc[i,0])
        tmpReg = conversionTable[df.iloc[i,1]]
        if type(tmpReg) is pd.core.series.Series:
            tmpReg=tmpReg[0]
        regs.append(tmpReg)
        tmpGene = conversionTable[df.iloc[i,2]]
        if type(tmpGene) is pd.core.series.Series:
            tmpGene = tmpGene[0]
        genes.append(tmpGene)
    regulonDfConverted = pd.DataFrame(np.vstack([regIds,regs,genes]).T)
    regulonDfConverted.columns = ["Regulon_ID","Regulator","Gene"]
    return regulonDfConverted

def generateInputForFIRM(revisedClusters,saveFile):

    identifier_mapping = pd.read_csv(os.path.join(os.path.split(os.getcwd())[0],"data","identifier_mappings.txt"),sep="\t")
    identifier_mapping_entrez = identifier_mapping[identifier_mapping.Source == "Entrez Gene ID"]
    identifier_mapping_entrez.index = identifier_mapping_entrez.iloc[:,0]

    Gene = []
    Group = []
    identified_genes = set(identifier_mapping_entrez.index)
    for key in revisedClusters.keys():
        cluster = revisedClusters[key]
        tmp_genes = list(set(cluster)&identified_genes)
        tmp_entrez = list(identifier_mapping_entrez.loc[tmp_genes,"Name"])
        tmp_group = [key for i in range(len(tmp_entrez))]
        Gene.extend(tmp_entrez)
        Group.extend(tmp_group)

    firm_df = pd.DataFrame(np.vstack([Gene,Group]).T)
    firm_df.columns = ["Gene","Group"]    
    firm_df.to_csv(saveFile,index=None,sep="\t")

    return firm_df
    
# =============================================================================
# Functions used for inferring sample subtypes
# =============================================================================

def sampleCoincidenceMatrix(dict_,freqThreshold = 0.333,frequencies=False):

    keys = dict_.keys()
    lists = [dict_[key] for key in keys]
    samples = list(set(np.hstack(lists)))
    
    template = pd.DataFrame(np.zeros((len(samples),len(samples))))
    template.index = samples
    template.columns = samples
    for key in keys:
        hits = dict_[key]
        template.loc[hits,hits]+=1
    trace = np.array([template.iloc[i,i] for i in range(template.shape[0])])
    normDf = ((template.T)/trace).T
    if frequencies is not False:
        return normDf
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1
    
    return normDf

def f1Decomposition(sampleMembers=None,thresholdSFM=0.333,sampleFrequencyMatrix=None):
    # thresholdSFM is the probability cutoff that makes the density of the binary similarityMatrix = 0.15
    # sampleMembers is a dictionary with features as keys and members as elements

    # sampleFrequencyMatrix[i,j] gives the probability that sample j appears in a cluster given that sample i appears
    if sampleFrequencyMatrix is None:
        sampleFrequencyMatrix = sampleCoincidenceMatrix(sampleMembers,freqThreshold = thresholdSFM,frequencies=True)
    # similarityMatrix is defined such that similarityMatrix[i,j] = 1 iff sampleFrequencyMatrix[i,j] >= thresholdSFM
    similarityMatrix = sampleFrequencyMatrix*sampleFrequencyMatrix.T
    similarityMatrix[similarityMatrix<thresholdSFM] = 0
    similarityMatrix[similarityMatrix>0] = 1
    # remainingMembers is the set of set of unclustered members
    remainingMembers = set(similarityMatrix.index)
    # probeSample is the sample that serves as a seed to identify a cluster in a given iteration
    probeSample = np.argmax(similarityMatrix.sum(axis=1))
    # members are the samples that satisfy the similarity condition with the previous cluster or probeSample
    members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
    # nonMembers are the remaining members not in the current cluster
    nonMembers = remainingMembers-members
    # instantiate list to collect clusters of similar members
    similarityClusters = []
    # instantiate f1 score for optimization
    f1 = 0

    for iteration in range(1500):

        predictedMembers = members
        predictedNonMembers = remainingMembers-predictedMembers

        sumSlice = np.sum(similarityMatrix.loc[:,list(predictedMembers)],axis=1)/float(len(predictedMembers))
        members = set(similarityMatrix.index[np.where(sumSlice>0.8)[0]])

        if members==predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break    
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        nonMembers = remainingMembers-members
        TP = len(members&predictedMembers)
        FN = len(predictedNonMembers&members)
        FP = len(predictedMembers&nonMembers)
        tmpf1 = TP/float(TP+FN+FP)

        if tmpf1 <= f1:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers)==0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = np.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[np.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue   

        elif tmpf1 > f1:
            f1 = tmpf1
            continue
    
    similarityClusters.sort(key = lambda s: -len(s))
    print('done!')
    return similarityClusters

def plotSimilarity(similarityMatrix,orderedSamples,vmin=0,vmax=0.5,title="Similarity matrix",xlabel="Samples",ylabel="Samples",fontsize=14,figsize=(7,7),savefig=None):
    fig = plt.figure(figsize=figsize)  
    ax = fig.add_subplot(111)
    try:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    except:
        pass    
    ax.imshow(similarityMatrix.loc[orderedSamples,orderedSamples],cmap='viridis',vmin=vmin,vmax=vmax)
    ax.grid(False)
    plt.title(title,FontSize=fontsize+2)
    plt.xlabel(xlabel,FontSize=fontsize)
    plt.ylabel(ylabel,FontSize=fontsize)
    if savefig is not None:
        plt.savefig(savefig,bbox_inches="tight")
    return

def f1(vector1,vector2):
    members = set(np.where(vector1==1)[0])
    nonMembers = set(np.where(vector1==0)[0])
    predictedMembers = set(np.where(vector2==1)[0])
    predictedNonMembers = set(np.where(vector2==0)[0])
    
    TP = len(members&predictedMembers)
    FN = len(predictedNonMembers&members)
    FP = len(predictedMembers&nonMembers)
    if TP == 0:
        return 0.0
    F1 = TP/float(TP+FN+FP)
    
    return F1

def centroidExpansion(classes,sampleMatrix,f1Threshold = 0.3,returnCentroids=None):
    centroids = []
    for i in range(len(classes)):
        clusterComponents = sampleMatrix.loc[:,classes[i]]
        class1 = np.mean(clusterComponents,axis=1)
        hits = np.where(class1>0.6)[0]
        centroid = pd.DataFrame(sampleMatrix.iloc[:,0])
        centroid.columns = [i]
        centroid[i] = 0
        centroid.iloc[hits,0] = 1
        centroids.append(centroid)

    miss = []
    centroidClusters = [[] for i in range(len(centroids))]
    for smpl in sampleMatrix.columns:
        probeVector = np.array(sampleMatrix[smpl])
        scores = []
        for ix in range(len(centroids)):
            tmp = f1(np.array(probeVector),centroids[ix])
            scores.append(tmp)
        scores = np.array(scores)
        match = np.argmax(scores)
        if scores[match] < f1Threshold:
            miss.append(smpl)
        elif scores[match] >= f1Threshold:
            centroidClusters[match].append(smpl)

    centroidClusters.append(miss)  

    if returnCentroids is not None:
        centroidMatrix = pd.DataFrame(pd.concat(centroids,axis=1))
        return centroidClusters, centroidMatrix
        
    return centroidClusters

def getCentroids(classes,sampleMatrix):
    centroids = []
    for i in range(len(classes)):
        clusterComponents = sampleMatrix.loc[:,classes[i]]
        class1 = np.mean(clusterComponents,axis=1)
        centroid = pd.DataFrame(class1)
        centroid.columns = [i]
        centroid.index = sampleMatrix.index
        centroids.append(centroid)
    return pd.concat(centroids,axis=1)

def mapExpressionToNetwork(centroidMatrix,membershipMatrix,threshold = 0.05):
    
    miss = []
    centroidClusters = [[] for i in range(centroidMatrix.shape[1])]
    for smpl in membershipMatrix.columns:
        probeVector = np.array(membershipMatrix[smpl])
        scores = []
        for ix in range(centroidMatrix.shape[1]):
            tmp = f1(np.array(probeVector),np.array(centroidMatrix.iloc[:,ix]))
            scores.append(tmp)
        scores = np.array(scores)
        match = np.argmax(scores)
        if scores[match] < threshold:
            miss.append(smpl)
        elif scores[match] >= threshold:
            centroidClusters[match].append(smpl)
    centroidClusters.append(miss)
    
    return centroidClusters

def orderMembership(centroidMatrix,membershipMatrix,mappedClusters,ylabel="",resultsDirectory=None,showplot=False):

    centroidRank = []
    alreadyMapped = []
    for ix in range(centroidMatrix.shape[1]):
        tmp = np.where(centroidMatrix.iloc[:,ix]==1)[0]
        signature = list(set(tmp)-set(alreadyMapped))
        centroidRank.extend(signature)
        alreadyMapped.extend(signature)
    orderedClusters = centroidMatrix.index[np.array(centroidRank)]
    try:
        ordered_matrix = membershipMatrix.loc[orderedClusters,np.hstack(mappedClusters)]
    except:
        ordered_matrix = membershipMatrix.loc[np.array(orderedClusters).astype(int),np.hstack(mappedClusters)]
    
    if showplot is False:
        return ordered_matrix
    
    if showplot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except:
            pass    
        ax.imshow(ordered_matrix,cmap='viridis',aspect="auto")
        ax.grid(False)
            
        plt.title(ylabel.split("s")[0]+"Activation",FontSize=16)
        plt.xlabel("Samples",FontSize=14)
        plt.ylabel(ylabel,FontSize=14)
        if resultsDirectory is not None:
            plt.savefig(os.path.join(resultsDirectory,"binaryActivityMap.pdf"))
    return ordered_matrix

def plotDifferentialMatrix(overExpressedMembersMatrix,underExpressedMembersMatrix,orderedOverExpressedMembers,cmap="viridis",aspect="auto",saveFile=None,showplot=False):
    differentialActivationMatrix = overExpressedMembersMatrix-underExpressedMembersMatrix
    orderedDM = differentialActivationMatrix.loc[orderedOverExpressedMembers.index,orderedOverExpressedMembers.columns]

    if showplot is False:
        
        return orderedDM
    
    elif showplot is True:
        fig = plt.figure(figsize=(7,7))  
        ax = fig.add_subplot(111)
        try:
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except:
            pass
        
        ax.imshow(orderedDM,cmap=cmap,vmin=-1,vmax=1,aspect=aspect)
        ax.grid(False)
        if saveFile is not None:
            plt.ylabel("Regulons",FontSize=14)
            plt.xlabel("Samples",FontSize=14)
            ax.grid(False)        
            plt.savefig(saveFile,bbox_inches="tight")
            
    return orderedDM

def kmeans(df,numClusters,random_state=None):
    from sklearn.cluster import KMeans

    if random_state is not None:
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters,random_state=random_state)

    elif random_state is None:    
        # Number of clusters
        kmeans = KMeans(n_clusters=numClusters)
    
    # Fitting the input data
    kmeans = kmeans.fit(df)
    # Getting the cluster labels
    labels = kmeans.predict(df)
    # Centroid values
    centroids = kmeans.cluster_centers_

    clusters = []
    for i in range(numClusters):
        clstr = df.index[np.where(labels==i)[0]]
        clusters.append(clstr)

    return clusters, labels, centroids

def mosaic(dfr,clusterList,minClusterSize_x=4,minClusterSize_y=5,allow_singletons=True,max_groups=50,saveFile=None,random_state=12):    
    
    import sklearn

    lowResolutionPrograms = [[] for i in range(len(clusterList))]
    sorting_hat = []
    for i in range(len(clusterList)):
        patients = clusterList[i]
        if len(patients) < minClusterSize_x:
            continue
        subset = dfr.loc[:,patients]
        density = subset.sum(axis=1)/float(subset.shape[1])
        sorting_hat.append(np.array(density))
    
    enrichment_matrix = np.vstack(sorting_hat).T    
    choice = np.argmax(enrichment_matrix,axis=1)
    for i in range(dfr.shape[0]):
        lowResolutionPrograms[choice[i]].append(dfr.index[i])
    
    #Cluster modules into transcriptional programs
    
    y_clusters = []
    for program in range(len(lowResolutionPrograms)):
        regs = lowResolutionPrograms[program]
        if len(regs) == 0:
            continue
        df = dfr.loc[regs,:]
        sil_scores = []
        max_clusters_y = min(max_groups,int(len(regs)/3.))
        for numClusters_y in range(2,max_clusters_y):
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=numClusters_y,random_state=random_state)
            lens_y = [len(c) for c in clusters_y]
            if min(lens_y) < minClusterSize_y:
                if allow_singletons is True:
                    if min(lens_y) != 1:
                        kmSS = 0
                        sil_scores.append(kmSS)
                        continue
                    elif min(lens_y) == 1:
                        pass
                elif allow_singletons is not True:
                    kmSS = 0
                    sil_scores.append(kmSS)
                    continue
                
            clusters_y.sort(key=lambda s: -len(s))
    
            kmSS=sklearn.metrics.silhouette_score(df,labels_y,metric='euclidean')
            sil_scores.append(kmSS)
    
        if len(sil_scores) > 0:
            top_hit = min(np.where(np.array(sil_scores)>=0.95*max(sil_scores))[0]+2)
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=top_hit,random_state=random_state)
            clusters_y.sort(key=lambda s: -len(s))
            y_clusters.append(list(clusters_y))
    
        elif len(sil_scores) == 0:
            y_clusters.append(regs)
        
    order_y = np.hstack([np.hstack(y_clusters[i]) for i in range(len(y_clusters))])
    
    #Cluster patients into subtype states
    
    x_clusters = []
    for c in range(len(clusterList)):
        patients = clusterList[c]
        if len(patients)<= minClusterSize_x:
            x_clusters.append(patients)
            continue
        
        if allow_singletons is not True:
            if len(patients)<= 2*minClusterSize_x:
                x_clusters.append(patients)
                continue            
        
        if len(patients) == 0:
            continue
        df = dfr.loc[order_y,patients].T
        sil_scores = []
        
        max_clusters_x = min(max_groups,int(len(patients)/3.))
        for numClusters_x in range(2,max_clusters_x):
            clusters_x, labels_x, centroids_x = kmeans(df,numClusters=numClusters_x,random_state=random_state)
            lens_x = [len(c) for c in clusters_x]
            if min(lens_x) < minClusterSize_x:
                if allow_singletons is True:
                    if min(lens_x) != 1:
                        kmSS = 0
                        sil_scores.append(kmSS)
                        continue
                    elif min(lens_x) == 1:
                        pass
                elif allow_singletons is not True:
                    kmSS = 0
                    sil_scores.append(kmSS)
                    continue
                
            clusters_x.sort(key=lambda s: -len(s))
    
            kmSS=sklearn.metrics.silhouette_score(df,labels_x,metric='euclidean')
            sil_scores.append(kmSS)
    
        if len(sil_scores) > 0:
            top_hit = min(np.where(np.array(sil_scores)>=0.999*max(sil_scores))[0]+2)
            clusters_x, labels_x, centroids_x = kmeans(df,numClusters=top_hit,random_state=random_state)
            clusters_x.sort(key=lambda s: -len(s))
            x_clusters.append(list(clusters_x))
        elif len(sil_scores) == 0:
            x_clusters.append(patients)
    try:
        micro_states = []
        for i in range(len(x_clusters)):
            if len(x_clusters[i])>0:
                if type(x_clusters[i][0]) is not str:
                    for j in range(len(x_clusters[i])):
                        micro_states.append(x_clusters[i][j])
                elif type(x_clusters[i][0]) is str:
                    micro_states.append(x_clusters[i])
                               
        order_x = np.hstack(micro_states)
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(dfr.loc[order_y,order_x],cmap="bwr",vmin=-1,vmax=1)
        ax.set_aspect(dfr.shape[1]/float(dfr.shape[0]))
        ax.grid(False)
        ax.set_ylabel("Regulons",FontSize=14)
        ax.set_xlabel("Samples",FontSize=14)
        if saveFile is not None:
            plt.savefig(saveFile,bbox_inches="tight")
            
        return y_clusters, micro_states
    
    except:
        pass
        
    return y_clusters, x_clusters

def transcriptionalPrograms(programs,reference_dictionary):
    transcriptionalPrograms = {}
    programRegulons = {}
    p_stack = []
    programs_flattened = np.array(programs).flatten()
    for i in range(len(programs_flattened)):
        if len(np.hstack(programs_flattened[i]))>len(programs_flattened[i]):
            for j in range(len(programs_flattened[i])):
                p_stack.append(list(programs_flattened[i][j]))        
        else:
            p_stack.append(list(programs_flattened[i]))
    
    
    for j in range(len(p_stack)):
        key = ("").join(["TP",str(j)])
        regulonList = [i for i in p_stack[j]]
        programRegulons[key] = regulonList
        tmp = [reference_dictionary[i] for i in p_stack[j]]
        transcriptionalPrograms[key] = list(set(np.hstack(tmp)))
    return transcriptionalPrograms, programRegulons

def reduceModules(df,programs,states,stateThreshold=0.75,saveFile=None):
    
    df = df.loc[:,np.hstack(states)]
    statesDf = pd.DataFrame(np.zeros((len(programs),df.shape[1])))
    statesDf.index = range(len(programs))
    statesDf.columns = df.columns

    for i in range(len(programs)):
        state = programs[i]
        subset = df.loc[state,:]

        state_scores = subset.sum(axis=0)/float(subset.shape[0])

        keep_high = np.where(state_scores>=stateThreshold)[0]
        keep_low = np.where(state_scores<=-1*stateThreshold)[0]
        hits_high = np.array(df.columns)[keep_high]
        hits_low = np.array(df.columns)[keep_low]

        statesDf.loc[i,hits_high] = 1
        statesDf.loc[i,hits_low] = -1
        
    if saveFile is not None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(statesDf,cmap="bwr",vmin=-1,vmax=1,aspect='auto')
        ax.grid(False)
        ax.set_ylabel("Transcriptional programs",FontSize=14)
        ax.set_xlabel("Samples",FontSize=14)
        plt.savefig(saveFile,bbox_inches="tight")
        
    return statesDf

def programsVsStates(statesDf,states,filename=None,showplot=False):
    pixel = np.zeros((statesDf.shape[0],len(states)))
    for i in range(statesDf.shape[0]):
        for j in range(len(states)):
            pixel[i,j] = np.mean(statesDf.loc[statesDf.index[i],states[j]])

    pixel = pd.DataFrame(pixel)
    pixel.index = statesDf.index

    if showplot is False:
        return pixel
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pixel,cmap="bwr",vmin=-1,vmax=1,aspect="auto")
    ax.grid(False)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Transcriptional programs",FontSize=14)
    plt.xlabel("Transcriptional states",FontSize=14)
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")

    return pixel

def getStratifyingRegulons(states_list_1,states_list_2,reference_matrix,p=0.05,plot=True):
    
    if type(states_list_1[0]) is 'str':
        states_list_1 = [states_list_1]
        
    if type(states_list_2[0]) is 'str':
        states_list_2 = [states_list_2]
        
    matrix1 = reference_matrix.loc[:,np.hstack(states_list_1)]
    matrix2 = reference_matrix.loc[:,np.hstack(states_list_2)]

    ttest = stats.ttest_ind(matrix1,matrix2,axis=1,equal_var=False)
    
    if min(ttest[1]) > p:
        print("No hits detected. Cutoff p-value is too strict")
        return []

    results = pd.DataFrame(np.vstack(ttest).T)
    results.index = reference_matrix.index
    results.columns = ["t-statistic","p-value"]
    results = results[results["p-value"]<=p]
    results.sort_values(by="t-statistic",ascending=False,inplace=True)
    print(results)

    if plot is True:
        ttest_data_source = reference_matrix.loc[results.index,np.hstack([np.hstack(states_list_1),np.hstack(states_list_2)])]
        figure = plt.figure()
        ax = figure.gca()
        ax.imshow(ttest_data_source,cmap="bwr",aspect="auto")
        ax.grid(False)

    return results

# =============================================================================
# Functions used for cluster analysis
# =============================================================================

def getEigengenes(coexpressionModules,expressionData,regulon_dict=None,saveFolder=None):
    eigengenes = principalDf(coexpressionModules,expressionData,subkey=None,regulons=regulon_dict,minNumberGenes=1)
    eigengenes = eigengenes.T
    index = np.sort(np.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    if saveFolder is not None:
        eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))
    return eigengenes

def parallelEnrichment(task):
    partition = task[0]
    test_keys, dict_, reference_dict, reciprocal_dict, population_len, threshold = task[1]
    test_keys = test_keys[partition[0]:partition[1]]
           
    results_dict = {}
    for ix in test_keys:
        basline_ps = {k:1 for k in reference_dict.keys()}
        genes_interrogated = dict_[ix]
        genes_overlapping = list(set(genes_interrogated)&set(reciprocal_dict))
        count_overlapping = Counter(np.hstack([reciprocal_dict[i] for i in genes_overlapping]))
        rank_overlapping = count_overlapping.most_common()

        for h in range(len(rank_overlapping)):
            ct = rank_overlapping[h][1]
            if ct==1:
                break
            key = rank_overlapping[h][0]
            basline_ps[key] = hyper(population_len,len(reference_dict[key]),len(genes_interrogated),rank_overlapping[h][1]) 

        above_basline_ps = {key:basline_ps[key] for key in basline_ps.keys() if basline_ps[key]<threshold}
        results_dict[ix] = above_basline_ps

    return results_dict

def enrichmentAnalysis(dict_,reference_dict,reciprocal_dict,genes_with_expression,resultsDirectory,numCores=5,min_overlap = 3,threshold = 0.05):
    t1 = time.time()
    print('initializing enrichment analysis')
    
    os.chdir(os.path.join(resultsDirectory,"..","data","network_dictionaries"))
    reference_dict = read_pkl(reference_dict)
    reciprocal_dict = read_pkl(reciprocal_dict)
    os.chdir(os.path.join(resultsDirectory,"..","src"))

    genes_in_reference_dict = reciprocal_dict.keys()
    population_len = len(set(genes_with_expression)&set(genes_in_reference_dict))

    reciprocal_keys = set(reciprocal_dict.keys())
    test_keys = []
    for key in dict_.keys():
        genes_interrogated = dict_[key]
        genes_overlapping = list(set(genes_interrogated)&reciprocal_keys)
        if len(genes_overlapping) < min_overlap:
            continue
        count_overlapping = Counter(np.hstack([reciprocal_dict[i] for i in genes_overlapping]))
        rank_overlapping = count_overlapping.most_common()
        if rank_overlapping[0][1] < min_overlap:
            continue

        test_keys.append(key)

    try:
        taskSplit = splitForMultiprocessing(test_keys,numCores)
        taskData = (test_keys, dict_, reference_dict, reciprocal_dict, population_len,threshold)
        tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
        enrichmentOutput = multiprocess(parallelEnrichment,tasks)
        combinedResults = condenseOutput(enrichmentOutput)
    except:
        combinedResults = {}
    
    t2 = time.time()
    print('completed enrichment analysis in {:.2f} seconds'.format(t2-t1))

    return combinedResults

def convertGO(goBio_enriched,resultsDirectory):
    goConversionPath = os.path.join(resultsDirectory,"..","data","network_dictionaries","GO_terms_conversion.csv")
    goBioConversion = pd.read_csv(goConversionPath,index_col=0,header=0)
    go_terms_enriched = {}
    for module in goBio_enriched.keys():
        conv = {}
        for key in goBio_enriched[module].keys():
            tmp = goBioConversion.loc[key,"GO_function"]
            conv[tmp] = goBio_enriched[module][key]
        go_terms_enriched[module] = conv
    return go_terms_enriched

def tsne(matrix,perplexity=100,n_components=2,n_iter=1000,plotOnly=True,plotColor="red",alpha=0.4,dataOnly=False):
    from sklearn.manifold import TSNE
    X = np.array(matrix.T)
    X_embedded = TSNE(n_components=n_components, n_iter=n_iter, n_iter_without_progress=300,init='random',
                             random_state=0, perplexity=perplexity).fit_transform(X)
    if plotOnly is True:
        plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)
        return
    if dataOnly is True:
        return X_embedded
    
    plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)
        
    return X_embedded

def tsneStateLabels(tsneDf,states):
    labelsDf = pd.DataFrame(1000*np.ones(tsneDf.shape[0]))
    labelsDf.index = tsneDf.index
    labelsDf.columns = ["label"]

    for i in range(len(states)):
        tagged = states[i]
        labelsDf.loc[tagged,"label"] = i
    state_labels = np.array(labelsDf.iloc[:,0])  
    return state_labels

def plotStates(statesDf,tsneDf,numCols=None,numRows=None,saveFile=None,size=10,aspect=1,scale=2):

    if numRows is None:
        if numCols is None:
            numRows = int(round(np.sqrt(statesDf.shape[0])))
            rat = np.floor(statesDf.shape[0]/float(numRows))
            rem = statesDf.shape[0]-numRows*rat
            numCols = int(rat+rem)
        elif numCols is not None:
            numRows = int(np.ceil(float(statesDf.shape[0])/numCols))
        
    fig = plt.figure(figsize=(scale*numRows,scale*numCols))
    for ix in range(statesDf.shape[0]):
        ax = fig.add_subplot(numRows,numCols,ix+1)
        # overlay single state onto tSNE plot
        stateIndex = ix

        group = pd.DataFrame(np.zeros(statesDf.shape[1]))
        group.index = statesDf.columns
        group.columns = ["status"]
        group.loc[statesDf.columns,"status"] = list(statesDf.iloc[stateIndex,:])
        group = np.array(group.iloc[:,0])
        ax.set_aspect(aspect)
        ax.scatter(tsneDf.iloc[:,0],tsneDf.iloc[:,1],cmap="bwr",c=group,vmin=-1,vmax=1,s=size)

    if saveFile is not None:
        plt.savefig(saveFile,bbox_inches="tight")

    return

# =============================================================================
# Functions used for survival analysis
# =============================================================================

def kmAnalysis(survivalDf,durationCol,statusCol,saveFile=None):
    
    from lifelines import KaplanMeierFitter
    from scipy import stats
    

    kmf = KaplanMeierFitter()
    kmf.fit(survivalDf.loc[:,durationCol],survivalDf.loc[:,statusCol])
    survFunc = kmf.survival_function_
    
    m, b, r, p, e = stats.linregress(list(survFunc.index),survFunc.iloc[:,0])
    
    survivalDf = survivalDf.sort_values(by=durationCol)
    ttpfs = np.array(survivalDf.loc[:,durationCol])
    survTime = np.array(survFunc.index)
    survProb = []
    
    for i in range(len(ttpfs)):
        date = ttpfs[i]
        if date in survTime:
            survProb.append(survFunc.loc[date,"KM_estimate"])
        elif date not in survTime:
            lbix = np.where(np.array(survFunc.index)<date)[0][-1]
            est = 0.5*(survFunc.iloc[lbix,0]+survFunc.iloc[lbix+1,0])
            survProb.append(est)
            
    kmEstimate = pd.DataFrame(survProb)
    kmEstimate.columns = ["kmEstimate"]
    kmEstimate.index = survivalDf.index
    
    pfsDf = pd.concat([survivalDf,kmEstimate],axis=1)
    
    if saveFile is not None:
        pfsDf.to_csv(saveFile)
        
    return pfsDf

def guanRank(kmSurvival,saveFile=None):

    gScore = []
    for A in range(kmSurvival.shape[0]):
        aScore = 0
        aPfs = kmSurvival.iloc[A,0]
        aStatus = kmSurvival.iloc[A,1]
        aProbPFS = kmSurvival.iloc[A,2]
        if aStatus == 1:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]            
                if bPfs > aPfs:
                    aScore+=1
                if bPfs <= aPfs:
                    if bStatus == 0:
                        aScore+=aProbPFS/bProbPFS
                if bPfs == aPfs:
                    if bStatus == 1:
                        aScore+=0.5
        elif aStatus == 0:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]            
                if bPfs >= aPfs:
                    if bStatus == 0:
                        tmp = 1-0.5*bProbPFS/aProbPFS
                        aScore+=tmp
                    elif bStatus == 1:
                        tmp = 1-bProbPFS/aProbPFS
                        aScore+=tmp                    
                if bPfs < aPfs:
                    if bStatus == 0:
                        aScore+=0.5*aProbPFS/bProbPFS
        gScore.append(aScore)
                        
    GuanScore = pd.DataFrame(gScore)  
    GuanScore = GuanScore/float(max(gScore))
    GuanScore.index = kmSurvival.index
    GuanScore.columns = ["GuanScore"] 
    survivalData = pd.concat([kmSurvival,GuanScore],axis=1)
    survivalData.sort_values(by="GuanScore",ascending=False,inplace=True)
    
    if saveFile is not None:
        survivalData.to_csv(saveFile)
        
    return survivalData

def survivalMedianAnalysis(task):
    
    from lifelines import CoxPHFitter
    
    start, stop = task[0]
    referenceDictionary,expressionDf,SurvivalDf = task[1]    
    
    overlapPatients = list(set(expressionDf.columns)&set(SurvivalDf.index))
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]    

    coxResults = {}
    keys = referenceDictionary.keys()[start:stop]
    ct=0
    for key in keys:
        ct+=1
        if ct%10==0:
            print(ct)
        try:
            geneset = referenceDictionary[key]
            cluster = expressionDf.loc[geneset,overlapPatients]
            nz = np.count_nonzero(cluster+4.01,axis=0)
            
            medians = []
            for i in range(cluster.shape[1]):
                if nz[i] >= 3:
                    median = np.median(cluster.iloc[:,i][cluster.iloc[:,i]>-4.01])
                elif nz[i] < 3:
                    median = np.median(cluster.iloc[:,i])
                medians.append(median)
                
            medianDf = pd.DataFrame(medians)
            medianDf.index = overlapPatients
            medianDf.columns = ["median"]
            Survival = pd.concat([Survival,medianDf],axis=1)
            Survival.sort_values(by=Survival.columns[0],inplace=True)
            
            cph = CoxPHFitter()
            cph.fit(Survival, duration_col=Survival.columns[0], event_col=Survival.columns[1])
        
            tmpcph = cph.summary
            
            cox_hr = tmpcph.loc[key,"z"]
            cox_p = tmpcph.loc[key,"p"]  
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)
        
    return coxResults

def survivalMembershipAnalysis(task):

    from lifelines import CoxPHFitter

    start, stop = task[0]
    membershipDf,SurvivalDf = task[1]
    
    overlapPatients = list(set(membershipDf.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return 
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
       
    coxResults = {}
    keys = membershipDf.index[start:stop]
    ct=0
    for key in keys:
        ct+=1
        if ct%10==0:
            print(ct)
        try:
            memberVector = pd.DataFrame(membershipDf.loc[key,overlapPatients])
            Survival2 = pd.concat([Survival,memberVector],axis=1)
            Survival2.sort_values(by=Survival2.columns[0],inplace=True)
            
            cph = CoxPHFitter()
            cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
            
            tmpcph = cph.summary
            
            cox_hr = tmpcph.loc[key,"z"]
            cox_p = tmpcph.loc[key,"p"]  
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)
    return coxResults

def survivalMembershipAnalysisDirect(membershipDf,SurvivalDf,key):

    from lifelines import CoxPHFitter
    
    overlapPatients = list(set(membershipDf.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return 
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]
       
    coxResults = {}

    memberVector = pd.DataFrame(membershipDf.loc[key,overlapPatients])
    Survival2 = pd.concat([Survival,memberVector],axis=1)
    Survival2.sort_values(by=Survival2.columns[0],inplace=True)
    
    cph = CoxPHFitter()
    cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
    
    tmpcph = cph.summary
    
    cox_hr = tmpcph.loc[key,"z"]
    cox_p = tmpcph.loc[key,"p"]  
    coxResults[key] = (cox_hr, cox_p)
        
    return coxResults

def parallelMedianSurvivalAnalysis(referenceDictionary,expressionDf,numCores=5,survivalPath=None,survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(referenceDictionary.keys(),numCores)
    taskData = (referenceDictionary,expressionDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMedianAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)

    return survivalAnalysis

def parallelMemberSurvivalAnalysis(membershipDf,numCores=5,survivalPath=None,survivalData=None):

    if survivalData is None:
        survivalData = pd.read_csv(survivalPath,index_col=0,header=0)
    taskSplit = splitForMultiprocessing(membershipDf.index,numCores)
    taskData = (membershipDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalMembershipAnalysis,tasks)
    survivalAnalysis = condenseOutput(coxOutput)
    
    return survivalAnalysis

def survivalAnalysis(task):
    
    from lifelines import CoxPHFitter
    
    start, stop = task[0]
    expressionDf,SurvivalDf = task[1]    
    
    overlapPatients = list(set(expressionDf.columns)&set(SurvivalDf.index))
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]    

    coxResults = {}
    keys = expressionDf.index[start:stop]

    for key in keys:
        values = np.array(expressionDf.loc[key,overlapPatients])
        try:        
            medianDf = pd.DataFrame(values)
            medianDf.index = overlapPatients
            medianDf.columns = ["value"]
            Survival2 = pd.concat([Survival,medianDf],axis=1)
            Survival2.sort_values(by=Survival2.columns[0],inplace=True)
        
            cph = CoxPHFitter()
            cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])
        
            tmpcph = cph.summary
        
            cox_hr = tmpcph.loc["value","z"]
            cox_p = tmpcph.loc["value","p"] 
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)
        
    return coxResults

def parallelSurvivalAnalysis(expressionDf,survivalData,numCores=5):

    taskSplit = splitForMultiprocessing(expressionDf.index,numCores)
    taskData = (expressionDf,survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = multiprocess(survivalAnalysis,tasks)
    survivalResults = condenseOutput(coxOutput)

    return survivalResults

def kmplot(srv,groups,labels,xlim_=None,filename=None):

    for group in groups:
        patients = list(set(srv.index)&set(group))
        kmDf = kmAnalysis(survivalDf=srv.loc[patients,["duration","observed"]],durationCol="duration",statusCol="observed")
        subset = kmDf[kmDf.loc[:,"observed"]==1]
        duration = np.concatenate([np.array([0]),np.array(subset.loc[:,"duration"])])
        kme = np.concatenate([np.array([1]),np.array(subset.loc[:,"kmEstimate"])])
        plt.step(duration,kme)

    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([0,1.09])
    if xlim_ is not None:
        axes.set_xlim([xlim_[0],xlim_[1]])
    axes.set_title("Progression-free survival",FontSize=16)
    axes.set_ylabel("Surviving proportion",FontSize=14)
    axes.set_xlabel("Timeline",FontSize=14)
    axes.legend(labels = labels)
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")
    return

def combinedStates(groups,ranked_groups,survivalDf,minSamples=4,maxStates=7):
    high_risk_indices = []
    for i in range(1,len(ranked_groups)+1):
        tmp_group = ranked_groups[-i]
        tmp_len = len(set(survivalDf.index)&set(groups[tmp_group]))
        if tmp_len >= minSamples:
            high_risk_indices.append(tmp_group)
        if len(high_risk_indices) >=maxStates:
            break

    combinations_high = []        
    for i in range(len(high_risk_indices)-1):
        combinations_high.append(high_risk_indices[0:i+1])

    low_risk_indices = []
    for i in range(len(ranked_groups)):
        tmp_group = ranked_groups[i]
        tmp_len = len(set(survivalDf.index)&set(groups[tmp_group]))
        if tmp_len >= minSamples:
            low_risk_indices.append(tmp_group)
        if len(low_risk_indices) >=maxStates:
            break

    combinations_low = []        
    for i in range(len(low_risk_indices)-1):
        combinations_low.append(low_risk_indices[0:i+1])

    combined_states_high = []
    for i in range(len(combinations_high)):
        tmp = []
        for j in range(len(combinations_high[i])):
            tmp.append(groups[combinations_high[i][j]]) 
        combined_states_high.append(np.hstack(tmp))

    combined_states_low = []
    for i in range(len(combinations_low)):
        tmp = []
        for j in range(len(combinations_low[i])):
            tmp.append(groups[combinations_low[i][j]]) 
        combined_states_low.append(np.hstack(tmp))

    combined_states = np.concatenate([combined_states_high,combined_states_low])
    combined_indices_high = ["&".join(np.array(combinations_high[i]).astype(str)) for i in range(len(combinations_high))]
    combined_indices_low = ["&".join(np.array(combinations_low[i]).astype(str)) for i in range(len(combinations_low))]
    combined_indices = np.concatenate([combined_indices_high,combined_indices_low])

    return combined_states, combined_indices
 
# =============================================================================
# Functions used for causal inference
# =============================================================================

def causalNetworkAnalysis(regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,resultsDirectory,minRegulons=1,significance_threshold=0.05,causalFolder="causal_results"):
    
    if not os.path.isdir(resultsDirectory):
        os.mkdir(resultsDirectory)    
    # create results directory
    causal_path = os.path.join(resultsDirectory,causalFolder)
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    t1 = time.time()
    ###
    regulon_df_bcindex = regulon_matrix.copy()
    regulon_df_bcindex.index = np.array(regulon_df_bcindex["Regulon_ID"]).astype(str)

    regulon_df_gene_index = regulon_matrix.copy()
    regulon_df_gene_index.index = regulon_df_gene_index["Gene"]

    tf_name = []
    bc_name = []
    rs_1 = []
    ps_1 = []
    index_1 = []
    
    missing_tfs = list(set(regulon_df_bcindex.loc[:,"Regulator"])-set(expression_matrix.index))
    for key in list(set(regulon_df_bcindex.index)):
        e_gene = reference_matrix.loc[str(key),:]
        tf = list(regulon_df_bcindex.loc[key,"Regulator"])[0]
        if tf not in missing_tfs:
            tf_exp = expression_matrix.loc[tf,reference_matrix.columns]
            r, p = stats.spearmanr(tf_exp, e_gene)
        else:
            r, p = (0,1)
        tf_name.append(tf)
        bc_name.append(key)
        rs_1.append(r)
        ps_1.append(p)
        index_1.append(key)

    correlation_df_bcindex = pd.DataFrame(np.vstack([tf_name,bc_name,rs_1,ps_1]).T) # Table
    correlation_df_bcindex.columns = ["Regulator","Regulon_ID","Spearman_R","Spearman_p"]
    correlation_df_bcindex.index = np.array(index_1).astype(str)

    correlation_df_regulator_index = correlation_df_bcindex.copy() # Table
    correlation_df_regulator_index.index = correlation_df_regulator_index["Regulator"]

    ###
    for mut_ix in range(mutation_matrix.shape[0]):

        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))
        phenotype_2 = list(set(phenotype_2)&set(reference_matrix.columns))
        phenotype_1 = list(set(phenotype_1)&set(reference_matrix.columns))
        
        regulon_ttests = pd.DataFrame(
            np.vstack(
                stats.ttest_ind(reference_matrix.loc[:,phenotype_2],reference_matrix.loc[:,phenotype_1],equal_var=False,axis=1)
            ).T
        )

        regulon_ttests.index = reference_matrix.index
        regulon_ttests.columns = ["Regulon_t-test_t","Regulon_t-test_p"] # Table1: eigengenes ttests

        result_dfs = []
        mean_ts = []
        mean_significance = []

        upstream_regulators = list(set(regulon_matrix.Regulator)-set(regulon_df_gene_index.index))
        for regulator_ in list(set(regulon_matrix.Regulator)&set(regulon_df_gene_index.index)): # analyze all regulators in regulon_matrix

            if regulator_ not in upstream_regulators:
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)

            else:
                xt, xp = stats.ttest_ind(expression_matrix.loc[regulator_,phenotype_2],expression_matrix.loc[regulator_,phenotype_1],equal_var=False)
                mean_ts = xt
                mean_significance = -np.log10(xp)

            if mean_significance >= -np.log10(significance_threshold):
                downstream_tmp = correlation_df_regulator_index.loc[regulator_,"Regulon_ID"]
                if type(downstream_tmp) is not pd.core.series.Series:
                    downstream_regulons = [str(downstream_tmp)]
                elif type(downstream_tmp) is pd.core.series.Series:
                    downstream_regulons = list(np.array(downstream_tmp).astype(str))

                if len(downstream_regulons)<minRegulons:
                    continue

                d_neglogps = []
                d_ts = []
                for downstream_regulon_ in downstream_regulons:
                    dt, dp = list(regulon_ttests.loc[downstream_regulon_,:])
                    tmp_neglogp = -np.log10(dp)
                    d_neglogps.append(tmp_neglogp)
                    d_ts.append(dt)

                d_neglogps = np.array(d_neglogps)
                d_ts = np.array(d_ts)

                mask = np.where(d_neglogps >= -np.log10(significance_threshold))[0]
                if len(mask) == 0:
                    continue

                significant_regulons = np.array(downstream_regulons)[mask]
                significant_regulon_ts = d_ts[mask]
                significant_regulon_ps = d_neglogps[mask]

                significant_Rs = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_R"]).astype(float)
                significant_ps = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_p"]).astype(float)

                assignment_values = mean_ts*significant_Rs*significant_regulon_ts
                #assignments = assignment_values/np.abs(assignment_values)
                alignment_mask = np.where(assignment_values>0)[0]

                if len(alignment_mask) == 0:
                    continue

                mutation_list = np.array([mutation_name for i in range(len(alignment_mask))])
                regulator_list = np.array([regulator_ for i in range(len(alignment_mask))])
                bicluster_list = significant_regulons[alignment_mask]
                mutation_regulator_edge_direction = np.array([mean_ts/np.abs(mean_ts) for i in range(len(alignment_mask))])
                mutation_regulator_edge_ps = np.array([mean_significance for i in range(len(alignment_mask))])
                regulator_bicluster_rs = significant_Rs[alignment_mask]
                regulator_bicluster_ps = significant_ps[alignment_mask]    
                bicluster_ts = significant_regulon_ts[alignment_mask]
                bicluster_ps = significant_regulon_ps[alignment_mask]  
                fraction_aligned = np.array([len(alignment_mask)/float(len(mask)) for i in range(len(alignment_mask))])


                results_ = pd.DataFrame(
                    np.vstack(
                        [
                            mutation_list,
                            regulator_list,
                            bicluster_list,
                            mutation_regulator_edge_direction,
                            mutation_regulator_edge_ps,
                            regulator_bicluster_rs,
                            regulator_bicluster_ps,
                            bicluster_ts,
                            bicluster_ps,
                            fraction_aligned
                        ]
                    ).T
                )

                results_.columns = [
                    "Mutation",
                    "Regulator",
                    "Regulon",
                    "MutationRegulatorEdge",
                    "-log10(p)_MutationRegulatorEdge",
                    "RegulatorRegulon_Spearman_R",
                    "RegulatorRegulon_Spearman_p-value",
                    "Regulon_stratification_t-statistic",
                    "-log10(p)_Regulon_stratification",
                    "Fraction_of_edges_correctly_aligned"
                ]

                results_.index = bicluster_list

                result_dfs.append(results_)

            elif mean_significance < -np.log10(significance_threshold):
                continue

        if len(result_dfs) == 0:
            continue
        elif len(result_dfs) == 1:
            causal_output = result_dfs[0]
        if len(result_dfs) > 1:
            causal_output = pd.concat(result_dfs,axis=0)
            
        output_file = ("").join([mutation_name,"_causal_results",".csv"])
        causal_output.to_csv(os.path.join(causal_path,output_file))


    t2 = time.time()

    print('completed causal analysis in {:.2f} minutes'.format((t2-t1)/60.))
    
    return

def causalNetworkImpact(target_genes,regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,resultsDirectory,minRegulons=1,significance_threshold=0.05,causalFolder="causal_results",return_df=False,tag=None):
       
    # create results directory
    if not os.path.isdir(resultsDirectory):
        os.mkdir(resultsDirectory)    
    causal_path = os.path.join(resultsDirectory,causalFolder)
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    ###
    regulon_df_bcindex = regulon_matrix.copy()
    regulon_df_bcindex.index = np.array(regulon_df_bcindex["Regulon_ID"]).astype(str)

    regulon_df_gene_index = regulon_matrix.copy()
    regulon_df_gene_index.index = regulon_df_gene_index["Gene"]

    dfs = []
    ###
    for mut_ix in range(mutation_matrix.shape[0]):
        rows = []
        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))
        phenotype_2 = list(set(phenotype_2)&set(reference_matrix.columns))
        phenotype_1 = list(set(phenotype_1)&set(reference_matrix.columns))
        
        regulon_ttests = pd.DataFrame(
            np.vstack(
                stats.ttest_ind(reference_matrix.loc[:,phenotype_2],reference_matrix.loc[:,phenotype_1],equal_var=False,axis=1)
            ).T
        )

        regulon_ttests.index = reference_matrix.index
        regulon_ttests.columns = ["Regulon_t-test_t","Regulon_t-test_p"] # Table1: eigengenes ttests

        mean_ts = []
        mean_significance = []

        target_genes = list(set(target_genes)&set(expression_matrix.index))
        target_genes_in_network = list(set(target_genes)&set(regulon_df_gene_index.index))
        for regulator_ in target_genes: # analyze all target_genes in expression_matrix

            if regulator_ in target_genes_in_network:
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)
                pp = 10**(-1*mean_significance)

            else:
                xt, xp = stats.ttest_ind(expression_matrix.loc[regulator_,phenotype_2],expression_matrix.loc[regulator_,phenotype_1],equal_var=False)
                mean_ts = xt
                mean_significance = -np.log10(xp)
                pp = 10**(-1*mean_significance)
            
            if mean_significance >= -np.log10(significance_threshold):
                results = [mutation_name,regulator_,mean_ts,mean_significance,pp]
                rows.append(results)

        if len(rows) == 0:
            continue
        
        output = pd.DataFrame(np.vstack(rows))
        output.columns = ["Mutation","Regulator","t-statistic","-log10(p)","p"]
        sort_values = np.argsort(np.array(output["p"]).astype(float))
        output = output.iloc[sort_values,:]
        
        if tag is None:
            tag = "network_impact"
        filename = ("_").join([mutation_name,tag])
        output.to_csv(("").join([os.path.join(causal_path,filename),".csv"]))
        if return_df is True:
            dfs.append(output)
    if return_df is True:
        concatenate_dfs = pd.concat(dfs,axis=0)
        concatenate_dfs.index = range(concatenate_dfs.shape[0])
        return concatenate_dfs
        
    return

def viewSelectedCausalResults(causalDf,selected_mutation,minimum_fraction_correctly_aligned=0.5,correlation_pValue_cutoff=0.05,regulon_stratification_pValue=0.05):
    causalDf = causalDf[causalDf.Mutation==selected_mutation]
    causalDf = causalDf[causalDf["RegulatorRegulon_Spearman_p-value"]<=correlation_pValue_cutoff]  
    causalDf = causalDf[causalDf["Fraction_of_edges_correctly_aligned"]>=minimum_fraction_correctly_aligned]  
    if '-log10(p)_Regulon_stratification' in causalDf.columns:
        causalDf = causalDf[causalDf["-log10(p)_Regulon_stratification"]>=-np.log10(regulon_stratification_pValue)]  
    elif 'Regulon_stratification_p-value' in causalDf.columns:
        causalDf = causalDf[causalDf["Regulon_stratification_p-value"]>=-np.log10(regulon_stratification_pValue)]  

    return causalDf

def causalNetworkAnalysisTask(task):    

    start, stop = task[0]    
    regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,minRegulons,significance_threshold,causal_path = task[1]
    ###
    regulon_df_bcindex = regulon_matrix.copy()
    regulon_df_bcindex.index = np.array(regulon_df_bcindex["Regulon_ID"]).astype(str)

    regulon_df_gene_index = regulon_matrix.copy()
    regulon_df_gene_index.index = regulon_df_gene_index["Gene"]

    tf_name = []
    bc_name = []
    rs_1 = []
    ps_1 = []
    index_1 = []
    for key in list(set(regulon_df_bcindex.index)):
        e_gene = reference_matrix.loc[str(key),:]
        tf = list(regulon_df_bcindex.loc[key,"Regulator"])[0]
        tf_exp = expression_matrix.loc[tf,reference_matrix.columns]
        r, p = stats.spearmanr(tf_exp, e_gene)
        tf_name.append(tf)
        bc_name.append(key)
        rs_1.append(r)
        ps_1.append(p)
        index_1.append(key)

    correlation_df_bcindex = pd.DataFrame(np.vstack([tf_name,bc_name,rs_1,ps_1]).T) # Table
    correlation_df_bcindex.columns = ["Regulator","Regulon_ID","Spearman_R","Spearman_p"]
    correlation_df_bcindex.index = np.array(index_1).astype(str)

    correlation_df_regulator_index = correlation_df_bcindex.copy() # Table
    correlation_df_regulator_index.index = correlation_df_regulator_index["Regulator"]

    ###
    for mut_ix in range(start,stop):

        mutation_name = mutation_matrix.index[mut_ix]

        phenotype_2 = mutation_matrix.columns[mutation_matrix.loc[mutation_name,:]==1]
        phenotype_1 = list(set(mutation_matrix.columns)-set(phenotype_2))

        regulon_ttests = pd.DataFrame(
            np.vstack(
                stats.ttest_ind(reference_matrix.loc[:,phenotype_2],reference_matrix.loc[:,phenotype_1],equal_var=False,axis=1)
            ).T
        )

        regulon_ttests.index = reference_matrix.index
        regulon_ttests.columns = ["Regulon_t-test_t","Regulon_t-test_p"] # Table1: eigengenes ttests

        result_dfs = []
        mean_ts = []
        mean_significance = []

        upstream_regulators = list(set(regulon_matrix.Regulator)-set(regulon_df_gene_index.index))
        for regulator_ in list(set(regulon_matrix.Regulator)&set(regulon_df_gene_index.index)): # analyze all regulators in regulon_matrix

            if regulator_ not in upstream_regulators:
                tmp = regulon_df_gene_index.loc[regulator_,"Regulon_ID"]
                if type(tmp) is not pd.core.series.Series:
                    regulons_ = [str(tmp)]
                elif type(tmp) is pd.core.series.Series:
                    regulons_ = list(np.array(tmp).astype(str))

                neglogps = []
                ts = []

                for regulon_ in regulons_:
                    t, p = list(regulon_ttests.loc[regulon_,:])
                    tmp_neglogp = -np.log10(p)
                    neglogps.append(tmp_neglogp)
                    ts.append(t)

                mean_ts = np.mean(ts)
                mean_significance = np.mean(neglogps)

            else:
                xt, xp = stats.ttest_ind(expression_matrix.loc[regulator_,phenotype_2],expression_matrix.loc[regulator_,phenotype_1],equal_var=False)
                mean_ts = xt
                mean_significance = -np.log10(xp)

            if mean_significance >= -np.log10(significance_threshold):
                downstream_tmp = correlation_df_regulator_index.loc[regulator_,"Regulon_ID"]
                if type(downstream_tmp) is not pd.core.series.Series:
                    downstream_regulons = [str(downstream_tmp)]
                elif type(downstream_tmp) is pd.core.series.Series:
                    downstream_regulons = list(np.array(downstream_tmp).astype(str))

                if len(downstream_regulons)<minRegulons:
                    continue

                d_neglogps = []
                d_ts = []
                for downstream_regulon_ in downstream_regulons:
                    dt, dp = list(regulon_ttests.loc[downstream_regulon_,:])
                    tmp_neglogp = -np.log10(dp)
                    d_neglogps.append(tmp_neglogp)
                    d_ts.append(dt)

                d_neglogps = np.array(d_neglogps)
                d_ts = np.array(d_ts)

                mask = np.where(d_neglogps >= -np.log10(significance_threshold))[0]
                if len(mask) == 0:
                    continue

                significant_regulons = np.array(downstream_regulons)[mask]
                significant_regulon_ts = d_ts[mask]
                significant_regulon_ps = d_neglogps[mask]

                significant_Rs = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_R"]).astype(float)
                significant_ps = np.array(correlation_df_bcindex.loc[significant_regulons,"Spearman_p"]).astype(float)

                assignment_values = mean_ts*significant_Rs*significant_regulon_ts
                #assignments = assignment_values/np.abs(assignment_values)
                alignment_mask = np.where(assignment_values>0)[0]

                if len(alignment_mask) == 0:
                    continue

                mutation_list = np.array([mutation_name for i in range(len(alignment_mask))])
                regulator_list = np.array([regulator_ for i in range(len(alignment_mask))])
                bicluster_list = significant_regulons[alignment_mask]
                mutation_regulator_edge_direction = np.array([mean_ts/np.abs(mean_ts) for i in range(len(alignment_mask))])
                mutation_regulator_edge_ps = np.array([mean_significance for i in range(len(alignment_mask))])
                regulator_bicluster_rs = significant_Rs[alignment_mask]
                regulator_bicluster_ps = significant_ps[alignment_mask]    
                bicluster_ts = significant_regulon_ts[alignment_mask]
                bicluster_ps = significant_regulon_ps[alignment_mask]  
                fraction_aligned = np.array([len(alignment_mask)/float(len(mask)) for i in range(len(alignment_mask))])


                results_ = pd.DataFrame(
                    np.vstack(
                        [
                            mutation_list,
                            regulator_list,
                            bicluster_list,
                            mutation_regulator_edge_direction,
                            mutation_regulator_edge_ps,
                            regulator_bicluster_rs,
                            regulator_bicluster_ps,
                            bicluster_ts,
                            bicluster_ps,
                            fraction_aligned
                        ]
                    ).T
                )

                results_.columns = [
                    "Mutation",
                    "Regulator",
                    "Regulon",
                    "MutationRegulatorEdge",
                    "-log10(p)_MutationRegulatorEdge",
                    "RegulatorRegulon_Spearman_R",
                    "RegulatorRegulon_Spearman_p-value",
                    "Regulon_stratification_t-statistic",
                    "Regulon_stratification_p-value",
                    "Fraction_of_edges_correctly_aligned"
                ]

                results_.index = bicluster_list

                result_dfs.append(results_)

            elif mean_significance < -np.log10(significance_threshold):
                continue

        causal_output = pd.concat(result_dfs,axis=0)
        output_file = ("").join([mutation_name,"_causal_results",".csv"])
        causal_output.to_csv(os.path.join(causal_path,output_file))

    return

def parallelCausalNetworkAnalysis(regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,causal_path,numCores,minRegulons=1,significance_threshold=0.05):

    # create results directory
    if not os.path.isdir(causal_path):
        os.mkdir(causal_path)

    t1 = time.time()
    
    taskSplit = splitForMultiprocessing(mutation_matrix.index,numCores)
    taskData = (regulon_matrix,expression_matrix,reference_matrix,mutation_matrix,minRegulons,significance_threshold,causal_path)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    multiprocess(causalNetworkAnalysisTask,tasks)

    t2 = time.time()
    print('completed causal analysis in {:.2f} minutes'.format((t2-t1)/60.))

    return 

def wiringDiagram(causal_results,regulonModules,coherent_samples_matrix,include_genes=False,savefile=None):
    cytoscape_output = []
    for regulon in list(set(causal_results.index)):

        genes = regulonModules[regulon]
        samples = coherent_samples_matrix.columns[coherent_samples_matrix.loc[int(regulon),:]==1]
        condensed_genes = (";").join(genes)
        condensed_samples = (";").join(samples)
        causal_info = causal_results.loc[regulon,:]
        if type(causal_info) is pd.core.frame.DataFrame:
            for i in range(causal_info.shape[0]):
                mutation = causal_info.iloc[i,0]
                reg = causal_info.iloc[i,1]
                tmp_edge1 = causal_info.iloc[i,3]
                if tmp_edge1 >0:
                    edge1 = "up-regulates"
                elif tmp_edge1 <0:
                    edge1 = "down-regulates"    
                tmp_edge2 = causal_info.iloc[i,5]
                if tmp_edge2 >0:
                    edge2 = "activates"
                elif tmp_edge2 <0:
                    edge2 = "represses"  

                if include_genes is True:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon,condensed_genes,condensed_samples])
                elif include_genes is False:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon])

        elif type(causal_info) is pd.core.series.Series:
            for i in range(causal_info.shape[0]):
                mutation = causal_info[0]
                reg = causal_info[1]
                tmp_edge1 = causal_info[3]
                if tmp_edge1 >0:
                    edge1 = "up-regulates"
                elif tmp_edge1 <0:
                    edge1 = "down-regulates"    
                tmp_edge2 = causal_info[5]
                if tmp_edge2 >0:
                    edge2 = "activates"
                elif tmp_edge2 <0:
                    edge2 = "represses"  

                if include_genes is True:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon,condensed_genes,condensed_samples])
                elif include_genes is False:
                    cytoscape_output.append([mutation,edge1,reg,edge2,regulon])

    cytoscapeDf = pd.DataFrame(np.vstack(cytoscape_output))

    if include_genes is True:
        cytoscapeDf.columns = ["mutation","mutation-regulator_edge","regulator","regulator-regulon_edge","regulon","genes","samples"]
    elif include_genes is False:
        cytoscapeDf.columns = ["mutation","mutation-regulator_edge","regulator","regulator-regulon_edge","regulon"]

    sort_by_regulon = np.argsort(np.array(cytoscapeDf["regulon"]).astype(int))
    cytoscapeDf = cytoscapeDf.iloc[sort_by_regulon,:]
    cytoscapeDf.index = cytoscapeDf["regulon"]
    rename = [("-").join(["R",name]) for name in cytoscapeDf.index]
    cytoscapeDf.loc[:,"regulon"] = rename
    if savefile is not None:
        cytoscapeDf.to_csv(savefile)

    return cytoscapeDf

def biclusterTfIncidence(mechanisticOutput,regulons=None):

    import pandas as pd
    import numpy as np
    
    if regulons is not None:
    
        allTfs = regulons.keys()
        
        tfCount = []
        ct=0
        for tf in regulons.keys():
            tfCount.append([])
            for key in regulons[tf].keys():
                tfCount[-1].append(str(ct))
                ct+=1
        
        allBcs = np.hstack(tfCount)
        
        bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
        bcTfIncidence.index = allBcs
        bcTfIncidence.columns = allTfs
        
        for i in range(len(allTfs)):
            tf = allTfs[i]
            bcs = tfCount[i]
            bcTfIncidence.loc[bcs,tf] = 1
            
        index = np.sort(np.array(bcTfIncidence.index).astype(int))
        if type(bcTfIncidence.index[0]) is str:
            bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
        elif type(bcTfIncidence.index[0]) is unicode:
            bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
        else:
            bcTfIncidence = bcTfIncidence.loc[index,:]  
        
        return bcTfIncidence

    allBcs = mechanisticOutput.keys()
    allTfs = list(set(np.hstack([mechanisticOutput[i].keys() for i in mechanisticOutput.keys()])))
    
    bcTfIncidence = pd.DataFrame(np.zeros((len(allBcs),len(allTfs))))
    bcTfIncidence.index = allBcs
    bcTfIncidence.columns = allTfs
    
    for bc in mechanisticOutput.keys():
        bcTfs = mechanisticOutput[bc].keys()
        bcTfIncidence.loc[bc,bcTfs] = 1
        
    index = np.sort(np.array(bcTfIncidence.index).astype(int))
    if type(bcTfIncidence.index[0]) is str:
        bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
    elif type(bcTfIncidence.index[0]) is unicode:
        bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
    else:
        bcTfIncidence = bcTfIncidence.loc[index,:]
    
    return bcTfIncidence

def tfExpression(expressionData,motifPath=os.path.join("..","data","all_tfs_to_motifs.pkl")):

    allTfsToMotifs = read_pkl(motifPath)
    tfs = list(set(allTfsToMotifs.keys())&set(expressionData.index))
    tfExp = expressionData.loc[tfs,:]
    return tfExp

def filterMutations(mutationFile, minNumMutations=None):

    mutations = pd.read_csv(mutationFile, index_col=0, header=0)
    if minNumMutations is None:
        minNumMutations = min(np.ceil(mutations.shape[1]*0.01),4)
    freqMuts = list(mutations.index[np.where(np.sum(mutations,axis=1)>=minNumMutations)[0]])
    filteredMutations = mutations.loc[freqMuts,:]
    
    return filteredMutations

def mutationMatrix(mutationFiles,minNumMutations=None):
    matrices = []
    for mutationFile in mutationFiles:
        matrix = filterMutations(mutationFile, minNumMutations=minNumMutations)
        matrices.append(matrix)
    filteredMutations = pd.concat(matrices,axis=0)

    return filteredMutations

def getMutations(mutationString,mutationMatrix):   
    return mutationMatrix.columns[np.where(mutationMatrix.loc[mutationString,:]>0)[0]]

def mutationRegulatorStratification(mutationDf,tfDf,threshold=0.05,dictionary_=False):
    
    incidence = pd.DataFrame(np.zeros((tfDf.shape[0],mutationDf.shape[0])))
    incidence.index = tfDf.index
    incidence.columns = mutationDf.index
    
    stratification = {}
    tfCols = set(tfDf.columns)
    mutCols = set(mutationDf.columns)
    for mutation in mutationDf.index:
        mut = getMutations(mutation,mutationDf)
        wt = list(mutCols-set(mut))
        mut = list(set(mut)&tfCols)
        wt = list(set(wt)&tfCols)
        tmpMut = tfDf.loc[:,mut]
        tmpWt = tfDf.loc[:,wt]
        ttest = stats.ttest_ind(tmpMut,tmpWt,axis=1,equal_var=False)
        significant = np.where(ttest[1]<=threshold)[0]
        hits = list(tfDf.index[significant])
        if len(hits) > 0:
            incidence.loc[hits,mutation] = 1
            if dictionary_ is not False:
                stratification[mutation] = {}
                for i in range(len(hits)):
                    stratification[mutation][hits[i]] = [ttest[0][significant[i]],ttest[1][significant[i]]]
    
    if dictionary_ is not False:
        return incidence, stratification
    return incidence

def generateEpigeneticMatrix(epigeneticFilename,expressionData,cutoff_pecentile=80,saveFile="epigeneticMatrix.csv"):
    epigenetic_regulators = pd.read_csv(os.path.join(os.path.split(os.getcwd())[0],"data",epigeneticFilename),sep="\t",header=None)
    epigenetic_regulators_list = list(epigenetic_regulators.iloc[:,0])
    epigenetic = list(set(epigenetic_regulators_list)&set(expressionData.index))
    epigenetic_expression = expressionData.loc[epigenetic,:]
    percentiles80 = np.percentile(epigenetic_expression,cutoff_pecentile,axis=1)
    epigenetic_cutoffs = [max(percentiles80[i],0) for i in range(len(percentiles80))]

    epigenetic_matrix = pd.DataFrame(np.zeros((len(epigenetic),expressionData.shape[1])))
    epigenetic_matrix.columns = expressionData.columns
    epigenetic_matrix.index = epigenetic

    for i in range(epigenetic_matrix.shape[0]):
        epi = epigenetic_matrix.index[i]
        hits = epigenetic_matrix.columns[np.where(expressionData.loc[epi,:]>=epigenetic_cutoffs[i])[0]]
        epigenetic_matrix.loc[epi,hits] = 1

    if saveFile is not None:
        epigenetic_matrix.to_csv(os.path.join(os.path.split(os.getcwd())[0],"data",saveFile))

    return epigenetic_matrix

def generateCausalInputs(expressionData,
                         mechanisticOutput,
                         coexpressionModules,
                         saveFolder,
                         dataFolder,
                         mutationFile="filteredMutationsIA12.csv",
                         regulon_dict=None):
    bcTfIncidence = biclusterTfIncidence(mechanisticOutput,regulons=regulon_dict)
    bcTfIncidence.to_csv(os.path.join(saveFolder,"bcTfIncidence.csv"))

    #eigengenes
    eigengenes = principalDf(coexpressionModules,expressionData,subkey=None,regulons=regulon_dict,minNumberGenes=1)
    eigengenes = eigengenes.T
    index = np.sort(np.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))
    
    #tfExpression
    tfExp = tfExpression(expressionData,
                         motifPath=os.path.join(dataFolder, "all_tfs_to_motifs.pkl"))
    tfExp.to_csv(os.path.join(saveFolder,"tfExpression.csv"))
    
    #filteredMutations:
    filteredMutations = filterMutations(mutationFile)
    filteredMutations.to_csv(os.path.join(saveFolder,"filteredMutations.csv"))    
    
    #regStratAll
    tfStratMutations = mutationRegulatorStratification(filteredMutations,tfDf=tfExp,threshold=0.01)                
    keepers = list(set(np.arange(tfStratMutations.shape[1]))-set(np.where(np.sum(tfStratMutations,axis=0)==0)[0]))
    tfStratMutations = tfStratMutations.iloc[:,keepers]    
    tfStratMutations.to_csv(os.path.join(saveFolder,"regStratAll.csv"))
    
    return

def processCausalResults(causalPath=os.path.join("..","results","causal"),causalDictionary=False):

    causalFiles = []           
    for root, dirs, files in os.walk(causalPath, topdown=True):
       for name in files:
          if name.split(".")[-1] == 'DS_Store':
              continue
          causalFiles.append(os.path.join(root, name))
          
    if causalDictionary is False:
        causalDictionary = {}
    for csv in causalFiles:
        tmpcsv = pd.read_csv(csv,index_col=False,header=None)
        for i in range(1,tmpcsv.shape[0]):
            score = float(tmpcsv.iloc[i,-2])
            if score <1:
                break
            bicluster = int(tmpcsv.iloc[i,-3].split(":")[-1].split("_")[-1])
            if bicluster not in causalDictionary.keys():
                causalDictionary[bicluster] = {}
            regulator = tmpcsv.iloc[i,-5].split(":")[-1]
            if regulator not in causalDictionary[bicluster].keys():
                causalDictionary[bicluster][regulator] = []
            mutation = tmpcsv.iloc[i,1].split(":")[-1]
            if mutation not in causalDictionary[bicluster][regulator]:
                causalDictionary[bicluster][regulator].append(mutation)
    
    return causalDictionary

def analyzeCausalResults(task):

    start, stop = task[0]
    preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes = task[1]
    postProcessed = {}
    if mechanisticOutput is not None:
        mechOutKeyType = type(mechanisticOutput.keys()[0])
    allPatients = set(filteredMutations.columns)
    keys = preProcessedCausalResults.keys()[start:stop]
    ct=-1
    for bc in keys:
        ct+=1
        if ct%10 == 0:
            print(ct)
        postProcessed[bc] = {}
        for tf in preProcessedCausalResults[bc].keys():
            for mutation in preProcessedCausalResults[bc][tf]:
                mut = getMutations(mutation,filteredMutations)
                wt = list(allPatients-set(mut))
                mutTfs = tfExp.loc[tf,mut][tfExp.loc[tf,mut]>-4.01]
                if len(mutTfs) <=1:
                    mutRegT = 0
                    mutRegP = 1
                elif len(mutTfs) >1:
                    wtTfs = tfExp.loc[tf,wt][tfExp.loc[tf,wt]>-4.01]
                    mutRegT, mutRegP = stats.ttest_ind(list(mutTfs),list(wtTfs),equal_var=False)
                mutBc = eigengenes.loc[bc,mut][eigengenes.loc[bc,mut]>-4.01]
                if len(mutBc) <=1:
                    mutBcT = 0
                    mutBcP = 1
                    mutCorrR = 0
                    mutCorrP = 1
                elif len(mutBc) >1:
                    wtBc = eigengenes.loc[bc,wt][eigengenes.loc[bc,wt]>-4.01]
                    mutBcT, mutBcP = stats.ttest_ind(list(mutBc),list(wtBc),equal_var=False)
                    if len(mutTfs) <=2:
                        mutCorrR = 0
                        mutCorrP = 1
                    elif len(mutTfs) >2:
                        nonzeroPatients = list(set(np.array(mut)[tfExp.loc[tf,mut]>-4.01])&set(np.array(mut)[eigengenes.loc[bc,mut]>-4.01]))
                        mutCorrR, mutCorrP = stats.pearsonr(list(tfExp.loc[tf,nonzeroPatients]),list(eigengenes.loc[bc,nonzeroPatients]))
                signMutTf = 1
                if mutRegT < 0:
                    signMutTf = -1
                elif mutRegT == 0:
                    signMutTf = 0
                signTfBc = 1
                if mutCorrR < 0:
                    signTfBc = -1
                elif mutCorrR == 0:
                    signTfBc = 0
                if mechanisticOutput is not None:
                    if mechOutKeyType is int:              
                        phyper = mechanisticOutput[bc][tf][0]
                    elif mechOutKeyType is not int:
                        phyper = mechanisticOutput[str(bc)][tf][0]
                elif mechanisticOutput is None:
                    phyper = 1e-10
                pMutRegBc = 10**-((-np.log10(mutRegP)-np.log10(mutBcP)-np.log10(mutCorrP)-np.log10(phyper))/4.)
                pWeightedTfBc = 10**-((-np.log10(mutCorrP)-np.log10(phyper))/2.)
                mutFrequency = len(mut)/float(filteredMutations.shape[1])
                postProcessed[bc][tf] = {}
                postProcessed[bc][tf]["regBcWeightedPValue"] = pWeightedTfBc
                postProcessed[bc][tf]["edgeRegBc"] = signTfBc
                postProcessed[bc][tf]["regBcHyperPValue"] = phyper
                if "mutations" not in postProcessed[bc][tf].keys():
                    postProcessed[bc][tf]["mutations"] = {}
                postProcessed[bc][tf]["mutations"][mutation] = {}
                postProcessed[bc][tf]["mutations"][mutation]["mutationFrequency"] = mutFrequency
                postProcessed[bc][tf]["mutations"][mutation]["mutRegBcWeightedPValue"] = pMutRegBc
                postProcessed[bc][tf]["mutations"][mutation]["edgeMutReg"] = signMutTf
                postProcessed[bc][tf]["mutations"][mutation]["mutRegPValue"] = mutRegP
                postProcessed[bc][tf]["mutations"][mutation]["mutBcPValue"] = mutBcP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrPValue"] = mutCorrP
                postProcessed[bc][tf]["mutations"][mutation]["regBcCorrR"] = mutCorrR
         
    return postProcessed

def postProcessCausalResults(preProcessedCausalResults,filteredMutations,tfExp,eigengenes,mechanisticOutput=None,numCores=5):
    
    taskSplit = splitForMultiprocessing(preProcessedCausalResults.keys(),numCores)
    taskData = (preProcessedCausalResults,mechanisticOutput,filteredMutations,tfExp,eigengenes)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    Output = multiprocess(analyzeCausalResults,tasks)
    postProcessedAnalysis = condenseOutput(Output)

    return postProcessedAnalysis

def causalMechanisticNetworkDictionary(postProcessedCausalAnalysis,biclusterRegulatorPvalue=0.05,regulatorMutationPvalue=0.05,mutationFrequency = 0.025,requireCausal=False):
    
    tabulatedResults = []
    ct=-1
    for key in postProcessedCausalAnalysis.keys():
        ct+=1 
        if ct%10==0:
            print(ct)
        lines = []
        regs = postProcessedCausalAnalysis[key].keys()
        for reg in regs:
            bcid = key
            regid = reg
            bcRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['edgeRegBc'])
            bcRegEdgePValue = postProcessedCausalAnalysis[key][reg]['regBcWeightedPValue']
            bcTargetEnrichmentPValue = postProcessedCausalAnalysis[key][reg]['regBcHyperPValue']
            if bcRegEdgePValue <= biclusterRegulatorPvalue:
                if len(postProcessedCausalAnalysis[key][reg]['mutations'])>0:
                    for mut in postProcessedCausalAnalysis[key][reg]['mutations'].keys():                        
                        mutFrequency = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutationFrequency']
                        mutRegPValue = postProcessedCausalAnalysis[key][reg]['mutations'][mut]['mutRegPValue']
                        if mutFrequency >= mutationFrequency:
                            if mutRegPValue <= regulatorMutationPvalue:                                
                                mutid = mut
                                mutRegEdgeType = int(postProcessedCausalAnalysis[key][reg]['mutations'][mut]['edgeMutReg'])                             
                            elif mutRegPValue > regulatorMutationPvalue:
                                mutid = np.nan #"NA"
                                mutRegEdgeType = np.nan #"NA"
                                mutRegPValue = np.nan #"NA"
                                mutFrequency = np.nan #"NA"                                  
                        elif mutFrequency < mutationFrequency:
                            mutid = np.nan #"NA"
                            mutRegEdgeType = np.nan #"NA"
                            mutRegPValue = np.nan #"NA"
                            mutFrequency = np.nan #"NA"    
                elif len(postProcessedCausalAnalysis[key][reg]['mutations'])==0:
                    mutid = np.nan #"NA"
                    mutRegEdgeType = np.nan #"NA"
                    mutRegPValue = np.nan #"NA"
                    mutFrequency = np.nan #"NA"
            elif bcRegEdgePValue > biclusterRegulatorPvalue:
                continue 
            line = [bcid,regid,bcRegEdgeType,bcRegEdgePValue,bcTargetEnrichmentPValue,mutid,mutRegEdgeType,mutRegPValue,mutFrequency]
            lines.append(line)
        if len(lines) == 0:
            continue
        stack = np.vstack(lines)
        df = pd.DataFrame(stack)
        df.columns = ["Cluster","Regulator","RegulatorToClusterEdge","RegulatorToClusterPValue","RegulatorBindingSiteEnrichment","Mutation","MutationToRegulatorEdge","MutationToRegulatorPValue","FrequencyOfMutation"]
        tabulatedResults.append(df)
        
    resultsDf = pd.concat(tabulatedResults,axis=0)
    resultsDf = resultsDf[resultsDf["RegulatorToClusterEdge"]!='0']
    resultsDf.index = np.arange(resultsDf.shape[0])

    if requireCausal is True:
        resultsDf = resultsDf[resultsDf["Mutation"]!="nan"]
        
    return resultsDf

def clusterInformation(causalMechanisticNetwork,key):
    return causalMechanisticNetwork[causalMechanisticNetwork["Cluster"]==key]
  
def showCluster(expressionData,coexpressionModules,key):
    plt.figure(figsize=(10,10))
    plt.imshow(expressionData.loc[coexpressionModules[key],:],vmin=-1,vmax=1)
    plt.title("Cluster Expression",FontSize=16)
    plt.xlabel("Patients",FontSize=14)
    plt.ylabel("Genes",FontSize=14)     
    return     

# =============================================================================
# Functions used for logic-based predictor
# =============================================================================

def precision(matrix, labels):
    
    vector = labels.iloc[:,0]
    vectorMasked = (matrix*vector).T
    TP = np.array(np.sum(vectorMasked,axis=0)).astype(float)
    FP = np.array(np.sum(matrix,axis=1)-TP).astype(float)
    prec = TP/(TP+FP)
    prec[np.where(TP<=5)[0]]=0
    return prec

def labelVector(hr,lr):
    labels = np.concatenate([np.ones(len(hr)),np.zeros(len(lr))]).astype(int)
    labelsDf = pd.DataFrame(labels)
    labelsDf.index = np.concatenate([hr,lr])
    labelsDf.columns = ["label"]
    return labelsDf

def predictRisk(expressionDf,regulonModules,model_filename):
    import pickle
    
    expressionDf, _ = identifierConversion(expressionData=expressionDf)
    expressionDf = zscore(expressionDf)
    bkgdDf = backgroundDf(expressionDf)
    overExpressedMembers = biclusterMembershipDictionary(regulonModules,bkgdDf,label=2,p=0.1)
    overExpressedMembersMatrix = membershipToIncidence(overExpressedMembers,expressionDf)

    labels = overExpressedMembersMatrix.columns
    predictor_formatted = np.array(overExpressedMembersMatrix).T

    loaded_model = pickle.load(open(model_filename, 'rb'))
    prediction = loaded_model.predict(predictor_formatted)

    hr_dt = labels[prediction.astype(bool)]
    lr_dt = labels[(1-prediction).astype(bool)]

    return hr_dt, lr_dt

def gene_conversion(gene_list,input_type="ensembl.gene", output_type="symbol",list_symbols=None):

    if input_type =="ensembl":
        input_type = "ensembl.gene"
    if output_type =="ensembl":
        output_type = "ensembl.gene"
    #kwargs = symbol,ensembl, entrezgene
    import mygene #requires pip install beyond anaconda
    mg = mygene.MyGeneInfo()
    gene_query = mg.querymany(gene_list, scopes=input_type, fields=[output_type], species="human", as_dataframe=True)

    if list_symbols is not None:
        if output_type == "ensembl.gene":
            list_ = list(gene_query[output_type])
            #print(list_)
            output = []
            for dict_ in list_:
                if type(dict_) is dict:
                    output.append(dict_["gene"])
                else:
                    for subdict in dict_:
                        output.append(subdict["gene"])
        else:
            output = list(gene_query[output_type])
        return output
    
    dict_ = {}
    try:
        trimmed_df = gene_query[gene_query.iloc[:,2]>0]
        for i in range(0,trimmed_df.shape[0]):
            tmp = trimmed_df.index[i]
            tmp1 = trimmed_df.iloc[i,2]
            dict_[tmp] = []
            lencheck = len(tmp1)
            if lencheck == 1:
                dict_[tmp].append(str(tmp1).split("'")[3])
            if lencheck > 1:
                for j in range(0,len(tmp1)):            
                    dict_[tmp].append(str(tmp1[j]).split("'")[3])
    except:
        return gene_query
    
    return dict_

def swarmplot(samples,survival,savefile,ylabel="Relative risk",labels = None):
    
    import seaborn as sns
    allSamples = samples
    try:
        allSamples = np.hstack(samples)
    except:
        pass
    
    survival_samples = list(set(survival.index)&set(allSamples))
    srv = survival.loc[survival_samples,:]
    guan_srv = pd.DataFrame(srv.loc[:,"GuanScore"])
    guan_srv.columns = ["value"]
    guan_srv_group = pd.DataFrame(-np.ones(guan_srv.shape[0]))
    guan_srv_group.index = guan_srv.index
    guan_srv_group.columns = ["group"]
    guan_srv_df = pd.concat([guan_srv,guan_srv_group],axis=1)
    
    if len(samples[0][0]) > 1:
        groups = samples
    
    elif len(samples[0][0]) == 1:
        groups = []
        groups.append(samples)
    
    if labels is None:
        labels = range(len(groups))
        
    label_dfs = []
    for i in range(len(groups)):
        group = list(set(srv.index)&set(groups[i]))
        if len(group)>=1:
            label = labels[i]
            tmp_df = guan_srv_df.loc[group,:]
            tmp_df.loc[:,"group"] = label
            label_dfs.append(tmp_df)
    if len(label_dfs)>1:
        guan_srv_df = pd.concat(label_dfs,axis=0)
    elif len(label_dfs)==1:
        guan_srv_df = label_dfs[0]
    
    plt.figure(figsize=(12,8))
    ax = sns.boxplot(x='group', y='value', data=guan_srv_df)
    for patch in ax.artists:
        patch.set_edgecolor('black')
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, 0.8))

    sns.swarmplot(x='group', y='value',data=guan_srv_df,size=7, color=[0.15,0.15,0.15],edgecolor="black")

    plt.ylabel(ylabel,FontSize=24)
    plt.xlabel("",FontSize=0)
    plt.ylim(-0.05,1.05)
    plt.xticks(FontSize=18)
    plt.yticks(FontSize=18)
    plt.savefig(savefile,bbox_inches="tight")

    
    return guan_srv_df

def generatePredictionMatrix(srv,mtrx,high_risk_cutoff = 0.2):
    
    highRiskSamples = list(srv.index[0:int(high_risk_cutoff*srv.shape[0])])
    lowRiskSamples = list(srv.index[int(high_risk_cutoff*srv.shape[0]):])

    hrFlag = pd.DataFrame(np.ones((len(highRiskSamples),1)).astype(int))
    hrFlag.index = highRiskSamples
    hrFlag.columns = ["HR_FLAG"]

    lrFlag = pd.DataFrame(np.zeros((len(lowRiskSamples),1)).astype(int))
    lrFlag.index = lowRiskSamples
    lrFlag.columns = ["HR_FLAG"]

    hrMatrix = pd.concat([mtrx.loc[:,highRiskSamples].T,hrFlag],axis=1)
    hrMatrix.columns = np.array(hrMatrix.columns).astype(str)
    lrMatrix = pd.concat([mtrx.loc[:,lowRiskSamples].T,lrFlag],axis=1)
    lrMatrix.columns = np.array(lrMatrix.columns).astype(str)
    #predictionMatrix = pd.concat([hrMatrix,lrMatrix],axis=0)
    
    return hrMatrix, lrMatrix

def plotRiskStratification(lbls,mtrx,srv,survival_tag,resultsDirectory=None):
    import warnings
    warnings.filterwarnings("ignore")
    
    hr_dt = mtrx.columns[lbls.astype(bool)]
    lr_dt = mtrx.columns[(1-lbls).astype(bool)]

    kmTag = "decision_tree"
    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])

    groups = [hr_dt,lr_dt]
    labels = ["High-risk","Low-risk"]

    cox_vectors = []
    srv_set = set(srv.index)
    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(group)&srv_set)
        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
        tmp_df.index = srv.index
        tmp_df.columns = [labels[i]]
        tmp_df.loc[patients,labels[i]] = 1
        cox_vectors.append(tmp_df)

    pre_cox = pd.concat(cox_vectors,axis=1).T
    pre_cox.head(5)

    cox_dict = parallelMemberSurvivalAnalysis(membershipDf = pre_cox,numCores=1,survivalPath="",survivalData=srv)
    print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))

    if resultsDirectory is not None:
        plotName = os.path.join(resultsDirectory,kmFilename)
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)  
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

    elif resultsDirectory is None:
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))
        
    return

def iAUC(srv,mtrx,classifier,plot_all=False):
    from sklearn import metrics
    
    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
    predicted_probabilities_df.index = mtrx.columns
    predicted_probabilities_df.columns = ["probability_high_risk"]
    
    srv_observed = srv[srv.iloc[:,1]==1]
    srv_unobserved = srv[srv.iloc[:,1]==0]

    aucs = []
    cutoffs = []
    tpr_list = []
    fpr_list = []
    for cutoff in 30.5*np.arange(12,25,2):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
        srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
        if srv_extended.shape[0] > 0:
            srv_total = pd.concat([srv_observed,srv_extended],axis=0)
        elif srv_extended.shape[0] == 0:
            srv_total = srv_observed.copy()
        pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
        true_hr = []
        if len(pass_index) > 0:
            true_hr = srv_total.index[pass_index]
        true_lr = list(set(srv_total.index)-set(true_hr))

        tpr = []
        fpr = []
        for threshold in np.arange(0,1.02,0.01):
            model_pp = predicted_probabilities_df.copy()
            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
            predicted_lr = list(set(model_pp.index)-set(predicted_hr))

            tp = set(true_hr)&set(predicted_hr)
            allpos = set(true_hr)
            tn = set(true_lr)&set(predicted_lr)
            allneg = set(true_lr)

            if len(allpos) == 0:
                tp_rate = 0
            elif len(allpos) > 0:
                tp_rate = len(tp)/float(len(allpos))

            if len(allneg) == 0:
                tn_rate = 0
            elif len(allneg) > 0:        
                tn_rate = len(tn)/float(len(allneg))

            tpr.append(tp_rate)
            fpr.append(1-tn_rate)
        
        if plot_all is True:
            plt.figure()
            plt.plot(fpr,tpr)  
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--") 
            plt.ylim(-0.05,1.05)
            plt.xlim(-0.05,1.05)
            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))

        area = metrics.auc(fpr,tpr)
        aucs.append(area)
        cutoffs.append(cutoff)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    integrated_auc = np.mean(aucs)

    print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))
    
    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
    fpr_means = np.mean(np.vstack(fpr_list),axis=0)
    
    plt.figure()
    plt.plot(fpr_means,tpr_means+tpr_stds,'-',color="blue",LineWidth=1) 
    plt.plot(fpr_means,tpr_means-tpr_stds,'-',color="blue",LineWidth=1)
    plt.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color="blue",alpha=0.2)
    plt.plot(fpr_means,tpr_means,'-k',LineWidth=1.5)
    plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--r") 
    plt.ylim(-0.05,1.05)
    plt.xlim(-0.05,1.05)
    plt.title('Integrated AUC = {:.2f}'.format(integrated_auc))
    plt.ylabel('Sensitivity',FontSize=14)
    plt.xlabel('1-Specificity',FontSize=14)

    return aucs, cutoffs, tpr_list, fpr_list

#def riskStratification(lbls,mtrx,srv,survival_tag,classifier,resultsDirectory=None,plot_all=False):
#    import warnings
#    warnings.filterwarnings("ignore")
#    
#    hr_dt = mtrx.columns[lbls.astype(bool)]
#    lr_dt = mtrx.columns[(1-lbls).astype(bool)]
#
#    kmTag = "decision_tree"
#    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])
#
#    groups = [hr_dt,lr_dt]
#    labels = ["High-risk","Low-risk"]
#
#    cox_vectors = []
#    srv_set = set(srv.index)
#    for i in range(len(groups)):
#        group = groups[i]
#        patients = list(set(group)&srv_set)
#        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
#        tmp_df.index = srv.index
#        tmp_df.columns = [labels[i]]
#        tmp_df.loc[patients,labels[i]] = 1
#        cox_vectors.append(tmp_df)
#
#    pre_cox = pd.concat(cox_vectors,axis=1).T
#    pre_cox.head(5)
#
#    cox_dict = parallelMemberSurvivalAnalysis(membershipDf = pre_cox,numCores=1,survivalPath="",survivalData=srv)
#    print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))
#
#    if resultsDirectory is not None:
#        plotName = os.path.join(resultsDirectory,kmFilename)
#        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)  
#        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))
#
#    elif resultsDirectory is None:
#        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
#        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))
#        
#
#    from sklearn import metrics
#    
#    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
#    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
#    predicted_probabilities_df.index = mtrx.columns
#    predicted_probabilities_df.columns = ["probability_high_risk"]
#    
#    srv_observed = srv[srv.iloc[:,1]==1]
#    srv_unobserved = srv[srv.iloc[:,1]==0]
#
#    aucs = []
#    cutoffs = []
#    tpr_list = []
#    fpr_list = []
#    for cutoff in 30.5*np.arange(12,25,2):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
#        srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
#        if srv_extended.shape[0] > 0:
#            srv_total = pd.concat([srv_observed,srv_extended],axis=0)
#        elif srv_extended.shape[0] == 0:
#            srv_total = srv_observed.copy()
#        pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
#        true_hr = []
#        if len(pass_index) > 0:
#            true_hr = srv_total.index[pass_index]
#        true_lr = list(set(srv_total.index)-set(true_hr))
#
#        tpr = []
#        fpr = []
#        for threshold in np.arange(0,1.02,0.01):
#            model_pp = predicted_probabilities_df.copy()
#            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
#            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
#            predicted_lr = list(set(model_pp.index)-set(predicted_hr))
#
#            tp = set(true_hr)&set(predicted_hr)
#            allpos = set(true_hr)
#            tn = set(true_lr)&set(predicted_lr)
#            allneg = set(true_lr)
#
#            if len(allpos) == 0:
#                tp_rate = 0
#            elif len(allpos) > 0:
#                tp_rate = len(tp)/float(len(allpos))
#
#            if len(allneg) == 0:
#                tn_rate = 0
#            elif len(allneg) > 0:        
#                tn_rate = len(tn)/float(len(allneg))
#
#            tpr.append(tp_rate)
#            fpr.append(1-tn_rate)
#        
#        if plot_all is True:
#            plt.figure()
#            plt.plot(fpr,tpr)  
#            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--") 
#            plt.ylim(-0.05,1.05)
#            plt.xlim(-0.05,1.05)
#            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))
#
#        area = metrics.auc(fpr,tpr)
#        aucs.append(area)
#        cutoffs.append(cutoff)
#        tpr_list.append(tpr)
#        fpr_list.append(fpr)
#
#    integrated_auc = np.mean(aucs)
#
#    print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))
#    
#    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
#    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
#    fpr_means = np.mean(np.vstack(fpr_list),axis=0)
#    
#    plt.figure()
#    plt.plot(fpr_means,tpr_means+tpr_stds,'-',color="blue",LineWidth=1) 
#    plt.plot(fpr_means,tpr_means-tpr_stds,'-',color="blue",LineWidth=1)
#    plt.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color="blue",alpha=0.2)
#    plt.plot(fpr_means,tpr_means,'-k',LineWidth=1.5)
#    plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--r") 
#    plt.ylim(-0.05,1.05)
#    plt.xlim(-0.05,1.05)
#    plt.title('Integrated AUC = {:.2f}'.format(integrated_auc))
#    plt.ylabel('Sensitivity',FontSize=14)
#    plt.xlabel('1-Specificity',FontSize=14)
#
#    return aucs, cutoffs, tpr_list, fpr_list

def riskStratification(lbls,mtrx,srv,survival_tag,classifier,resultsDirectory=None,plot_all=False):
    import warnings
    warnings.filterwarnings("ignore")

    from sklearn import metrics
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    f.tight_layout(pad=1.08)
    f.set_figwidth(10)
    f.set_figheight(4)

    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
    predicted_probabilities_df.index = mtrx.columns
    predicted_probabilities_df.columns = ["probability_high_risk"]
    
    srv_observed = srv[srv.iloc[:,1]==1]
    srv_unobserved = srv[srv.iloc[:,1]==0]

    aucs = []
    cutoffs = []
    tpr_list = []
    fpr_list = []
    for cutoff in 30.5*np.arange(12,25,2):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
        srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
        if srv_extended.shape[0] > 0:
            srv_total = pd.concat([srv_observed,srv_extended],axis=0)
        elif srv_extended.shape[0] == 0:
            srv_total = srv_observed.copy()
        pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
        true_hr = []
        if len(pass_index) > 0:
            true_hr = srv_total.index[pass_index]
        true_lr = list(set(srv_total.index)-set(true_hr))

        tpr = []
        fpr = []
        for threshold in np.arange(0,1.02,0.01):
            model_pp = predicted_probabilities_df.copy()
            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
            predicted_lr = list(set(model_pp.index)-set(predicted_hr))

            tp = set(true_hr)&set(predicted_hr)
            allpos = set(true_hr)
            tn = set(true_lr)&set(predicted_lr)
            allneg = set(true_lr)

            if len(allpos) == 0:
                tp_rate = 0
            elif len(allpos) > 0:
                tp_rate = len(tp)/float(len(allpos))

            if len(allneg) == 0:
                tn_rate = 0
            elif len(allneg) > 0:        
                tn_rate = len(tn)/float(len(allneg))

            tpr.append(tp_rate)
            fpr.append(1-tn_rate)
        
        if plot_all is True:
            plt.figure()
            plt.plot(fpr,tpr)  
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--") 
            plt.ylim(-0.05,1.05)
            plt.xlim(-0.05,1.05)
            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))

        area = metrics.auc(fpr,tpr)
        aucs.append(area)
        cutoffs.append(cutoff)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    integrated_auc = np.mean(aucs)

    print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))
    
    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
    fpr_means = np.mean(np.vstack(fpr_list),axis=0)
    
    ax1.plot(fpr_means,tpr_means+tpr_stds,'-',color="blue",LineWidth=1) 
    ax1.plot(fpr_means,tpr_means-tpr_stds,'-',color="blue",LineWidth=1)
    ax1.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color="blue",alpha=0.2)
    ax1.plot(fpr_means,tpr_means,'-k',LineWidth=1.5)
    ax1.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--r") 
    ax1.set_ylim(-0.05,1.05)
    ax1.set_xlim(-0.05,1.05)
    ax1.set_title('Integrated AUC = {:.2f}'.format(integrated_auc))
    ax1.set_ylabel('Sensitivity',FontSize=14)
    ax1.set_xlabel('1-Specificity',FontSize=14)
    
    hr_dt = mtrx.columns[lbls.astype(bool)]
    lr_dt = mtrx.columns[(1-lbls).astype(bool)]

    kmTag = "decision_tree"
    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])

    groups = [hr_dt,lr_dt]
    labels = ["High-risk","Low-risk"]

    cox_vectors = []
    srv_set = set(srv.index)
    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(group)&srv_set)
        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
        tmp_df.index = srv.index
        tmp_df.columns = [labels[i]]
        tmp_df.loc[patients,labels[i]] = 1
        cox_vectors.append(tmp_df)

    pre_cox = pd.concat(cox_vectors,axis=1).T
    pre_cox.head(5)

    cox_dict = parallelMemberSurvivalAnalysis(membershipDf = pre_cox,numCores=1,survivalPath="",survivalData=srv)
    print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))

    if resultsDirectory is not None:
        plotName = os.path.join(resultsDirectory,kmFilename)
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)  
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

    elif resultsDirectory is None:
        kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
        plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))
        
    return aucs, cutoffs, tpr_list, fpr_list

def differentialActivity(regulon_matrix,reference_matrix,baseline_patients,relapse_patients,minRegulons = 5,useAllRegulons = False,maxRegulons = 5,highlight=None,savefile = None):

    reference_matrix.index = np.array(reference_matrix.index).astype(str)
    
    genes = []
    mean_baseline_frequency = []
    mean_relapse_frequency = []
    mean_significance = []
    skipped = []

    t1 = time.time()
    for gene in list(set(regulon_matrix["Gene"])):
        regulons_ = np.array(regulon_matrix[regulon_matrix.Gene==gene]["Regulon_ID"]).astype(str)
        if len(regulons_)<minRegulons:
            skipped.append(gene)
            continue

        baseline_freq = []
        relapse_freq = []
        neglogps = []

        for regulon_ in regulons_:

            baseline_values = reference_matrix.loc[regulon_,baseline_patients]
            relapse_values = reference_matrix.loc[regulon_,relapse_patients]

            indicator = len(set(reference_matrix.iloc[0,:]))

            if indicator>2:
                t, p = stats.ttest_ind(relapse_values,baseline_values)  

            elif indicator ==2:
                # chi square
                rpos = np.sum(relapse_values)
                if np.sum(rpos) == 0:
                    continue
                rneg = len(relapse_values)-rpos

                bpos = np.sum(baseline_values)
                if np.sum(bpos) == 0:
                    continue        
                bneg = len(baseline_values)-bpos

                obs = np.array([[rpos,rneg],[bpos,bneg]])
                chi2, p, dof, ex = stats.chi2_contingency(obs, correction=False)

            if useAllRegulons is True:             
                neglogps.append(-np.log10(p))
                baseline_freq.append(np.mean(baseline_values))        
                relapse_freq.append(np.mean(relapse_values))

            elif useAllRegulons is False: 
                if len(neglogps)<=maxRegulons:
                    neglogps.append(-np.log10(p))
                    baseline_freq.append(np.mean(baseline_values))
                    relapse_freq.append(np.mean(relapse_values))

                if len(neglogps)>maxRegulons:
                    tmp_nlp = -np.log10(p)
                    if min(neglogps) < tmp_nlp:
                        argmin = np.argmin(neglogps)
                        neglogps[argmin] = tmp_nlp
                        tmp_baseline = np.mean(baseline_values)
                        baseline_freq[argmin] = tmp_baseline
                        tmp_relapse = np.mean(relapse_values)                    
                        relapse_freq[argmin] = tmp_relapse

        mean_relapse_frequency.append(np.mean(relapse_freq))
        mean_baseline_frequency.append(np.mean(baseline_freq))    
        mean_significance.append(np.mean(neglogps))
        genes.append(gene)

    baseline_over_relapse = np.array(mean_baseline_frequency).astype(float)/np.array(mean_relapse_frequency)
    volcano_data_ = pd.DataFrame(np.vstack([mean_baseline_frequency,mean_relapse_frequency,baseline_over_relapse,mean_significance]).T)
    volcano_data_.index = genes
    volcano_data_.columns = ["phenotype1_frequency","phenotype2_frequency","phenotype1/phenotype2","-log10(p)"]
    volcano_data_.sort_values(by="-log10(p)",ascending=False,inplace=True)
    volcano_data_

    t2 = time.time()

    print('completed in {:.2f} minutes'.format((t2-t1)/60.))

    insigvoldata_patients = volcano_data_.index[volcano_data_["-log10(p)"]<=-np.log10(0.05)]
    sigvoldata_patients = volcano_data_.index[volcano_data_["-log10(p)"]>-np.log10(0.05)]

    insigvoldata = volcano_data_.loc[insigvoldata_patients,:]
    sigvoldata = volcano_data_.loc[sigvoldata_patients,:]

    try:        
        plt.scatter(sigvoldata["phenotype2_frequency"],sigvoldata["phenotype1/phenotype2"],color = [0.3,0.4,1],edgecolor = [0,0,1],alpha=0.4,s=5)
        plt.scatter(insigvoldata["phenotype2_frequency"],insigvoldata["phenotype1/phenotype2"],color = [0.5,0.5,0.5],edgecolor = [0.25,0.25,0.25],alpha=0.4,s=5)
        if highlight is not None:
            if type(highlight) is not str:
                if type(highlight) is not unicode:
                    highlight = list(set(highlight)-set(skipped))
            else:
                highlight = list(set([highlight])-set(skipped))
            
            if len(highlight)>0:
                plt.scatter(volcano_data_.loc[highlight,"phenotype2_frequency"],volcano_data_.loc[highlight,"phenotype1/phenotype2"],color = "red",edgecolor="blue",s=5)
    
        plt.ylim(-0.1,0.4+max(list(sigvoldata["phenotype1/phenotype2"])))
        plt.ylabel("Phenotype1/Phenotype2",FontSize=14)
        plt.xlabel("Phenotype2 frequency",FontSize=14)
    
        if savefile is not None:
            plt.savefig(savefile,bbox_inches="tight")
    except:
        print('Error: Analysis was successful, but could not generate plot')

    return volcano_data_
