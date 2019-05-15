import numpy,datetime,pandas,sys
from pkg_resources import Requirement, resource_filename
from collections import Counter

def correctBatchEffects(df): 
    zscoredExpression = zscore(df)
    means = []
    stds = []
    for i in range(zscoredExpression.shape[1]):
        mean = numpy.mean(zscoredExpression.iloc[:,i])
        std = numpy.std(zscoredExpression.iloc[:,i])
        means.append(mean)
        stds.append(std)
    if numpy.std(means) >= 0.15:
        zscoredExpression = preProcessTPM(df)
    return zscoredExpression

def identifierConversion(expressionData):

    print('arrived to indetifierConversion',numpy.mean(expressionData.iloc[:,0]))
    
    conversionTableFile = resource_filename(Requirement.parse("miner2"), 'miner2/data/identifier_mappings.txt')
    print('\t\t\t\t******** ',conversionTableFile)
    idMap=pandas.read_csv(conversionTableFile,sep='\t')
    
    genetypes = list(set(idMap.iloc[:,2]))
    previousIndex = numpy.array(expressionData.index).astype(str)    
    previousColumns = numpy.array(expressionData.columns).astype(str)  
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

    print()
    print(state)
    mappedGenes = bestMatch
    mappedGenes.sort() ### ALO this new line in miner2 is surprisingly important for reproducibility. Otherwise expressionData varies in last digits of floats and every thing down the road changes slightly
    subset = idMap[idMap.iloc[:,2]==gtype] 
    subset.index = subset.iloc[:,1]

    if len(bestMatch) == 0:
        print("Error: Gene identifiers not recognized")
    
    if state == "transpose":
        expressionData = expressionData.T
        
    try:
        print('before mapping',numpy.mean(expressionData.iloc[:,0]))
        print(len(mappedGenes),mappedGenes[:10])
        print('going into conversion...')
        convertedData = expressionData.loc[mappedGenes,:]
        print('after conversion',numpy.mean(convertedData.iloc[:,0]))
        print()
    except:
        convertedData = expressionData.loc[numpy.array(mappedGenes).astype(int),:]
    
    conversionTable = subset.loc[mappedGenes,:]
    conversionTable.index = conversionTable.iloc[:,0]
    conversionTable = conversionTable.iloc[:,1]
    conversionTable.columns = ["Name"]
    
    newIndex = list(subset.loc[mappedGenes,"Preferred_Name"])
    convertedData.index = newIndex

    print('after mapped genes',numpy.mean(convertedData.iloc[:,0]))
    
    duplicates = [item for item, count in Counter(newIndex).items() if count > 1]
    singles = list(set(convertedData.index)-set(duplicates))

    ### ALO these two new lines in miner2 are surprisingly important for reproducibility. Otherwise expressionData varies in last digits of floats and every thing down the road changes slightly and reproducibility gets compromised
    duplicates.sort()
    singles.sort()

    corrections = []

    print('duplicates',type(duplicates),len(duplicates),duplicates)
    print('singles',type(singles),len(singles))
    
    for duplicate in duplicates:
        dupData = convertedData.loc[duplicate,:]
        firstChoice = pandas.DataFrame(dupData.iloc[0,:]).T
        corrections.append(firstChoice)

    print('corrections',type(corrections),len(corrections),type(corrections[0]))

    print('right before corrections',numpy.mean(convertedData.iloc[:,0]))
    
    if len(corrections) > 0:
        print('there are corrections')
        correctionsDf = pandas.concat(corrections,axis=0)
        uncorrectedData = convertedData.loc[singles,:]
        print('\t before concat',numpy.mean(convertedData.iloc[:,0]))
        convertedData = pandas.concat([uncorrectedData,correctionsDf],axis=0)
        print('\t during corrections',numpy.mean(convertedData.iloc[:,0]))
        
    print('right after corrections',numpy.mean(convertedData.iloc[:,0]))
    print()
              
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t {} out of {} gene names converted to ENSEMBL IDs".format(convertedData.shape[0],expressionData.shape[0])))
    
    
    return convertedData, conversionTable

def main(filename):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data reading"))
    rawExpression = readFileToDf(filename)

    firstPatient = rawExpression.iloc[:,0]
    print('raw',type(firstPatient),len(firstPatient),numpy.mean(firstPatient))
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data recovered: {} features by {} samples".format(rawExpression.shape[0],rawExpression.shape[1])))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data transformation"))

    
    rawExpressionZeroFiltered = removeNullRows(rawExpression)
    zscoredExpression = correctBatchEffects(rawExpressionZeroFiltered)

    firstPatient = zscoredExpression.iloc[:,0]
    print('zscore',type(firstPatient),len(firstPatient),numpy.mean(firstPatient))
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t gene ID conversion"))
    expressionData, conversionTable = identifierConversion(zscoredExpression)

    firstPatient = expressionData.iloc[:,0]
    print('expressionData',type(firstPatient),len(firstPatient),numpy.mean(firstPatient))
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t working expression data: {} features by {} samples".format(expressionData.shape[0],expressionData.shape[1])))
    return expressionData, conversionTable

def readFileToDf(filename):
    extension = filename.split(".")[-1]
    if extension == "csv":
        df = pandas.read_csv(filename,index_col=0,header=0)
        shape = df.shape
        if shape[1] == 0:
            df = pandas.read_csv(filename,index_col=0,header=0,sep="\t")
    elif extension == "txt":
        df = pandas.read_csv(filename,index_col=0,header=0,sep="\t")
        shape = df.shape
        if shape[1] == 0:
            df = pandas.read_csv(filename,index_col=0,header=0)    
    return df

def removeNullRows(df):
    minimum = numpy.percentile(df,0)
    if minimum == 0:
        filteredDf = df.loc[df.sum(axis=1)>0,:]
    else:
        filteredDf = df
    return filteredDf

def zscore(expressionData):
    zero = numpy.percentile(expressionData,0)
    meanCheck = numpy.mean(expressionData[expressionData>zero].mean(axis=1,skipna=True))
    if meanCheck<0.1:
        return expressionData
    means = expressionData.mean(axis=1,skipna=True)
    stds = expressionData.std(axis=1,skipna=True)
    try:
        transform = ((expressionData.T - means)/stds).T
    except:
        passIndex = numpy.where(stds>0)[0]
        transform = ((expressionData.iloc[passIndex,:].T - means[passIndex])/stds[passIndex]).T
    print("completed z-transformation.")
    return transform
