import pandas
import numpy
import pickle
import os
from scipy import stats

from miner2 import subtypes
from miner2 import util


def _bicluster_tf_incidence(mechanisticOutput, regulons=None):

    if regulons is not None:
        allTfs = regulons.keys()

        tfCount = []
        ct = 0
        for tf in regulons.keys():
            tfCount.append([])
            for key in regulons[tf].keys():
                tfCount[-1].append(str(ct))
                ct+=1

        allBcs = numpy.hstack(tfCount)

        bcTfIncidence = pandas.DataFrame(numpy.zeros((len(allBcs), len(allTfs))))
        bcTfIncidence.index = allBcs
        bcTfIncidence.columns = allTfs

        for i in range(len(allTfs)):
            tf = allTfs[i]
            bcs = tfCount[i]
            bcTfIncidence.loc[bcs,tf] = 1

        index = numpy.sort(numpy.array(bcTfIncidence.index).astype(int))
        if type(bcTfIncidence.index[0]) is str:
            bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
        elif type(bcTfIncidence.index[0]) is unicode:
            bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
        else:
            bcTfIncidence = bcTfIncidence.loc[index,:]

        return bcTfIncidence

    allBcs = mechanisticOutput.keys()
    allTfs = list(set(numpy.hstack([mechanisticOutput[i].keys() for i in mechanisticOutput.keys()])))

    bcTfIncidence = pandas.DataFrame(numpy.zeros((len(allBcs),len(allTfs))))
    bcTfIncidence.index = allBcs
    bcTfIncidence.columns = allTfs

    for bc in mechanisticOutput.keys():
        bcTfs = mechanisticOutput[bc].keys()
        bcTfIncidence.loc[bc,bcTfs] = 1

    index = numpy.sort(numpy.array(bcTfIncidence.index).astype(int))
    if type(bcTfIncidence.index[0]) is str:
        bcTfIncidence = bcTfIncidence.loc[index.astype(str),:]
    elif type(bcTfIncidence.index[0]) is unicode:
        bcTfIncidence = bcTfIncidence.loc[index.astype(unicode),:]
    else:
        bcTfIncidence = bcTfIncidence.loc[index,:]

    return bcTfIncidence


def __read_pkl(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)


def tf_expression(expressionData,motifPath=os.path.join("..","data","all_tfs_to_motifs.pkl")):
    allTfsToMotifs = __read_pkl(motifPath)
    tfs = list(set(allTfsToMotifs.keys())&set(expressionData.index))
    tfExp = expressionData.loc[tfs,:]
    return tfExp


def __filter_mutations(mutationFile, minNumMutations=None):
    mutations = pandas.read_csv(mutationFile, index_col=0, header=0)
    if minNumMutations is None:
        minNumMutations = min(numpy.ceil(mutations.shape[1] * 0.01), 4)
    freqMuts = list(mutations.index[numpy.where(numpy.sum(mutations,axis=1) >= minNumMutations)[0]])
    return mutations.loc[freqMuts,:]


def __get_mutations(mutationString,mutationMatrix):
    return mutationMatrix.columns[numpy.where(mutationMatrix.loc[mutationString,:]>0)[0]]

def __mutation_regulator_stratification(mutationDf,tfDf,threshold=0.05,dictionary_=False):
    incidence = pandas.DataFrame(numpy.zeros((tfDf.shape[0],mutationDf.shape[0])))
    incidence.index = tfDf.index
    incidence.columns = mutationDf.index

    stratification = {}
    tfCols = set(tfDf.columns)
    mutCols = set(mutationDf.columns)
    for mutation in mutationDf.index:
        mut = __get_mutations(mutation,mutationDf)
        wt = list(mutCols-set(mut))
        mut = list(set(mut)&tfCols)
        wt = list(set(wt)&tfCols)
        tmpMut = tfDf.loc[:,mut]
        tmpWt = tfDf.loc[:,wt]
        ttest = stats.ttest_ind(tmpMut,tmpWt,axis=1,equal_var=False)
        significant = numpy.where(ttest[1]<=threshold)[0]
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


def generate_inputs(expressionData,
                    mechanisticOutput,
                    coexpressionModules,
                    saveFolder,
                    dataFolder,
                    mutationFile="filteredMutationsIA12.csv",
                    regulon_dict=None):
    #bcTfIncidence
    bcTfIncidence = _bicluster_tf_incidence(mechanisticOutput,regulons=regulon_dict)
    bcTfIncidence.to_csv(os.path.join(saveFolder,"bcTfIncidence.csv"))

    #eigengenes
    eigengenes = subtypes.principal_df(coexpressionModules, expressionData, subkey=None,
                                       regulons=regulon_dict, minNumberGenes=1)
    eigengenes = eigengenes.T
    index = numpy.sort(numpy.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))

    #tfExpression
    tfExp = tf_expression(expressionData,
                            motifPath=os.path.join(dataFolder, "all_tfs_to_motifs.pkl"))
    tfExp.to_csv(os.path.join(saveFolder,"tfExpression.csv"))

    #filteredMutations:
    filteredMutations = __filter_mutations(mutationFile)
    filteredMutations.to_csv(os.path.join(saveFolder,"filteredMutations.csv"))

    #regStratAll
    tfStratMutations = __mutation_regulator_stratification(filteredMutations, tfDf=tfExp,
                                                           threshold=0.01)
    keepers = list(set(numpy.arange(tfStratMutations.shape[1])) -
                   set(numpy.where(numpy.sum(tfStratMutations, axis=0) == 0)[0]))
    tfStratMutations = tfStratMutations.iloc[:,keepers]
    tfStratMutations.to_csv(os.path.join(saveFolder,"regStratAll.csv"))


def process_causal_results(causalPath=os.path.join("..","results","causal"), causalDictionary=False):
    causalFiles = []
    for root, dirs, files in os.walk(causalPath, topdown=True):
       for name in files:
          if name.split(".")[-1] == 'DS_Store':
              continue
          causalFiles.append(os.path.join(root, name))

    if causalDictionary is False:
        causalDictionary = {}
    for csv in causalFiles:
        tmpcsv = pandas.read_csv(csv,index_col=False,header=None)
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

def mutation_matrix(mutation_files, minNumMutations=None):
    matrices = []
    for f in mutation_files:
        matrix = __filter_mutations(mutationFile=f, minNumMutations=minNumMutations)
        matrices.append(matrix)
    return pandas.concat(matrices,axis=0)


def analyze_causal_results(task):
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
                mut = __get_mutations(mutation,filteredMutations)
                wt = list(allPatients - set(mut))
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
                        nonzeroPatients = list(set(numpy.array(mut)[tfExp.loc[tf,mut]>-4.01]) &
                                               set(numpy.array(mut)[eigengenes.loc[bc,mut]>-4.01]))
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
                pMutRegBc = 10**-((-numpy.log10(mutRegP) - numpy.log10(mutBcP) -
                                   numpy.log10(mutCorrP) - numpy.log10(phyper))/4.)

                pWeightedTfBc = 10**-((-numpy.log10(mutCorrP) - numpy.log10(phyper))/2.)
                mutFrequency = len(mut) / float(filteredMutations.shape[1])
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



def post_process_causal_results(preProcessedCausalResults, filteredMutations, tfExp,
                                eigengenes, mechanisticOutput=None, numCores=5):
    taskSplit = util.split_for_multiprocessing(preProcessedCausalResults.keys(),numCores)
    taskData = (preProcessedCausalResults, mechanisticOutput, filteredMutations, tfExp, eigengenes)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    Output = util.multiprocess(analyze_causal_results, tasks)
    return util.condense_output(Output)


def causal_mechanistic_network_dictionary(postProcessedCausalAnalysis,
                                          biclusterRegulatorPvalue=0.05,
                                          regulatorMutationPvalue=0.05,
                                          mutationFrequency=0.025,
                                          requireCausal=False):
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
                                mutid = numpy.nan #"NA"
                                mutRegEdgeType = numpy.nan #"NA"
                                mutRegPValue = numpy.nan #"NA"
                                mutFrequency = numpy.nan #"NA"
                        elif mutFrequency < mutationFrequency:
                            mutid = numpy.nan #"NA"
                            mutRegEdgeType = numpy.nan #"NA"
                            mutRegPValue = numpy.nan #"NA"
                            mutFrequency = numpy.nan #"NA"
                elif len(postProcessedCausalAnalysis[key][reg]['mutations'])==0:
                    mutid = numpy.nan #"NA"
                    mutRegEdgeType = numpy.nan #"NA"
                    mutRegPValue = numpy.nan #"NA"
                    mutFrequency = numpy.nan #"NA"
            elif bcRegEdgePValue > biclusterRegulatorPvalue:
                continue
            line = [bcid,regid,bcRegEdgeType,bcRegEdgePValue,bcTargetEnrichmentPValue,mutid,mutRegEdgeType,mutRegPValue,mutFrequency]
            lines.append(line)
        if len(lines) == 0:
            continue
        stack = numpy.vstack(lines)
        df = pandas.DataFrame(stack)
        df.columns = ["Cluster","Regulator","RegulatorToClusterEdge","RegulatorToClusterPValue","RegulatorBindingSiteEnrichment","Mutation","MutationToRegulatorEdge","MutationToRegulatorPValue","FrequencyOfMutation"]
        tabulatedResults.append(df)

    resultsDf = pandas.concat(tabulatedResults,axis=0)
    resultsDf = resultsDf[resultsDf["RegulatorToClusterEdge"]!='0']
    resultsDf.index = numpy.arange(resultsDf.shape[0])

    if requireCausal is True:
        resultsDf = resultsDf[resultsDf["Mutation"]!="nan"]

    return resultsDf
