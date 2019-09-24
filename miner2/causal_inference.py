import pandas
import numpy
import pickle
import os
from scipy import stats

from miner2 import subtypes


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


def __tf_expression(expressionData,motifPath=os.path.join("..","data","all_tfs_to_motifs.pkl")):
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
    tfExp = __tf_expression(expressionData,
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
