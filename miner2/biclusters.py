import numpy
import pandas
from scipy import stats


def __assign_membership(geneset, background, p=0.05):

    cluster = background.loc[geneset, :]
    classNeg1 = len(geneset) - numpy.count_nonzero(cluster + 1, axis=0)
    class0 = len(geneset) - numpy.count_nonzero(cluster, axis=0)
    class1 = len(geneset) - numpy.count_nonzero(cluster - 1, axis=0)
    observations = zip(classNeg1, class0, class1)

    highpass = stats.binom.ppf(1 - p / 3, len(geneset), 1./3)
    classes = []
    for i in range(len(observations)):
        check = numpy.where(numpy.array(observations[i]) >= highpass)[0]
        if len(check) > 1:
            check = numpy.array([numpy.argmax(numpy.array(observations[i]))])
        classes.append(check)

    return classes


def make_membership_dictionary(revisedClusters, background, label=2, p=0.05):

    background_genes = set(background.index)
    if label == "excluded":
        members = {}
        for key in revisedClusters.keys():
            tmp_genes = list(set(revisedClusters[key]) & background_genes)
            if len(tmp_genes) > 1:
                assignments = __assign_membership(tmp_genes, background,p=p)
            else:
                assignments = [numpy.array([]) for i in range(background.shape[1])]
            nonMembers = numpy.array([i for i in range(len(assignments)) if len(assignments[i])==0])
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
            if len(tmp_genes) > 1:
                assignments = __assign_membership(tmp_genes, background, p=p)
            else:
                assignments = [numpy.array([]) for i in range(background.shape[1])]
            included = numpy.array([i for i in range(len(assignments)) if len(assignments[i])!=0])
            if len(included) == 0:
                members[key] = []
                continue

            members[key] = list(background.columns[included])
        print("done!")
        return members

    members = {}
    for key in revisedClusters.keys():
        tmp_genes = list(set(revisedClusters[key])&background_genes)
        if len(tmp_genes) > 1:
            assignments = __assign_membership(tmp_genes, background, p=p)
        else:
            assignments = [numpy.array([]) for i in range(background.shape[1])]
        overExpMembers = numpy.array([i for i in range(len(assignments)) if label in assignments[i]])
        if len(overExpMembers) ==0:
            members[key] = []
            continue
        members[key] = list(background.columns[overExpMembers])
    print("done!")
    return members


def membership_to_incidence(membershipDictionary, expressionData):

    incidence = numpy.zeros((len(membershipDictionary), expressionData.shape[1]))
    incidence = pandas.DataFrame(incidence)
    incidence.index = membershipDictionary.keys()
    incidence.columns = expressionData.columns
    for key in membershipDictionary.keys():
        samples = membershipDictionary[key]
        incidence.loc[key,samples] = 1

    try:
        orderIndex = numpy.array(incidence.index).astype(int)
        orderIndex = numpy.sort(orderIndex)
    except:
        orderIndex = incidence.index
    try:
        incidence = incidence.loc[orderIndex, :]
    except:
        incidence = incidence.loc[orderIndex.astype(str), :]

    return incidence
