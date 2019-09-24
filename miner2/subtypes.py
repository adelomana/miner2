import numpy
import pandas
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


"""
Functions used for inferring sample subtypes
"""

def f1_decomposition(sampleMembers=None, thresholdSFM=0.333, sampleFrequencyMatrix=None):
    # thresholdSFM is the probability cutoff that makes the density of the binary
    # similarityMatrix = 0.15
    # sampleMembers is a dictionary with features as keys and members as elements
    # sampleFrequencyMatrix[i,j] gives the probability that sample j appears in a
    # cluster given that sample i appears
    if sampleFrequencyMatrix is None:
        sampleFrequencyMatrix = sample_coincidence_matrix(sampleMembers, freqThreshold=thresholdSFM,
                                                          frequencies=True)
    # similarityMatrix is defined such that similarityMatrix[i,j] = 1 iff
    # sampleFrequencyMatrix[i,j] >= thresholdSFM
    similarityMatrix = sampleFrequencyMatrix * sampleFrequencyMatrix.T
    similarityMatrix[similarityMatrix < thresholdSFM] = 0
    similarityMatrix[similarityMatrix > 0] = 1

    # remainingMembers is the set of set of unclustered members
    remainingMembers = set(similarityMatrix.index)
    # probeSample is the sample that serves as a seed to identify a cluster in a given iteration
    probeSample = numpy.argmax(similarityMatrix.sum(axis=1))
    # members are the samples that satisfy the similarity condition with the previous cluster or probeSample
    members = set(similarityMatrix.index[numpy.where(similarityMatrix[probeSample] == 1)[0]])
    # nonMembers are the remaining members not in the current cluster
    nonMembers = remainingMembers-members
    # instantiate list to collect clusters of similar members
    similarityClusters = []
    # instantiate f1 score for optimization
    f1 = 0

    for iteration in range(1500):

        predictedMembers = members
        predictedNonMembers = remainingMembers-predictedMembers

        sumSlice = numpy.sum(similarityMatrix.loc[:,list(predictedMembers)], axis=1) / float(len(predictedMembers))
        members = set(similarityMatrix.index[numpy.where(sumSlice > 0.8)[0]])

        if members == predictedMembers:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers) == 0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers, predictedNonMembers]
            probeSample = numpy.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[numpy.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        nonMembers = remainingMembers-members
        TP = len(members&predictedMembers)
        FN = len(predictedNonMembers & members)
        FP = len(predictedMembers & nonMembers)
        tmpf1 = TP/float(TP+FN+FP)

        if tmpf1 <= f1:
            similarityClusters.append(list(predictedMembers))
            if len(predictedNonMembers) == 0:
                break
            similarityMatrix = similarityMatrix.loc[predictedNonMembers,predictedNonMembers]
            probeSample = numpy.argmax(similarityMatrix.sum(axis=1))
            members = set(similarityMatrix.index[numpy.where(similarityMatrix[probeSample]==1)[0]])
            remainingMembers = predictedNonMembers
            nonMembers = remainingMembers-members
            f1 = 0
            continue

        elif tmpf1 > f1:
            f1 = tmpf1
            continue

    similarityClusters.sort(key = lambda s: -len(s))
    return similarityClusters


def sample_coincidence_matrix(dict_, freqThreshold=0.333, frequencies=False):

    keys = dict_.keys()
    lists = [dict_[key] for key in keys]
    samples = list(set(numpy.hstack(lists)))

    template = pandas.DataFrame(numpy.zeros((len(samples),len(samples))))
    template.index = samples
    template.columns = samples
    for key in keys:
        hits = dict_[key]
        template.loc[hits,hits]+=1
    trace = numpy.array([template.iloc[i,i] for i in range(template.shape[0])])
    normDf = ((template.T)/trace).T
    if frequencies is not False:
        return normDf
    normDf[normDf<freqThreshold]=0
    normDf[normDf>0]=1

    return normDf


def plot_similarity(similarityMatrix, orderedSamples, vmin=0, vmax=0.5,
                    title="Similarity matrix", xlabel="Samples", ylabel="Samples",
                    fontsize=14, figsize=(7,7), savefig=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    try:
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


def centroid_expansion(classes, sampleMatrix, f1Threshold=0.3, returnCentroids=None):
    centroids = []
    for i in range(len(classes)):
        clusterComponents = sampleMatrix.loc[:,classes[i]]
        class1 = numpy.mean(clusterComponents, axis=1)
        hits = numpy.where(class1 > 0.6)[0]
        centroid = pandas.DataFrame(sampleMatrix.iloc[:, 0])
        centroid.columns = [i]
        centroid[i] = 0
        centroid.iloc[hits, 0] = 1
        centroids.append(centroid)

    miss = []
    centroidClusters = [[] for i in range(len(centroids))]
    for smpl in sampleMatrix.columns:
        probeVector = numpy.array(sampleMatrix[smpl])
        scores = []
        for ix in range(len(centroids)):
            tmp = __f1(numpy.array(probeVector), centroids[ix])
            scores.append(tmp)
        scores = numpy.array(scores)
        match = numpy.argmax(scores)
        if scores[match] < f1Threshold:
            miss.append(smpl)
        elif scores[match] >= f1Threshold:
            centroidClusters[match].append(smpl)

    centroidClusters.append(miss)

    if returnCentroids is not None:
        centroidMatrix = pandas.DataFrame(pandas.concat(centroids, axis=1))
        return centroidClusters, centroidMatrix

    return centroidClusters


def __f1(vector1, vector2):
    members = set(numpy.where(vector1 == 1)[0])
    nonMembers = set(numpy.where(vector1 == 0)[0])
    predictedMembers = set(numpy.where(vector2 == 1)[0])
    predictedNonMembers = set(numpy.where(vector2 == 0)[0])

    TP = len(members & predictedMembers)
    FN = len(predictedNonMembers & members)
    FP = len(predictedMembers & nonMembers)
    if TP == 0:
        return 0.0
    F1 = TP / float(TP + FN + FP)

    return F1


def map_expression_to_network(centroidMatrix, membershipMatrix, threshold=0.05):

    miss = []
    centroidClusters = [[] for i in range(centroidMatrix.shape[1])]
    for smpl in membershipMatrix.columns:
        probeVector = numpy.array(membershipMatrix[smpl])
        scores = []
        for ix in range(centroidMatrix.shape[1]):
            tmp = __f1(numpy.array(probeVector), numpy.array(centroidMatrix.iloc[:, ix]))
            scores.append(tmp)

        scores = numpy.array(scores)
        match = numpy.argmax(scores)

        if scores[match] < threshold:
            miss.append(smpl)
        elif scores[match] >= threshold:
            centroidClusters[match].append(smpl)

    centroidClusters.append(miss)
    return centroidClusters


def order_membership(centroidMatrix, membershipMatrix, mappedClusters, ylabel="",
                     resultsDirectory=None, showplot=False):
    centroidRank = []
    alreadyMapped = []
    for ix in range(centroidMatrix.shape[1]):
        tmp = numpy.where(centroidMatrix.iloc[:,ix] == 1)[0]
        signature = list(set(tmp) - set(alreadyMapped))
        centroidRank.extend(signature)
        alreadyMapped.extend(signature)

    orderedClusters = centroidMatrix.index[numpy.array(centroidRank)]
    try:
        ordered_matrix = membershipMatrix.loc[orderedClusters, numpy.hstack(mappedClusters)]
    except:
        ordered_matrix = membershipMatrix.loc[numpy.array(orderedClusters).astype(int),
                                              numpy.hstack(mappedClusters)]

    if showplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
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

def plot_differential_matrix(overExpressedMembersMatrix, underExpressedMembersMatrix,
                             orderedOverExpressedMembers, cmap="viridis", aspect="auto",
                             saveFile=None, showplot=False):
    differentialActivationMatrix = overExpressedMembersMatrix - underExpressedMembersMatrix
    orderedDM = differentialActivationMatrix.loc[orderedOverExpressedMembers.index,
                                                 orderedOverExpressedMembers.columns]

    if showplot:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        try:
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


def mosaic(dfr, clusterList, minClusterSize_x=4, minClusterSize_y=5, allow_singletons=True,
           max_groups=50, saveFile=None, random_state=12):

    lowResolutionPrograms = [[] for i in range(len(clusterList))]
    sorting_hat = []
    for i in range(len(clusterList)):
        patients = clusterList[i]
        if len(patients) < minClusterSize_x:
            continue
        subset = dfr.loc[:,patients]
        density = subset.sum(axis=1)/float(subset.shape[1])
        sorting_hat.append(numpy.array(density))

    enrichment_matrix = numpy.vstack(sorting_hat).T
    choice = numpy.argmax(enrichment_matrix,axis=1)
    for i in range(dfr.shape[0]):
        lowResolutionPrograms[choice[i]].append(dfr.index[i])

    # Cluster modules into transcriptional programs
    y_clusters = []
    for program in range(len(lowResolutionPrograms)):
        regs = lowResolutionPrograms[program]
        if len(regs) == 0:
            continue
        df = dfr.loc[regs,:]
        sil_scores = []
        max_clusters_y = min(max_groups,int(len(regs)/3.))
        for numClusters_y in range(2,max_clusters_y):
            clusters_y, labels_y, centroids_y = kmeans(df,numClusters=numClusters_y,
                                                       random_state=random_state)
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
            top_hit = min(numpy.where(numpy.array(sil_scores)>=0.95*max(sil_scores))[0] + 2)
            clusters_y, labels_y, centroids_y = kmeans(df, numClusters=top_hit,
                                                       random_state=random_state)
            clusters_y.sort(key=lambda s: -len(s))
            y_clusters.append(list(clusters_y))

        elif len(sil_scores) == 0:
            y_clusters.append(regs)

    order_y = numpy.hstack([numpy.hstack(y_clusters[i]) for i in range(len(y_clusters))])

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
            clusters_x, labels_x, centroids_x = kmeans(df, numClusters=numClusters_x,
                                                       random_state=random_state)
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
            top_hit = min(numpy.where(numpy.array(sil_scores) >= 0.999 * max(sil_scores))[0] + 2)
            clusters_x, labels_x, centroids_x = kmeans(df, numClusters=top_hit,
                                                       random_state=random_state)
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

        order_x = numpy.hstack(micro_states)
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


def kmeans(df, numClusters, random_state=None):

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
        clstr = df.index[numpy.where(labels==i)[0]]
        clusters.append(clstr)

    return clusters, labels, centroids


def transcriptional_programs(programs, reference_dictionary):
    transcriptionalPrograms = {}
    programRegulons = {}
    p_stack = []
    programs_flattened = numpy.array(programs).flatten()
    for i in range(len(programs_flattened)):
        if len(numpy.hstack(programs_flattened[i])) > len(programs_flattened[i]):
            for j in range(len(programs_flattened[i])):
                p_stack.append(list(programs_flattened[i][j]))
        else:
            p_stack.append(list(programs_flattened[i]))

    for j in range(len(p_stack)):
        key = ("").join(["TP",str(j)])
        regulonList = [i for i in p_stack[j]]
        programRegulons[key] = regulonList
        tmp = [reference_dictionary[i] for i in p_stack[j]]
        transcriptionalPrograms[key] = list(set(numpy.hstack(tmp)))
    return transcriptionalPrograms, programRegulons


# =============================================================================
# Functions used for cluster analysis
# =============================================================================

def get_eigengenes(coexpressionModules, expressionData, regulon_dict=None, saveFolder=None):
    eigengenes = principal_df(coexpressionModules, expressionData, subkey=None,
                              regulons=regulon_dict, minNumberGenes=1)
    eigengenes = eigengenes.T
    index = numpy.sort(numpy.array(eigengenes.index).astype(int))
    eigengenes = eigengenes.loc[index.astype(str),:]
    if saveFolder is not None:
        eigengenes.to_csv(os.path.join(saveFolder,"eigengenes.csv"))
    return eigengenes


def __regulon_dictionary(regulons):
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

    array = numpy.vstack(df_list)
    df = pandas.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]

    return regulonModules, df


def principal_df(dict_, expressionData, regulons=None, subkey='genes',
                   minNumberGenes=8, random_state=12):

    pcDfs = []
    setIndex = set(expressionData.index)

    if regulons is not None:
        dict_, df = __regulon_dictionary(regulons)
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
        principalDf = pandas.DataFrame(principalComponents)
        principalDf.index = expressionData.columns
        principalDf.columns = [str(i)]

        normPC = numpy.linalg.norm(numpy.array(principalDf.iloc[:,0]))
        pearson = stats.pearsonr(principalDf.iloc[:,0],
                                 numpy.median(expressionData.loc[genes,:],
                                              axis=0))
        signCorrection = pearson[0] / numpy.abs(pearson[0])

        principalDf = signCorrection*principalDf / normPC

        pcDfs.append(principalDf)

    principalMatrix = pandas.concat(pcDfs,axis=1)

    return principalMatrix


def reduce_modules(df, programs, states, stateThreshold=0.75, saveFile=None):

    df = df.loc[:, numpy.hstack(states)]
    statesDf = pandas.DataFrame(numpy.zeros((len(programs), df.shape[1])))
    statesDf.index = range(len(programs))
    statesDf.columns = df.columns

    for i in range(len(programs)):
        state = programs[i]
        subset = df.loc[state, :]

        state_scores = subset.sum(axis=0) / float(subset.shape[0])

        keep_high = numpy.where(state_scores >= stateThreshold)[0]
        keep_low = numpy.where(state_scores <= -1 * stateThreshold)[0]
        hits_high = numpy.array(df.columns)[keep_high]
        hits_low = numpy.array(df.columns)[keep_low]

        statesDf.loc[i, hits_high] = 1
        statesDf.loc[i, hits_low] = -1

    if saveFile is not None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.gca()
        ax.imshow(statesDf, cmap="bwr", vmin=-1, vmax=1, aspect='auto')
        ax.grid(False)
        ax.set_ylabel("Transcriptional programs", FontSize=14)
        ax.set_xlabel("Samples", FontSize=14)
        plt.savefig(saveFile, bbox_inches="tight")

    return statesDf


def programs_vs_states(statesDf, states, filename=None, showplot=False):
    pixel = numpy.zeros((statesDf.shape[0], len(states)))
    for i in range(statesDf.shape[0]):
        for j in range(len(states)):
            pixel[i,j] = numpy.mean(statesDf.loc[statesDf.index[i],states[j]])

    pixel = pandas.DataFrame(pixel)
    pixel.index = statesDf.index

    if showplot is False:
        return pixel

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pixel,cmap="bwr",vmin=-1,vmax=1,aspect="auto")
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylabel("Transcriptional programs",FontSize=14)
    plt.xlabel("Transcriptional states",FontSize=14)
    if filename is not None:
        plt.savefig(filename,bbox_inches="tight")

    return pixel


def tsne(matrix, perplexity=100, n_components=2, n_iter=1000, plotOnly=True, plotColor="red",
         alpha=0.4, dataOnly=False):
    X = numpy.array(matrix.T)
    X_embedded = TSNE(n_components=n_components, n_iter=n_iter, n_iter_without_progress=300,
                      init='random', random_state=0, perplexity=perplexity).fit_transform(X)
    if plotOnly is True:
        plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)
        return
    if dataOnly is True:
        return X_embedded

    plt.scatter(X_embedded[:,0],X_embedded[:,1],color=plotColor,alpha=alpha)

    return X_embedded


def tsne_state_labels(tsneDf, states):
    labelsDf = pandas.DataFrame(1000 * numpy.ones(tsneDf.shape[0]))
    labelsDf.index = tsneDf.index
    labelsDf.columns = ["label"]

    for i in range(len(states)):
        tagged = states[i]
        labelsDf.loc[tagged,"label"] = i
    state_labels = numpy.array(labelsDf.iloc[:,0])
    return state_labels


def plot_states(statesDf, tsneDf, numCols=None, numRows=None, saveFile=None,
                size=10, aspect=1, scale=2):
    if numRows is None:
        if numCols is None:
            numRows = int(round(numpy.sqrt(statesDf.shape[0])))
            rat = numpy.floor(statesDf.shape[0] / float(numRows))
            rem = statesDf.shape[0] - numRows * rat
            numCols = int(rat + rem)
        elif numCols is not None:
            numRows = int(numpy.ceil(float(statesDf.shape[0]) / numCols))

    fig = plt.figure(figsize=(scale * numRows, scale * numCols))

    for ix in range(statesDf.shape[0]):
        ax = fig.add_subplot(numRows, numCols, ix + 1)

        # overlay single state onto tSNE plot
        stateIndex = ix

        group = pandas.DataFrame(numpy.zeros(statesDf.shape[1]))
        group.index = statesDf.columns
        group.columns = ["status"]
        group.loc[statesDf.columns,"status"] = list(statesDf.iloc[stateIndex,:])
        group = numpy.array(group.iloc[:,0])
        ax.set_aspect(aspect)
        ax.scatter(tsneDf.iloc[:,0], tsneDf.iloc[:,1],
                   cmap="bwr", c=group, vmin=-1, vmax=1, s=size)

    if saveFile is not None:
        plt.savefig(saveFile, bbox_inches="tight")
