import datetime,numpy,pandas,time,sys,itertools
import sklearn,sklearn.decomposition
import multiprocessing, multiprocessing.pool
from collections import Counter

# Some default constants if the user does not specify any
# default number of iterations for algorithms with iterations
NUM_ITERATIONS = 25
MIN_NUM_GENES = 6
MIN_NUM_OVEREXP_SAMPLES = 4
MAX_SAMPLES_EXCLUDED = 0.5
RANDOM_STATES = 12
OVEREXP_THRESHOLD = 80
NUM_CORES = 1
RECONSTRUCTION_THRESHOLD = 0.925
SIZE_LONG_SET = 50

def cluster(expression_data,
            min_number_genes=MIN_NUM_GENES,
            min_number_overexp_samples=MIN_NUM_OVEREXP_SAMPLES,
            max_samples_excluded=MAX_SAMPLES_EXCLUDED,
            random_state=RANDOM_STATES,
            overexpression_threshold=OVEREXP_THRESHOLD,
            num_cores=NUM_CORES):
    """
    Create a list of initial clusters. This is a list of list of gene names
    """

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t coexpression"))

    df = expression_data.copy()

    max_step = int(numpy.round(10 * max_samples_excluded))
    all_genes_mapped = []
    best_hits = []

    zero = numpy.percentile(expression_data, 0)
    expression_threshold = numpy.mean([numpy.percentile(expression_data.iloc[:,i][expression_data.iloc[:,i] > zero], overexpression_threshold) for i in range(expression_data.shape[1])])

    trial = -1

    for step in range(max_step):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t working on coexpression step {} out of {}".format(step + 1, max_step)))
        trial += 1
        genes_mapped = []
        best_mapped = []

        pca = sklearn.decomposition.PCA(10, random_state=random_state)
        principal_components = pca.fit_transform(df.T)
        principal_df = pandas.DataFrame(principal_components)
        principal_df.index = df.columns

        # explore PCs in parallel
        pcs = numpy.arange(10)
        tasks = [(df, principal_df, element, min_number_genes) for element in pcs]
        hydra = multiprocessing.pool.Pool(num_cores)
        genes_mapped_parallel = hydra.map(gene_mapper, tasks)
        for element in genes_mapped_parallel:
            for gene in element:
                genes_mapped.append(gene)

        all_genes_mapped.extend(genes_mapped)

        try:
            stack_genes = numpy.hstack(genes_mapped)
        except:
            stack_genes = []

        residual_genes = sorted(list(set(df.index) - set(stack_genes)))
        df = df.loc[residual_genes, :]

        # significance surrogate in parallel
        tasks = [(element, expression_data, expression_threshold)
                 for element in genes_mapped]
        parallel_hits = hydra.map(parallel_overexpress_surrogate, tasks)

        for element, hits in parallel_hits:
            best_mapped.append(hits)
            if len(hits) > min_number_overexp_samples:
                best_hits.append(element)

        if len(best_mapped) > 0:
            count_hits = Counter(numpy.hstack(best_mapped))
            ranked = count_hits.most_common()
            dominant = [i[0] for i in ranked[0:int(numpy.ceil(0.1 * len(ranked)))]]
            remainder = [i for i in numpy.arange(df.shape[1]) if i not in dominant]
            df = df.iloc[:, remainder]

    return sorted(best_hits, key=lambda s: -len(s))


def combine_clusters(axes, clusters, threshold):
    combine_axes = {}
    filter_keys = numpy.array(list(axes.keys())) # ALO: changed to list because of Py3
    axes_matrix = numpy.vstack([axes[i] for i in filter_keys])
    for key in filter_keys:
        axis = axes[key]
        pearson = pearson_array(axes_matrix,axis)
        combine = numpy.where(pearson > threshold)[0]
        combine_axes[key] = filter_keys[combine]

    revised_clusters = {}
    combined_keys = decompose_dictionary_to_lists(combine_axes)
    for key_list in combined_keys:
        genes = list(set(numpy.hstack([clusters[i] for i in key_list])))
        revised_clusters[len(revised_clusters)] = sorted(genes)

    return revised_clusters


def decompose(geneset, expression_data, min_number_genes):
    fm = make_frequency_matrix(expression_data.loc[geneset,:])
    tst = numpy.multiply(fm, fm.T)
    tst[tst < numpy.percentile(tst, 80)] = 0
    tst[tst > 0] = 1
    unmix_tst = unmix(tst)
    return [i for i in unmix_tst if len(i) >= min_number_genes]


def decompose_dictionary_to_lists(dict_):
    decomposed_sets = []
    for key in dict_.keys():
        new_set = iterative_combination(dict_, key, NUM_ITERATIONS)
        if new_set not in decomposed_sets:
            decomposed_sets.append(new_set)
    return decomposed_sets


def make_frequency_matrix(matrix, overexp_threshold=1):

    num_rows = matrix.shape[0]

    if type(matrix) == pandas.core.frame.DataFrame:
        index = matrix.index
        matrix = numpy.array(matrix)
    else:
        index = numpy.arange(numRows)

    matrix[matrix < overexp_threshold] = 0
    matrix[matrix > 0] = 1

    frequency_matrix = make_hits_matrix_new(matrix)
    trace_fm = numpy.array([frequency_matrix[i,i]
                           for i in range(frequency_matrix.shape[0])]).astype(float)

    if numpy.count_nonzero(trace_fm) < len(trace_fm):
        #subset nonzero. computefm. normFM zeros with input shape[0]. overwrite by slice np.where trace>0
        nonzero_genes = numpy.where(trace_fm > 0)[0]
        norm_fm_nonzero = numpy.transpose(numpy.transpose(frequency_matrix[nonzero_genes,:][:, nonzero_genes]) /
                                          trace_fm[nonzero_genes])
        norm_df = pandas.DataFrame(norm_fm_nonzero, index=index[nonzero_genes],
                                   columns=index[nonzero_genes])
    else:
        norm_fm = numpy.transpose(numpy.transpose(frequency_matrix) / trace_fm)
        norm_df = pandas.DataFrame(norm_fm, index=index, columns=index)

    return norm_df


def get_axes(clusters, expression_data):
    axes = {}
    for key in clusters.keys():
        genes = clusters[key]
        fpc = sklearn.decomposition.PCA(1)
        principal_components = fpc.fit_transform(expression_data.loc[genes,:].T)
        axes[key] = principal_components.ravel()
    return axes


def gene_mapper(task):
    genes_mapped = []
    df, principal_df, i, min_number_genes = task
    pearson = pearson_array(numpy.array(df), numpy.array(principal_df[i]))
    highpass = max(numpy.percentile(pearson,95), 0.1)
    lowpass = min(numpy.percentile(pearson,5), -0.1)
    cluster1 = numpy.array(df.index[numpy.where(pearson > highpass)[0]])
    cluster2 = numpy.array(df.index[numpy.where(pearson < lowpass)[0]])

    for clst in [cluster1, cluster2]:
        pdc = recursive_alignment(clst, df, min_number_genes)
        if len(pdc) == 0:
            continue
        elif len(pdc) == 1:
            genes_mapped.append(pdc[0])
        elif len(pdc) > 1:
            for j in range(len(pdc)-1):
                if len(pdc[j]) > min_number_genes:
                    genes_mapped.append(pdc[j])

    return genes_mapped


def iterative_combination(dict_, key, iterations):
    initial = dict_[key]
    initial_length = len(initial)

    for iteration in range(iterations):
        revised = [i for i in initial]
        for element in initial:
            # WW: sorting for comparability
            revised = sorted(list(set(revised) | set(dict_[element])))
        revised_length = len(revised)
        if revised_length == initial_length:
            return revised
        elif revised_length > initial_length:
            initial = [i for i in revised]
            initial_length = len(initial)
    return revised


def make_hits_matrix_new(matrix): ### new function developped by Wei-Ju
    num_rows = matrix.shape[0]
    hits_values = numpy.zeros((num_rows,num_rows))

    for column in range(matrix.shape[1]):
        geneset = matrix[:,column]
        hits = numpy.where(geneset > 0)[0]
        rows = []
        cols = []
        cp = itertools.product(hits, hits)
        for row, col in cp:
            rows.append(row)
            cols.append(col)
        hits_values[rows, cols] += 1

    return hits_values


def parallel_overexpress_surrogate(task):
    element, expression_data, expression_threshold = task

    tmp_cluster = expression_data.loc[element, :]
    tmp_cluster[tmp_cluster < expression_threshold] = 0
    tmp_cluster[tmp_cluster > 0] = 1
    sum_cluster = numpy.array(numpy.sum(tmp_cluster, axis=0))
    hits = numpy.where(sum_cluster > 0.333 * len(element))[0]

    return (element, hits)


def pearson_array(array, vector):
    ybar = numpy.mean(vector)
    sy = numpy.std(vector, ddof=1)
    yterms = (vector - ybar) / float(sy)

    array_sx = numpy.std(array, axis=1, ddof=1)

    if 0 in array_sx:
        pass_index = numpy.where(array_sx > 0)[0]
        array = array[pass_index, :]
        array_sx = array_sx[pass_index]

    array_xbar = numpy.mean(array, axis=1)
    product_array = numpy.zeros(array.shape)

    for i in range(0,product_array.shape[1]):
        product_array[:, i] = yterms[i] * (array[:, i] - array_xbar) / array_sx

    return numpy.sum(product_array, axis=1) / float(product_array.shape[1]-1)


def process_coexpression_lists(lists, expression_data, threshold):
    reconstructed = reconstruction(lists, expression_data, threshold)
    reconstructed_list = [reconstructed[i] for i in reconstructed.keys()]
    reconstructed_list.sort(key = lambda s: -len(s))
    return reconstructed_list


def reconstruction(decomposed_list, expression_data, threshold=RECONSTRUCTION_THRESHOLD):
    clusters = {i:decomposed_list[i] for i in range(len(decomposed_list))}
    axes = get_axes(clusters, expression_data)
    return combine_clusters(axes, clusters, threshold)


def recursive_alignment(geneset, expression_data, min_number_genes):
    rec_decomp = recursive_decomposition(geneset, expression_data, min_number_genes)
    if len(rec_decomp) == 0:
        return []
    reconstructed = reconstruction(rec_decomp, expression_data)

    # ALO: call list() on keys() changed to list because of Python3 behaviour
    reconstructed_list = [reconstructed[i] for i in list(reconstructed.keys())
                          if len(reconstructed[i]) > min_number_genes]
    return sorted(reconstructed_list, key=lambda s: -len(s))


def recursive_decomposition(geneset, expression_data, min_number_genes):
    unmixed_filtered = decompose(geneset, expression_data, min_number_genes)
    if len(unmixed_filtered) == 0:
        return []

    short_sets = [i for i in unmixed_filtered if len(i) < SIZE_LONG_SET]
    long_sets = [i for i in unmixed_filtered if len(i) >= SIZE_LONG_SET]

    if len(long_sets) == 0:
        return unmixed_filtered

    for ls in long_sets:
        unmixed_filtered = decompose(ls, expression_data, min_number_genes)
        if len(unmixed_filtered) == 0:
            continue
        short_sets.extend(unmixed_filtered)
    return short_sets


def revise_initial_clusters(cluster_list, expression_data, threshold=RECONSTRUCTION_THRESHOLD):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t genes clustered: {}".format(len(set(numpy.hstack(cluster_list))))))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t revising initial clusters"))
    coexpression_lists = process_coexpression_lists(cluster_list, expression_data, threshold)

    for iteration in range(5):
        previous_length = len(coexpression_lists)
        coexpression_lists = process_coexpression_lists(coexpression_lists,
                                                        expression_data,
                                                        threshold)
        new_length = len(coexpression_lists)
        if new_length == previous_length:
            break

    coexpression_dict = {str(i):list(coexpression_lists[i])
                         for i in range(len(coexpression_lists))}

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t revision completed"))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t genes clustered: {}".format(len(set(numpy.hstack(cluster_list))))))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t unique clusters: {}".format(len(coexpression_dict))))

    return coexpression_dict


def unmix(df, iterations=NUM_ITERATIONS, return_all=False):
    frequency_clusters = []
    for iteration in range(iterations):
        # WW: replaced it with the old idxmax()
        # call for now before checking against Python 3
        max_sum = df.sum(axis=1).idxmax()
        """# ALO: consistent return in case of ties
        sum_df1 = df.sum(axis=1)
        selected=sumDf1[sumDf1.values == sumDf1.values.max()]
        chosen=selected.index.tolist()
        if len(chosen) > 1:
            chosen.sort()
        max_sum=chosen[0]
        # end ALO"""

        hits = numpy.where(df.loc[max_sum] > 0)[0]
        hit_index = list(df.index[hits])
        block = df.loc[hit_index, hit_index]
        block_sum = block.sum(axis=1)
        core_block = list(block_sum.index[numpy.where(block_sum >= numpy.median(block_sum))[0]])

        # WW: sorting for comparability
        remainder = sorted(list(set(df.index) - set(core_block)))
        frequency_clusters.append(core_block)
        if len(remainder) == 0:
            return frequency_clusters
        if len(core_block) == 1:
            return frequency_clusters
        df = df.loc[remainder, remainder]

    if return_all:
        frequency_clusters.append(remainder)

    return frequency_clusters
