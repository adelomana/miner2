import datetime, pandas, numpy, os, pickle, sys
import sklearn, sklearn.decomposition
import scipy, scipy.stats
import multiprocessing
from pkg_resources import Requirement, resource_filename
import logging

import miner2.coexpression

def axis_tfs(axes_df,tf_list,expression_data,correlation_threshold=0.3):

    axes_array = numpy.array(axes_df.T)
    if correlation_threshold > 0:
        tf_array=numpy.array(expression_data.reindex(tf_list)) # ALO Py3
    axes = numpy.array(axes_df.columns)
    tf_dict = {}

    if type(tf_list) is list:
        tfs = numpy.array(tf_list)
    elif type(tf_list) is not list:
        tfs = tf_list

    if correlation_threshold == 0:
        for axis in range(axes_array.shape[0]):
            tf_dict[axes[axis]] = tfs

        return tf_dict

    for axis in range(axes_array.shape[0]):
        tf_correlation = miner2.coexpression.pearson_array(tf_array,axes_array[axis,:])
        ### ALO, fixed warning over nan evaluations
        condition1=numpy.greater_equal(numpy.abs(tf_correlation),correlation_threshold,where=numpy.isnan(tf_correlation) == False)
        condition2=numpy.isnan(tf_correlation)
        tf_dict[axes[axis]]=tfs[numpy.where(numpy.bitwise_and(condition1 == True, condition2 == False))[0]]
        ### end ALO

    return tf_dict

def enrichment(axes, revised_clusters, expression_data, correlation_threshold=0.3,
               num_cores=1, p=0.05,
               database="tfbsdb_tf_to_genes.pkl",
               database_path=None,
               single_cell=False):

    logging.info("mechanistic inference")

    if database_path is None:
        tf_2_genes_path = resource_filename(Requirement.parse("miner2"),
                                            'miner2/data/{}'.format(database))
    else:
        tf_2_genes_path = database_path

    with open(tf_2_genes_path, 'rb') as f:
        tf_2_genes = pickle.load(f)

    if single_cell:
        # clean tf_2_genes to only include genes in revised_clusters
        revised_clusters_gene_set = set()
        for key in revised_clusters:
            revised_clusters_gene_set.update(revised_clusters[key])
        for key in tf_2_genes:
            genes = tf_2_genes[key]
            new_genes = [gene for gene in genes if gene in revised_clusters_gene_set]
            tf_2_genes.update({key: new_genes})

    if correlation_threshold <= 0:
        all_genes = [int(len(expression_data.index))]
    elif correlation_threshold > 0:
        all_genes = list(expression_data.index)

    tfs = list(tf_2_genes.keys())
    tf_map = axis_tfs(axes, tfs, expression_data, correlation_threshold=correlation_threshold)

    tasks = [[cluster_key,(all_genes,revised_clusters,tf_map,tf_2_genes,p)]
             for cluster_key in list(revised_clusters.keys())]

    hydra = multiprocessing.pool.Pool(num_cores)
    results = hydra.map(tfbsdb_enrichment, tasks)

    mechanistic_output = {}
    for result in results:
        for key in result.keys():
            if key not in mechanistic_output:
                mechanistic_output[key]=result[key]
            else:
                raise Exception('key "%s" twice' % key)

    return mechanistic_output


def hyper(population,set1,set2,overlap):

    b = max(set1,set2)
    c = min(set1,set2)
    hyp = scipy.stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])

    return prb

def get_principal_df(revised_clusters,expression_data,regulons=None,subkey='genes',min_number_genes=8,random_state=12):

    logging.info("preparing mechanistic inference")

    pc_Dfs = []
    set_index = set(expression_data.index)

    if regulons is not None:
        revised_clusters, df = get_regulon_dictionary(regulons)
    for i in revised_clusters.keys():
        if subkey is not None:
            genes = list(set(revised_clusters[i][subkey])&set_index)
            if len(genes) < min_number_genes:
                continue
        elif subkey is None:
            genes = list(set(revised_clusters[i])&set_index)
            if len(genes) < min_number_genes:
                continue

        pca = sklearn.decomposition.PCA(1,random_state=random_state)
        principal_components = pca.fit_transform(expression_data.loc[genes,:].T)
        principal_Df = pandas.DataFrame(principal_components)
        principal_Df.index = expression_data.columns
        principal_Df.columns = [str(i)]

        norm_PC = numpy.linalg.norm(numpy.array(principal_Df.iloc[:,0]))
        pearson = scipy.stats.pearsonr(principal_Df.iloc[:,0],numpy.median(expression_data.loc[genes,:],axis=0))
        sign_correction = pearson[0]/numpy.abs(pearson[0])

        principal_Df = sign_correction*principal_Df/norm_PC

        pc_Dfs.append(principal_Df)

    principal_matrix = pandas.concat(pc_Dfs,axis=1)

    return principal_matrix

def get_regulon_dictionary(regulons):
    regulon_modules = {}
    df_list = []

    for tf in regulons.keys():
        for key in regulons[tf].keys():
            genes = regulons[tf][key]
            id_ = str(len(regulon_modules))
            regulon_modules[id_] = regulons[tf][key]
            for gene in genes:
                df_list.append([id_,tf,gene])

    array = numpy.vstack(df_list)
    df = pandas.DataFrame(array)
    df.columns = ["Regulon_ID","Regulator","Gene"]

    return regulon_modules, df

def tfbsdb_enrichment(task):

    cluster_key=task[0]
    all_genes=task[1][0]
    revised_clusters=task[1][1]
    tf_map=task[1][2]
    tf_2_genes=task[1][3]
    p=task[1][4]

    population_size = len(all_genes)

    cluster_tfs = {}
    for tf in tf_map[str(cluster_key)]:
        hits0_tf_targets = tf_2_genes[tf]
        hits0_cluster_genes = revised_clusters[cluster_key]
        overlap_cluster = list(set(hits0_tf_targets)&set(hits0_cluster_genes))
        if len(overlap_cluster) <= 1:
            continue
        p_hyper = hyper(population_size,len(hits0_tf_targets),len(hits0_cluster_genes),len(overlap_cluster))
        if p_hyper < p:
            if cluster_key not in cluster_tfs.keys():
                cluster_tfs[cluster_key] = {}
                cluster_tfs[cluster_key][tf] = [p_hyper,overlap_cluster]

    return cluster_tfs


def get_coregulation_modules(mechanistic_output):
    coregulation_modules = {}
    for i in mechanistic_output.keys():
        for key in mechanistic_output[i].keys():
            if key not in coregulation_modules.keys():
                coregulation_modules[key] = {}
            genes = mechanistic_output[i][key][1]
            coregulation_modules[key][i] = genes
    return coregulation_modules


def get_regulons(coregulation_modules, min_number_genes=5, freq_threshold=0.333):
    """TODO: There is still a discrepancy between Python 2 and 3 here"""
    regulons = {}
    for tf in sorted(coregulation_modules.keys()):
        norm_df = __coincidence_matrix(coregulation_modules, tf, freq_threshold=freq_threshold)
        unmixed = __unmix(norm_df)
        remixed = __remix(norm_df, unmixed)

        if len(remixed) > 0:
            for cluster in remixed:
                if len(cluster) > min_number_genes:
                    if tf not in regulons.keys():
                        regulons[tf] = {}
                    regulons[tf][len(regulons[tf])] = cluster
    return regulons


def __coincidence_matrix(coregulation_modules, tf, freq_threshold):
    sub_regulons = coregulation_modules[tf]
    sr_genes = list(set(numpy.hstack([sub_regulons[i] for i in sorted(sub_regulons.keys())])))

    template = pandas.DataFrame(numpy.zeros((len(sr_genes), len(sr_genes))))
    template.index = sr_genes
    template.columns = sr_genes

    for key in sorted(sub_regulons.keys()):
        genes = sub_regulons[key]
        template.loc[genes, genes] += 1

    trace = numpy.array([template.iloc[i, i] for i in range(template.shape[0])]).astype(float)
    norm_df = ((template.T) / trace).T
    norm_df[norm_df < freq_threshold] = 0
    norm_df[norm_df > 0] = 1

    return norm_df


def __unmix(df, iterations=25, return_all=False):
    frequency_clusters = []

    for iteration in range(iterations):
        sum_df1 = df.sum(axis=1)
        max_sum = numpy.argmax(sum_df1)
        hits = numpy.where(df.loc[max_sum] > 0)[0]
        hit_index = list(df.index[hits])
        block = df.loc[hit_index, hit_index]
        block_sum = block.sum(axis=1)
        core_block = list(block_sum.index[numpy.where(block_sum >= numpy.median(block_sum))[0]])
        remainder = list(set(df.index) - set(core_block))

        frequency_clusters.append(core_block)
        if len(remainder) == 0:
            return frequency_clusters
        if len(core_block) == 1:
            return frequency_clusters
        df = df.loc[remainder, remainder]

    if return_all:
        frequency_clusters.append(remainder)
    return frequency_clusters


def __remix(df, frequency_clusters):
    final_clusters = []
    for cluster in frequency_clusters:
        slice_df = df.loc[cluster,:]
        sum_slice = slice_df.sum(axis=0)
        cut = min(0.8, numpy.percentile(sum_slice.loc[cluster] / float(len(cluster)), 90))
        min_genes = max(4, cut * len(cluster))
        keepers = list(slice_df.columns[numpy.where(sum_slice >= min_genes)[0]])
        keepers = list(set(keepers) | set(cluster))
        final_clusters.append(keepers)
        final_clusters.sort(key=lambda s: -len(s))
    return final_clusters


def get_coexpression_modules(mechanistic_output):
    coexpression_modules = {}
    for i in mechanistic_output.keys():
        genes = list(set(numpy.hstack([mechanistic_output[i][key][1]
                                       for key in mechanistic_output[i].keys()])))
        coexpression_modules[i] = genes
    return coexpression_modules


"""
Postprocessing functions
"""
def convert_dictionary(dic, conversion_table):
    converted = {}
    for i in dic.keys():
        genes = dic[i]
        conv_genes = conversion_table[genes]
        for j in range(len(conv_genes)):
            if type(conv_genes[j]) is pandas.core.series.Series:
                conv_genes[j] = conv_genes[j][0]
        converted[i] = list(conv_genes)
    return converted


def convert_regulons(df, conversionTable):
    regIds = []
    regs = []
    genes = []
    for i in range(df.shape[0]):
        regIds.append(df.iloc[i,0])
        tmpReg = conversionTable[df.iloc[i,1]]
        if type(tmpReg) is pandas.core.series.Series:
            tmpReg = tmpReg[0]
        regs.append(tmpReg)
        tmpGene = conversionTable[df.iloc[i, 2]]
        if type(tmpGene) is pandas.core.series.Series:
            tmpGene = tmpGene[0]
        genes.append(tmpGene)

    regulonDfConverted = pandas.DataFrame(numpy.vstack([regIds, regs, genes]).T)
    regulonDfConverted.columns = ["Regulon_ID","Regulator","Gene"]
    return regulonDfConverted
