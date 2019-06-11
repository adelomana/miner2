import datetime,pandas,numpy,os,pickle,sys
import sklearn,sklearn.decomposition
import scipy,scipy.stats
import multiprocessing
from pkg_resources import Requirement, resource_filename
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

def enrichment(axes,revised_clusters,expression_data,correlation_threshold=0.3,num_cores=1,p=0.05,database="tfbsdb_tf_to_genes.pkl"):
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t mechanistic inference"))
    
    tf_2_genes_path = resource_filename(Requirement.parse("miner2"), 'miner2/data/{}'.format(database))
    with open(tf_2_genes_path, 'rb') as f:
        tf_2_genes = pickle.load(f)

    if correlation_threshold <= 0:
        all_genes = [int(len(expression_data.index))]
    elif correlation_threshold > 0:
        all_genes = list(expression_data.index)
        
    tfs = list(tf_2_genes.keys())
    tf_map = axis_tfs(axes,tfs,expression_data,correlation_threshold=correlation_threshold)

    tasks=[[cluster_key,(all_genes,revised_clusters,tf_map,tf_2_genes,p)] for cluster_key in list(revised_clusters.keys())]

    hydra=multiprocessing.pool.Pool(num_cores)
    results=hydra.map(tfbsdb_enrichment,tasks)

    mechanistic_output={}
    for result in results:
        for key in result.keys():
            if key not in mechanistic_output:
                mechanistic_output[key]=result[key]
            else:
                print('key twice')
                sys.exit()
    print('completed')

    sys.exit()

    return mechanistic_output

def hyper(population,set1,set2,overlap):
    
    b = max(set1,set2)
    c = min(set1,set2)
    hyp = scipy.stats.hypergeom(population,b,c)
    prb = sum([hyp.pmf(l) for l in range(overlap,c+1)])
    
    return prb 

def get_principal_df(revised_clusters,expression_data,regulons=None,subkey='genes',min_number_genes=8,random_state=12):

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t preparing mechanistic inference"))

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
    
    return regulonModules, df

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
