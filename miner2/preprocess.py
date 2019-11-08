import numpy,datetime,pandas,sys,datetime,os
from pkg_resources import Requirement, resource_filename
from collections import Counter
from scipy import stats
from scipy.stats import rankdata

import logging

def correct_batch_effects(df):
    zscored_expression = zscore(df)
    means = []
    stds = []
    for i in range(zscored_expression.shape[1]):
        mean = numpy.mean(zscored_expression.iloc[:,i])
        std = numpy.std(zscored_expression.iloc[:,i])
        means.append(mean)
        stds.append(std)
    if numpy.std(means) >= 0.15:
        zscored_expression = preprocess_tpm(df)
    return zscored_expression

def identifier_conversion(expression_data, conversion_table_path=None):

    # if not specified, read conversion table from package data
    if conversion_table_path is None:
        conversion_table_path = resource_filename(Requirement.parse("miner2"),'miner2/data/identifier_mappings.txt')

    id_map = pandas.read_csv(conversion_table_path,sep='\t')

    gene_types = list(set(id_map.iloc[:,2]))

    ### ALO.2019.06.19. Addressing dot transcripts in ENSEMBL annotations from GDC portal
    #previous_index = numpy.array(expression_data.index).astype(str) # this line is obsolete. Kept for reference. To be removed in the future
    original_IDs=numpy.array(expression_data.index).astype(str)
    logging.info('evaluate removing transcript isoform information')
    original_IDs_list=list(original_IDs)
    original_annotation=numpy.array([element.split('.')[0] for element in original_IDs_list])
    if original_IDs.shape == original_annotation.shape:
        logging.info('no information loss')
        substitution={element:element.split('.')[0] for element in original_IDs_list}
        expression_data.rename(index=substitution,inplace=True)
    else:
        logging.warn('some gene IDs have multiple transcript isoforms. Initial {} versus new {} dimensions'.format(original_IDs.shape,original_annotation.shape))
    ### end ALO

    previous_columns = numpy.array(expression_data.columns).astype(str)
    best_match = []

    # WW: This weird loop actually tries to determine whether the expression
    # matrix is original (in which case the genes are the rows) or a transpose
    # (in which case the genes are the columns). It does does so by trying to
    # map the genes in the id map to both the rows and columns and if there
    # are multiple matches in either it will determine those to be the genes.
    # It also tries to find the gene type with the most gene matches and build
    # the conversion table based on the gene type with the most gene matches
    #
    # The code seems to have multiple issues, and is very sensitive to changes
    # we should see if we can simplify it

    for gene_type in gene_types:
        subset = id_map[id_map.iloc[:,2] == gene_type]
        subset.index = subset.iloc[:,1]
        mapped_genes = list(set(original_annotation) & set(subset.index))
        mapped_samples = list(set(previous_columns) & set(subset.index))

        if len(mapped_genes) >= max(10, 0.01 * expression_data.shape[0]):
            if len(mapped_genes) > len(best_match):
                best_match = mapped_genes
                state = "original"
                gtype = gene_type
                logging.info('expression data arrangement detected: genes as rows ({}) and samples as columns ({})'.format(expression_data.shape[0],expression_data.shape[1]))
                continue

        if len(mapped_samples) >= max(10, 0.01 * expression_data.shape[1]):
            if len(mapped_samples) > len(best_match):
                best_match = mapped_samples
                state = "transpose"
                gtype = gene_type
                logging.info('expression data arrangement detected: genes as columns ({}) and samples as rows ({})'.format(expression_data.shape[0],expression_data.shape[1]))
                continue

    if len(best_match) == 0:
        raise Exception("Error: Gene identifiers not recognized")

    # ALO/WW: Without sorting the mapped_genes, the values will differ in
    # multiple places, we should check why this has such a great impact
    mapped_genes = sorted(best_match)

    subset = id_map[id_map.iloc[:,2]==gtype]
    subset.index = subset.iloc[:,1]

    if state == "transpose":
        logging.info('transpose expression data')
        expression_data = expression_data.T

    converted_data = expression_data.loc[mapped_genes,:]

    conversion_table = subset.loc[mapped_genes,:]
    conversion_table.index = conversion_table.iloc[:,0]
    conversion_table = conversion_table.iloc[:,1]
    conversion_table.columns = ["Name"]

    new_index = list(subset.loc[mapped_genes, "Preferred_Name"])
    converted_data.index = new_index

    # WW: Due to ambiguous mapping converted_data will potentially have
    # multiple rows containing the same row name, so this conflict
    # needs to be resolved
    duplicates = [item for item, count in Counter(new_index).items() if count > 1]
    singles = list(set(converted_data.index) - set(duplicates))

    # WW: with the sorting of the duplicate and singles lists we are eliminating the
    # order ambiguity that exists when we switch between Python 2 and 3, but
    # in actuality the fundamental problem is the gene mapping
    duplicates.sort()
    singles.sort()

    corrections = []

    if len(duplicates) > 0:
        logging.warn("There were %d ambiguously mapped genes !", len(duplicates))

    # WW: The way duplicates are resolved is by retrieving the duplicate rows
    # as individual DataFrames and picking the first one
    # TODO: we might have to rethink the strategy because the ambiguous row names
    # were introduced by the way the rows were remapped and we actually need to
    # map them to individual genes
    for duplicate in duplicates:
        dup_data = converted_data.loc[duplicate,:]
        first_choice = pandas.DataFrame(dup_data.iloc[0,:]).T
        corrections.append(first_choice)

    if len(corrections) > 0:
        corrections_df = pandas.concat(corrections, axis=0)
        uncorrected_data = converted_data.loc[singles,:]
        converted_data = pandas.concat([uncorrected_data, corrections_df], axis=0)

    logging.info("{} out of {} gene names converted to ENSEMBL IDs".format(converted_data.shape[0], expression_data.shape[0]))

    return converted_data,conversion_table

def entropy(vector):

    data = numpy.array(vector)
    hist = numpy.histogram(data, bins=50)[0]
    length = len(hist)

    if length <= 1:
        return 0

    counts = numpy.bincount(hist)
    probs = [float(i) / length for i in counts]
    n_classes = numpy.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        if i > 0:
            ent -= float(i) * numpy.log(i)
    return ent

def main(input_path,conversion_table_path=None):

    # first detect if it's a dataframe or dir or a file to parse data differently
    if isinstance(input, pandas.DataFrame):
        raw_expression = input
    elif os.path.isfile(input_path) == True:
        logging.info('detected expression data file')
        raw_expression = read_file_to_df(input_path)
    elif os.path.isdir(input_path) == True:
        logging.info('detected folder from GDC')
        raw_expression_data = read_expression_from_GDC_download(input_path)
        raw_expression = transform_to_FPKM(raw_expression_data,fpkm_threshold=1,min_fraction_above_threshold=0.5,highly_expressed=False)
    else:
        raise Exception("Error: Unable to identify if input path is directory or expression file.")
    logging.info("expression data recovered: {} features by {} samples".format(raw_expression.shape[0], raw_expression.shape[1]))

    # data transformations
    logging.info("expression data transformation")
    raw_expression_zero_filtered = remove_null_rows(raw_expression)
    zscored_expression = correct_batch_effects(raw_expression_zero_filtered)

    ("gene ID conversion")
    expression_data, conversion_table = identifier_conversion(zscored_expression,conversion_table_path)
    logging.info("working expression data: {} features by {} samples".format(expression_data.shape[0], expression_data.shape[1]))

    return expression_data, conversion_table

def read_expression_from_GDC_download(directory):

    sample_dfs=[]
    for dirName, subdirList, fileList in os.walk(directory):
        for fname in fileList:
            if fname[0] != '.':
                extension=fname.split(".")[-1]
                if extension == 'gz':
                    path=os.path.join(directory,dirName,fname)
                    df=pandas.read_csv(path,compression='gzip',index_col=0,header=None,sep='\t',quotechar='"')
                    df.columns=[fname.split(".")[0]]
                    sample_dfs.append(df)

    expressionData = pandas.concat(sample_dfs,axis=1)

    return expressionData


def read_file_to_df(filename):
    """read expression file into a Pandas DataFrame depending on the file extension"""
    extension = filename.split(".")[-1]
    if extension == "csv":
        df = pandas.read_csv(filename, index_col=0, header=0)
        num_rows, num_cols = df.shape
        if num_cols == 0:
            df = pandas.read_csv(filename, index_col=0, header=0, sep="\t")
    elif extension == "txt":
        df = pandas.read_csv(filename, index_col=0, header=0, sep="\t")
        num_rows, num_cols = df.shape
        if num_cols == 0:
            df = pandas.read_csv(filename, index_col=0, header=0)
    return df


def remove_null_rows(df):
    minimum = numpy.percentile(df, 0)
    if minimum == 0:
        filtered_df = df.loc[df.sum(axis=1)>0,:]
    else:
        filtered_df = df
    return filtered_df


def preprocess_tpm(tpm):
    cutoff = stats.norm.ppf(0.00001)
    tmp_array_raw = numpy.array(tpm)
    keep = []
    keepappend = keep.append
    for i in range(0,tmp_array_raw.shape[0]):
        if numpy.count_nonzero(tmp_array_raw[i,:]) >= round(float(tpm.shape[1])*0.5):
            keepappend(i)

    tpm_zero_filtered = tmp_array_raw[keep,:]
    tpm_array = numpy.array(tpm_zero_filtered)
    positive_medians = []

    for i in range(0,tpm_array.shape[1]):
        tmp1 = tpm_array[:,i][tpm_array[:,i]>0]
        positive_medians.append(numpy.median(tmp1))

    # 2^10 - 1 = 1023
    scale_factors = [float(1023)/positive_medians[i] for i in range(0,len(positive_medians))]

    tpm_scale = numpy.zeros(tpm_array.shape)
    for i in range(0,tpm_scale.shape[1]):
        tpm_scale[:,i] = tpm_array[:,i]*scale_factors[i]

    tpm_scale_log2 = numpy.zeros(tpm_scale.shape)
    for i in range(0,tpm_scale_log2.shape[1]):
        tpm_scale_log2[:,i] = numpy.log2(tpm_scale[:,i]+1)

    tpm_filtered_df = pandas.DataFrame(tpm_scale_log2)
    tpm_filtered_df.columns = list(tpm.columns)
    tpm_filtered_df.index = list(numpy.array(tpm.index)[keep])

    qn_tpm_filtered = quantile_norm(tpm_filtered_df,axis=0)
    qn_tpm = quantile_norm(qn_tpm_filtered,axis=1)

    qn_tpm_array = numpy.array(qn_tpm)

    tpm_z = numpy.zeros(qn_tpm_array.shape)
    for i in range(0,tpm_z.shape[0]):
        tmp = qn_tpm_array[i,:][qn_tpm_array[i,:]>0]
        mean = numpy.mean(tmp)
        std = numpy.std(tmp)
        for j in range(0,tpm_z.shape[1]):
            tpm_z[i,j] = float(qn_tpm_array[i,j] - mean)/std
            if tpm_z[i,j] < -4:
                tpm_z[i,j] = cutoff

    tpm_entropy = []
    for i in range(0,tpm_z.shape[0]):
        tmp = entropy(tpm_z[i,:])
        tpm_entropy.append(tmp)

    tpmz_df = pandas.DataFrame(tpm_z)
    tpmz_df.columns = list(tpm.columns)
    tpmz_df.index = list(numpy.array(tpm.index)[keep])


    ent = pandas.DataFrame(tpm_entropy)
    ent.index = list(tpmz_df.index)
    ent.columns = ['entropy']

    tpm_ent_df = pandas.concat([tpmz_df,ent],axis=1)

    tpm_entropy_sorted = tpm_ent_df.sort_values(by='entropy',ascending=False)

    tmp = tpm_entropy_sorted[tpm_entropy_sorted.loc[:,'entropy']>=0]
    tpm_select = tmp.iloc[:,0:-1]

    return tpm_select


def quantile_norm(df,axis=1):

    if axis == 1:
        array = numpy.array(df)

        ranked_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[0]):
            ranked_array[i,:] = rankdata(array[i,:],method='min') - 1

        sorted_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[0]):
            sorted_array[i,:] = numpy.sort(array[i,:])

        qn_values = numpy.nanmedian(sorted_array,axis=0)

        quant_norm_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]

        quant_norm = pandas.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)

    if axis == 0:
        array = numpy.array(df)

        ranked_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[1]):
            ranked_array[:,i] = rankdata(array[:,i],method='min') - 1

        sorted_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[1]):
            sorted_array[:,i] = numpy.sort(array[:,i])

        qn_values = numpy.nanmedian(sorted_array,axis=1)

        quant_norm_array = numpy.zeros(array.shape)
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                quant_norm_array[i,j] = qn_values[int(ranked_array[i,j])]

        quant_norm = pandas.DataFrame(quant_norm_array)
        quant_norm.columns = list(df.columns)
        quant_norm.index = list(df.index)

    return quant_norm

def transform_to_FPKM(expression_data,fpkm_threshold=1,min_fraction_above_threshold=0.5,highly_expressed=False,quantile_normalize=False):

    median = numpy.median(numpy.median(expression_data,axis=1))
    expDataCopy = expression_data.copy()
    expDataCopy[expDataCopy<fpkm_threshold]=0
    expDataCopy[expDataCopy>0]=1
    cnz = numpy.count_nonzero(expDataCopy,axis=1)
    keepers = numpy.where(cnz>=int(min_fraction_above_threshold*expDataCopy.shape[1]))[0]
    threshold_genes = expression_data.index[keepers]
    expDataFiltered = expression_data.loc[threshold_genes,:]

    if highly_expressed is True:
        median = numpy.median(numpy.median(expDataFiltered,axis=1))
        expDataCopy = expDataFiltered.copy()
        expDataCopy[expDataCopy<median]=0
        expDataCopy[expDataCopy>0]=1
        cnz = numpy.count_nonzero(expDataCopy,axis=1)
        keepers = numpy.where(cnz>=int(0.5*expDataCopy.shape[1]))[0]
        median_filtered_genes = expDataFiltered.index[keepers]
        expDataFiltered = expression_data.loc[median_filtered_genes,:]

    if quantile_normalize is True:
        expDataFiltered = quantile_norm(expDataFiltered,axis=0)

    finalExpData = pandas.DataFrame(numpy.log2(expDataFiltered+1))
    finalExpData.index = expDataFiltered.index
    finalExpData.columns = expDataFiltered.columns

    return finalExpData

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
    return transform


def background_df(expression_data):

    low = numpy.percentile(expression_data, 100. / 3, axis=0)
    high = numpy.percentile(expression_data, 200. / 3, axis=0)
    even_cuts = list(zip(low, high))

    bkgd = expression_data.copy()
    for i in range(bkgd.shape[1]):
        low_cut = even_cuts[i][0]
        high_cut = even_cuts[i][1]
        bkgd.iloc[:, i][bkgd.iloc[:, i] >= high_cut] = 1
        bkgd.iloc[:, i][bkgd.iloc[:, i] <= low_cut] = -1
        bkgd.iloc[:, i][numpy.abs(bkgd.iloc[:, i]) != 1] = 0

    return bkgd
