import numpy,datetime,pandas,sys
from pkg_resources import Requirement, resource_filename
from collections import Counter

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
        # WW: TODO: THIS IS STILL MISSING IN THIS MODULE, BUT IS IN ORIGINAL MINER !!!
        zscored_expression = preProcessTPM(df)
    return zscored_expression


def identifier_conversion(expression_data, conversion_table_path=None):

    # if not specified, read conversion table from package data
    if conversion_table_path is None:
        conversion_table_path = resource_filename(Requirement.parse("miner2"),
                                                  'miner2/data/identifier_mappings.txt')

    id_map = pandas.read_csv(conversion_table_path, sep='\t')

    gene_types = list(set(id_map.iloc[:,2]))
    previous_index = numpy.array(expression_data.index).astype(str)
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
        mapped_genes = list(set(previous_index) & set(subset.index))
        mapped_samples = list(set(previous_columns) & set(subset.index))

        if len(mapped_genes) >= max(10, 0.01 * expression_data.shape[0]):
            if len(mapped_genes) > len(best_match):
                best_match = mapped_genes
                state = "original"
                gtype = gene_type
                continue

        if len(mapped_samples) >= max(10, 0.01 * expression_data.shape[1]):
            if len(mapped_samples) > len(best_match):
                best_match = mapped_samples
                state = "transpose"
                gtype = gene_type
                continue

    if len(best_match) == 0:
        raise Exception("Error: Gene identifiers not recognized")

    # ALO/WW: Without sorting the mapped_genes, the values will differ in
    # multiple places, we should check why this has such a great impact
    mapped_genes = sorted(best_match)

    subset = id_map[id_map.iloc[:,2]==gtype]
    subset.index = subset.iloc[:,1]

    if state == "transpose":
        expression_data = expression_data.T

    try:
        converted_data = expression_data.loc[mapped_genes,:]
    except Exception as e:
        print(e)
        converted_data = expression_data.loc[numpy.array(mapped_genes).astype(int),:]

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

    print("WARNING: There were %d ambiguously mapped genes !" % len(duplicates))

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

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t {} out of {} gene names converted to ENSEMBL IDs".format(converted_data.shape[0], expression_data.shape[0])))

    return converted_data, conversion_table


def main(filename, conversion_table_path=None):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data reading"))
    raw_expression = read_file_to_df(filename)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data recovered: {} features by {} samples".format(raw_expression.shape[0], raw_expression.shape[1])))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t expression data transformation"))

    raw_expression_zero_filtered = remove_null_rows(raw_expression)
    zscored_expression = correct_batch_effects(raw_expression_zero_filtered)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t gene ID conversion"))
    expression_data, conversion_table = identifier_conversion(zscored_expression,
                                                              conversion_table_path)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S \t working expression data: {} features by {} samples".format(expression_data.shape[0], expression_data.shape[1])))

    return expression_data, conversion_table


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
