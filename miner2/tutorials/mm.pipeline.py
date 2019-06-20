import sys,os,dill,numpy
import matplotlib,matplotlib.pyplot

matplotlib.rcParams.update({'font.size':18,'font.family':'Arial','xtick.labelsize':14,'ytick.labelsize':14})
matplotlib.rcParams['pdf.fonttype']=42

import miner2
import miner2.preprocess
import miner2.coexpression
import miner2.mechanistic_inference

# 0.0. user defined variables
#expression_file='/Volumes/omics4tb2/alomana/projects/miner2/data/IA12Zscore.csv'
#results_dir='/Volumes/omics4tb2/alomana/projects/miner2/results/IA12Zscore/'

expression_file='/Volumes/omics4tb2/alomana/projects/miner2/data/IA12Zscore.lines.10000.csv'
results_dir='/Volumes/omics4tb2/alomana/projects/miner2/results/IA12Zscore.lines.10000/'

num_cores = 4          # required for coexpression
min_number_genes = 6   # required for coexpression
min_correlation = 0.2  # required for mechanistic inference. Bulk RNAseq default=0.2;single cell RNAseq default=0.05

# 0.1. build results directory tree
if os.path.exists(results_dir) == False:
    os.mkdir(results_dir)
    os.mkdir(results_dir+'figures')
    os.mkdir(results_dir+'info')

# use as needed
# dill.dump_session(results_dir+'info/bottle.dill')
# dill.load_session(results_dir+'info/bottle.dill')

# STEP 0: load the data
expression_data, conversion_table = miner2.preprocess.main(expression_file)

individual_expression_data = [expression_data.iloc[:,i] for i in range(50)]
matplotlib.pyplot.boxplot(individual_expression_data)
matplotlib.pyplot.title("Patient expression profiles")
matplotlib.pyplot.ylabel("Relative expression")
matplotlib.pyplot.xlabel("Sample ID")
matplotlib.pyplot.xticks(fontsize=6)

figure_name=results_dir+'figures/boxplots.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figure_name)
matplotlib.pyplot.clf()

matplotlib.pyplot.hist(expression_data.iloc[0,:],bins=100,alpha=0.75)
matplotlib.pyplot.title("Expression of single gene")
matplotlib.pyplot.ylabel("Frequency")
matplotlib.pyplot.xlabel("Relative expression")

figure_name=results_dir+'figures/singleGene.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figure_name)
matplotlib.pyplot.clf()

matplotlib.pyplot.hist(expression_data.iloc[:,0],bins=200,color=[0,0.4,0.8],alpha=0.75)
matplotlib.pyplot.ylim(0,350)
matplotlib.pyplot.title("Expression of single patient sample",FontSize=14)
matplotlib.pyplot.ylabel("Frequency")
matplotlib.pyplot.xlabel("Relative expression")

figure_name=results_dir+'figures/singlePatient.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figure_name)
matplotlib.pyplot.clf()

sys.exit()

# STEP 1: clustering
initial_clusters = miner2.coexpression.cluster(expression_data,min_number_genes=min_number_genes,num_cores=num_cores)
revised_clusters = miner2.coexpression.revise_initial_clusters(initial_clusters,expression_data)

# QC: visualize coexpression clusters

# retrieve first 10 clusters for visual inspection
first_clusters = numpy.hstack([revised_clusters[i] for i in numpy.arange(10).astype(str)])
# visualize first 10 clusters
matplotlib.pyplot.imshow(expression_data.loc[first_clusters,:],aspect="auto",cmap="viridis",vmin=-1,vmax=1)
matplotlib.pyplot.grid(False)
matplotlib.pyplot.ylabel("Genes")
matplotlib.pyplot.xlabel("Samples")
matplotlib.pyplot.title("First 10 coexpression clusters")
figure_name=results_dir+'figures/first.coexpression.clusters.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figure_name)
matplotlib.pyplot.clf()
# visualize 10 random clusters
matplotlib.pyplot.imshow(expression_data.loc[numpy.random.choice(expression_data.index,len(first_clusters),replace=False),:],aspect="auto",cmap="viridis",vmin=-1,vmax=1)
matplotlib.pyplot.grid(False)
matplotlib.pyplot.ylabel("Genes")
matplotlib.pyplot.xlabel("Samples")
matplotlib.pyplot.title("Random coexpression genes")
figure_name=results_dir+'figures/random.coexpression.clusters.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figure_name)
matplotlib.pyplot.clf()

# STEP 2: mechanistic inference
dill.dump_session(results_dir+'info/bottle.dill')
dill.load_session(results_dir+'info/bottle.dill')

# get first principal component axes of clusters
axes = miner2.mechanistic_inference.get_principal_df(revised_clusters,expression_data,subkey=None,min_number_genes=1)

# analyze revised clusters for enrichment in relational database 
mechanistic_output = miner2.mechanistic_inference.enrichment(axes,revised_clusters,expression_data,correlation_threshold=min_correlation,num_cores=num_cores)

# STEP 3: network mapping
dill.dump_session(results_dir+'info/bottle.dill')
dill.load_session(results_dir+'info/bottle.dill')




