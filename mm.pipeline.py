import sys,os,dill,numpy
import matplotlib,matplotlib.pyplot

matplotlib.rcParams.update({'font.size':18,'font.family':'Arial','xtick.labelsize':14,'ytick.labelsize':14})
matplotlib.rcParams['pdf.fonttype']=42

import miner2
import miner2.preprocess
import miner2.coexpression

# 0.0. user defined variables
expressionFile='/Volumes/omics4tb2/alomana/projects/PSL/MM/data/IA12Zscore.csv'
resultsDir='/Volumes/omics4tb2/alomana/projects/PSL/MM/results/'

numCores = 8         # required for coexpression
minNumberGenes = 6   # required for coexpression

# 0.1. build results directory tree
if os.path.exists(resultsDir) == False:
    os.mkdir(resultsDir)
    os.mkdir(resultsDir+'figures')
    os.mkdir(resultsDir+'info')

# STEP 0: load the data
expressionData, conversionTable = miner2.preprocess.main(expressionFile)

"""
individual_expression_data = [expressionData.iloc[:,i] for i in range(50)]
matplotlib.pyplot.boxplot(individual_expression_data)
matplotlib.pyplot.title("Patient expression profiles")
matplotlib.pyplot.ylabel("Relative expression")
matplotlib.pyplot.xlabel("Sample ID")
matplotlib.pyplot.xticks(fontsize=6)

figureName=resultsDir+'figures/boxplots.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figureName)
matplotlib.pyplot.clf()

matplotlib.pyplot.hist(expressionData.iloc[0,:],bins=100,alpha=0.75)
matplotlib.pyplot.title("Expression of single gene")
matplotlib.pyplot.ylabel("Frequency")
matplotlib.pyplot.xlabel("Relative expression")

figureName=resultsDir+'figures/singleGene.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figureName)
matplotlib.pyplot.clf()

matplotlib.pyplot.hist(expressionData.iloc[:,0],bins=200,color=[0,0.4,0.8],alpha=0.75)
matplotlib.pyplot.ylim(0,350)
matplotlib.pyplot.title("Expression of single patient sample",FontSize=14)
matplotlib.pyplot.ylabel("Frequency")
matplotlib.pyplot.xlabel("Relative expression")

figureName=resultsDir+'figures/singlePatient.pdf'
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig(figureName)
matplotlib.pyplot.clf()
"""

#dill.dump_session(resultsDir+'info/bottle.dill')
    
# STEP 1: clustering
#dill.load_session(resultsDir+'info/bottle.dill')

initialClusters = miner2.coexpression.cluster(expressionData,minNumberGenes=minNumberGenes,numCores=numCores)

#dill.dump_session(resultsDir+'info/bottle.dill')
