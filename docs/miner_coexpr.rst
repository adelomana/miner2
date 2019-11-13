The miner2-coexpr tool
======================

This utility generates revised coexpression clusters from a gene expression
file.

You can see the tool's available options when you enter ``miner2-coexpr -h``
at the command prompt:

.. highlight:: none

::


    usage: miner2-coexpr [-h] [-mg MINGENES] [-moxs MINOVEREXPSAMP]
                         [-mx MAXEXCLUSION] [-rs RANDSTATE] [-oxt OVEREXPTHRESH]
                         expfile mapfile outdir

    miner2-coexpr - MINER cluster expression data.

    positional arguments:
      expfile               input matrix
      mapfile               identifier mapping file
      outdir                output directory

    optional arguments:
      -h, --help            show this help message and exit
      -mg MINGENES, --mingenes MINGENES
                            min number genes
      -moxs MINOVEREXPSAMP, --minoverexpsamp MINOVEREXPSAMP
                            minimum overexpression samples
      -mx MAXEXCLUSION, --maxexclusion MAXEXCLUSION
                            maximum samples excluded
      -rs RANDSTATE, --randstate RANDSTATE
                            random state
      -oxt OVEREXPTHRESH, --overexpthresh OVEREXPTHRESH
                            overexpression threshold


Parameters in detail
--------------------

``miner2-coexpr`` expects at least these 3 arguments:

  * **expfile:** The gene expression file a matrix in csv format.
  * **mapfile:** The gene identifier map file.
  * **outdir:** The path to the output directory

In addition, you can specify the following optional arguments:

  * ``--mingenes`` or ``--mg``: the minimum number of genes in a cluster.
  * ``--minoverexpsamp`` or ``--moxs``: minimum number of overexpression samples
  * ``--maxexclusion`` or ``-mx``: maximum exclusion
  * ``--randstate`` or ``-rs``: random state
  * ``--overexpthresh`` or ``-oxt``: overexpression threshold


Output in detail
----------------

After successful completion there will be the following files in the output directory

  * ``cluster_size_distribution.pdf``: a plot of the cluster size distribution for visual verification
  * ``coexpressionDictionary.json``: this file is used in subsequent stages
