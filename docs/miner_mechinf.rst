The miner-mechinf tool
======================

This utility computes the mechanistic inference.

You can see the tool's available options when you enter ``miner-mechinf -h``
at the command prompt:

.. highlight:: none

::

    usage: miner-mechinf [-h] [-mc MINCORR]
                         expfile mapfile coexprdict datadir outdir

    miner-mechinf - MINER compute mechanistic inference

    positional arguments:
      expfile               input matrix
      mapfile               identifier mapping file
      coexprdict            coexpressionDictionary.json file from miner-coexpr
      datadir               data directory
      outdir                output directory

    optional arguments:
      -h, --help            show this help message and exit
      -mc MINCORR, --mincorr MINCORR
                            minimum correlation


Parameters in detail
--------------------

``miner-mechinf`` expects at least these 5 arguments:

  * **expfile:** The gene expression file a matrix in csv format.
  * **mapfile:** The gene identifier map file.
  * **coexprdict:** The path coexpressionDictionary.json file from the miner-coexpr tool
  * **datadir:** The path to the data directory
  * **outdir:** The path to the output directory

In addition, you can specify the following optional arguments:

  * ``--mincorr`` or ``--mc``: the minimum correlation value.

Output in detail
----------------

After successful completion there will be the following files in the output directory


  * ``regulons.json`` - use this file in subsequent tools
  * ``coexpressionDictionary_annotated.json``
  * ``mechanisticOutput.json``
  * ``coexpressionModules_annotated.json``
  * ``regulons_annotated.csv``
  * ``coexpressionModules.json``
  * ``regulons_annotated.json``
  * ``coregulationModules.json``
