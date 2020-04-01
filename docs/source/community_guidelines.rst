.. _community_guidelines:

Community guidelines
====================
*PRISM* is an open-source and free-to-use software package (and it always will be), provided under the BSD-3 license (see below for the full license).

Users are highly encouraged to make contributions to the package or request new features by opening a `GitHub issue`_.
If you would like to contribute to the package, but do not know what, then there are quite a few ToDos in the code that may give you some inspiration.
As with contributions, if you find a problem or issue with *PRISM*, please do not hesitate to open a `GitHub issue`_ about it or post it on `Gitter`_.

And, finally, if you use *PRISM* as part of your workflow in a scientific publication, please consider including an acknowledgement like *"Parts of the results in this work were derived using the PRISM Python package."* and citing the *PRISM* pipeline paper using the BibTeX-entry below.

.. _GitHub issue: https://github.com/1313e/PRISM/issues
.. _Gitter: https://gitter.im/1313e/PRISM


License
-------
.. literalinclude:: ../../LICENSE
    :language: none


Citation
--------
This BibTeX-entry is also available in *PRISM* using :func:`~prism._internal.get_bibtex` (available as :pycode:`prism.get_bibtex()`).

.. code-block:: none

    @ARTICLE{2019ApJS..242...22V,
           author = {{van der Velden}, Ellert and {Duffy}, Alan R. and {Croton}, Darren and
             {Mutch}, Simon J. and {Sinha}, Manodeep},
            title = "{Model Dispersion with PRISM: An Alternative to MCMC for Rapid Analysis of Models}",
          journal = {\apjs},
         keywords = {methods: data analysis, methods: numerical,
             Astrophysics - Instrumentation and Methods for Astrophysics, Physics - Computational Physics},
             year = 2019,
            month = jun,
           volume = {242},
           number = {2},
              eid = {22},
            pages = {22},
              doi = {10.3847/1538-4365/ab1f7d},
    archivePrefix = {arXiv},
           eprint = {1901.08725},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJS..242...22V},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


Additions
---------
.. include:: additions.inc
