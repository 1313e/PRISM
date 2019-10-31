Model dispersion with *PRISM*; an alternative to MCMC for rapid analysis of models
==================================================================================

This is the documentation for the *PRISM* package, an efficient and rapid alternative to MCMC methods for optimizing and analyzing scientific models.
*PRISM* was made by **Ellert van der Velden** (`@1313e`_) as part of a Ph.D under supervision of `A/Prof. Alan Duffy`_ at `Swinburne University of Technology`_.
It is written in pure `Python 3`_ and `publicly available on GitHub`_.

.. _@1313e: https://github.com/1313e
.. _A/Prof. Alan Duffy: https://www.alanrduffy.com
.. _Swinburne University of Technology: https://www.swinburne.edu.au
.. _Python 3: https://www.python.org
.. _publicly available on GitHub: https://github.com/1313e/PRISM

----

The documentation of *PRISM* is spread out over several sections:

* :ref:`user-docs`
* :ref:`api-docs`

.. _user-docs:

.. toctree::
    :maxdepth: 3
    :caption: User Documentation
   
    introduction
    user/getting_started
    user/prism_pipeline
    user/modellink_crash_course
    user/using_prism
    user/descriptions
    user/FAQ
    community_guidelines
   
.. _api-docs:

.. toctree::
    :maxdepth: 4
    :caption: API Reference
   
    api/prism._pipeline
    api/prism.emulator
    api/prism.modellink
    api/prism.utils
    api/prism._gui
    api/prism._internal

.. role:: pycode(code)
    :language: python3
    :class: highlight

----

Acknowledgements
----------------
Special thanks to Alan Duffy, Darren Croton, Simon Mutch and Manodeep Sinha for providing many valuable suggestions and constructive feedback points.
Huge thanks to James Josephides for making the *PRISM* logo.
