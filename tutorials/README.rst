PRISM tutorials
===============
This directory contains a series of Jupyter notebooks that serve as tutorials for the *PRISM* package.

Getting started
---------------
In order to use these tutorials, one first has to make sure that *PRISM* is already installed.
After this, one has to clone the `repository`_, install the `requirements_tutorials`_ and launch Jupyter::

    $ git clone https://github.com/1313e/PRISM
    $ cd PRISM/tutorials
    $ pip install -r requirements_tutorials.txt
    $ jupyter lab

If *PRISM* was installed by cloning the repository instead of using ``pip``, one can skip the first step and directly go to the directory containing the tutorials.

.. _repository: https://github.com/1313e/PRISM
.. _requirements_tutorials: https://github.com/1313e/PRISM/raw/master/tutorials/requirements_tutorials.txt

Available tutorials
-------------------
Below is a list of all tutorials that are available:

1. `Basic usage <1_basic_usage.ipynb>`_: Basic overview of the user-methods in *PRISM* and how to use the ``Pipeline`` class;
2. `ModelLink subclasses <2_modellink_subclasses.ipynb>`_: Introduction to the ``ModelLink`` abstract base class and how to write a subclass;
3. `PRISM class properties <3_class_properties.ipynb>`_: Overview of the various different class properties in *PRISM* and how to use them;
4. `Hybrid sampling <4_hybrid_sampling.ipynb>`_: Introduction to the concept of hybrid sampling and how to use it with ``emcee``.
