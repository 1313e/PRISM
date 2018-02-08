PRISM, a "Probabilisic Regression Instrument for Simulating Models"
===================================================================
PRISM is a Python 2/3 package that provides an alternative method to MCMC for analyzing models.

Introduction
------------
Typically we probe the universe by making models that try to reconstruct reality based on our scientific knowledge.
Since our knowledge is limited, models tend to only tell part of the story.
Commonly we utilize MCMC methods in order to check how closely this resembles reality.
Although MCMC can precisely return the model realization that does this, it has a few drawbacks: It is slow, requires much additional knowledge about the model for a full Bayesian analysis, is vulnerable to irregularities and its convergence probability vs. speed depends on the initial conditions.
This makes MCMC hard to use for complex models, eliminating the possibility for developers to discover additional details about their model, be it new physics, interesting effects or mistakes.

PRISM tries to tackle this problem by providing a different way for analyzing models.
Instead of evaluating a model millions of times, often in regions of parameter space that do not contain interesting model realizations, PRISM constructs an approximate version of the model with polynomial functions based on a few thousand model evaluations.
By utilizing this system, PRISM is capable of identifying large parts of parameter space as 'implausible' with only limited model knowledge.
Additionally, PRISM will map out the behavior of a model, allowing developers to study its properties.
This makes PRISM an excellent alternative to ordinary MCMC methods for developers that seek to analyze and optimize their models.

What can PRISM do for your model?
---------------------------------
- Rapid analysis of your model, being several magnitudes faster than ordinary MCMC methods;
- Provide an approximated version of the model;
- Analyze and map out its physical behavior;
- Detect inaccuracies and flaws;
- Advise on important missing constraining data;
- Massively reduce relevant parameter space by factors over 10,000, allowing existing MCMC methods to explore and obtain the optimal model realizations much faster.

What characterizes PRISM?
-------------------------
- Written in pure Python 2/3, for versatility;
- Compatible with both Windows and UNIX-based machines;
- Accepts any type of model and comparison data;
- Built as a plug-and-play tool;
- Easily linked to any model by writing a single custom ModelLink class;
- Suited for both simple and advanced projects.


Installation and Usage
======================
PRISM can be easily installed by either pulling the repository and running the setup-file with ``python setup.py install`` or by installing it directly from the PyPI system with ``pip install prism_tool``.
PRISM can now be imported as a package with ``import prism``.

Dependencies
------------
PRISM requires the following non-standard dependencies:

- ``python == 2.7`` or ``python >= 3.4``,
- ``e13tools >= 0.4.2a0``,
- ``h5py >= 2.7.1``,
- ``mlxtend >= 0.9.0``,
- ``sklearn >= 0.19.1``.
