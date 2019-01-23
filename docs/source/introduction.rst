Introduction
============
Rapid technological advancements allow for both computational resources and observational/experimental instruments to become better, faster and more precise with every passing year.
This leads to an ever-increasing amount of scientific data being available and more research questions being raised.
As a result, scientific models that attempt to address these questions are becoming more abundant, and are pushing the available resources to the limit as these models incorporate more complex science and more closely resemble reality.

However, as the number of available models increases, they also tend to become more distinct, making it difficult to keep track of their individual qualities.
A full analysis of every model would be required in order to recognize these qualities.
It is common to employ Markov chain Monte Carlo (MCMC) methods and Bayesian statistics for performing this task.
However, as these methods are meant to be used for making approximations of the posterior probability distribution function, there must be a more efficient way of analyzing them.

*PRISM* tries to tackle this problem by using the Bayes linear approach, the emulation technique and history matching to construct an approximation ('emulator') of any given model.
The use of these techniques can be seen as special cases of Bayesian statistics, where limited model evaluations are combined with advanced regression techniques, covariances and probability calculations.
*PRISM* is designed to easily facilitate and enhance existing MCMC methods by restricting plausible regions and exploring parameter space efficiently.
However, *PRISM* can additionally be used as a standalone alternative to MCMC for model analysis, providing insight into the bahvior of complex scientific models.
With *PRISM*, the time spent on evaluating a model is minimized, providing developers with an advanced model analysis for a fraction of the time required by more traditional methods.

Why use PRISM?
--------------
- Written in pure Python 2/3, for versatility;
- Stores results in `HDF5-files`_, allowing for easy user-access;
- Can be executed in serial or MPI, on any number of processes;
- Compatible with Windows, Mac OS and Unix-based machines;
- Accepts any type of model and comparison data;
- Built as a plug-and-play tool: all main classes can also be used as base classes;
- Easily linked to any model by writing a single custom ModelLink subclass;
- Capable of reducing relevant parameter space by factors over 100,000 using only a few thousand model evaluations;
- Can be used alone for analyzing models, or combined with MCMC for efficient model parameter estimations.

.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5
