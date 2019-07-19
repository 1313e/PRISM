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
However, *PRISM* can additionally be used as a standalone alternative to MCMC for model analysis, providing insight into the behavior of complex scientific models.
With *PRISM*, the time spent on evaluating a model is minimized, providing developers with an advanced model analysis for a fraction of the time required by more traditional methods.

Why use PRISM?
--------------
- Written in pure Python 3, for versatility;
- Stores results in `HDF5-files`_, allowing for easy user-access;
- Can be executed in serial or MPI, on any number of processes;
- Compatible with Windows, Mac OS and Unix-based machines;
- Accepts any type of model and comparison data;
- Built as a plug-and-play tool: all main classes can also be used as base classes;
- Easily linked to any model by writing a single custom :class:`~prism.modellink.ModelLink` subclass (see :ref:`modellink_crash_course`);
- Capable of reducing relevant parameter space by factors over 100,000 using only a few thousand model evaluations;
- Can be used alone for analyzing models, or combined with MCMC for efficient model parameter estimations.

When (not) to use PRISM?
--------------------------
It may look very tempting to use *PRISM* for basically everything, but keep in mind that emulation has its limits.
Below is a general (but non-exhaustive) list of scenarios where *PRISM* can become really valuable:

- In almost any situation where one wishes to perform a parameter estimation using an MCMC Bayesian analysis (by using :ref:`hybrid_sampling`).
  This is especially true for poorly constrained models (low number of available observational constraints);
- Whenever one wishes to visualize the correlation behavior between different model parameters;
- For quickly exploring the parameter space of a model without performing a full parameter estimation.
  This can be very useful when trying out different sets of observational data to study their constraining power;
- For obtaining a reasonably accurate approximation of a model in very close proximity to the most optimal parameter set.

There are however also situations where one is better off using a different technique, with a general non-exhaustive list below:

- For obtaining a reasonably accurate approximation of a model in all of parameter space.
  Due to the way an emulator is constructed, this could easily require millions of model evaluations and a lot of time and memory;
- When dealing with a model that has a large number of parameters/degrees-of-freedom (>50).
  This however still heavily depends on the type of model that is used;
- Whenever a very large number of observational constraints are available and one wishes to use all of them (unless one also has access to a large supercomputer).
  In this case, it is a better idea to use full Bayesian instead;
- One wishes to obtain the posterior probability distribution function (PDF) of a model.

A very general and easy way to check if one should use *PRISM*, is to ask oneself the question: *"Would I use a full Bayesian analysis for this problem, given the required time and resources?"*.
If the answer is 'yes', then *PRISM* is probably a good choice, especially as it requires near-similar resources as a Bayesian analysis does (definition of parameter space; provided comparison data; and a way to evaluate the model).

.. _HDF5-files: https://portal.hdfgroup.org/display/HDF5/HDF5
