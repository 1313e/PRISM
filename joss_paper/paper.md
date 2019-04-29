---
title: 'Model dispersion with PRISM; an alternative to MCMC for rapid analysis of models'
tags:
  - Python 3
  - model analysis
  - emulation
  - history matching
  - Bayesian
  - MCMC
authors:
  - name: Ellert van der Velden
    orcid: 0000-0002-1559-9832
    affiliation: "1, 2"
affiliations:
- name: Centre for Astrophysics and Supercomputing, Swinburne University of Technology, PO Box 218, Hawthorn, VIC 3122, Australia
  index: 1
- name: ARC Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D)
  index: 2
date: XXX
bibliography: paper.bib
---

# Summary

Rapid technological advancements allow for both computational resources and observational/experimental instruments to become better, faster and more precise with every passing year.
This leads to an ever-increasing amount of scientific data being available and more research questions being raised.
As a result, scientific models that attempt to address these questions are becoming more abundant, and are pushing the available resources to the limit as these models incorporate more complex science and more closely resemble reality.

However, as the number of available models increases, they also tend to become more distinct, making it difficult to keep track of their individual qualities.
A full analysis of every model would be required in order to recognize these qualities.
We commonly employ Markov chain Monte Carlo (MCMC) methods and Bayesian statistics for performing this task.
However, as these methods are meant to be used for making approximations of the posterior probability distribution function, we think there is a more efficient way of analyzing them.

Based on the algorithms described by @Vernon10, we have built *PRISM*, a publicly available _Probabilistic Regression Instrument for Simulating Models_ for Python 3.
*PRISM* uses the Bayes linear approach [@BLA], emulation technique [@Craig96, @Craig97] and history matching [@Raftery95, @Craig96, @Craig97] to construct an approximation ('emulator') of any given model, by combining limited model evaluations with advanced regression techniques, covariances and probability calculations.
It is designed to easily facilitate and enhance existing MCMC methods by restricting plausible regions and exploring parameter space more efficiently.
However, *PRISM* can additionally be used as a standalone alternative to MCMC for model analysis, providing insight into the behavior of complex scientific models.
These techniques have been used successfully in the past [@Bower10, @Vernon10, @Vernon18] to speed up model analyses, but their algorithms are typically written for a specific application and are not publicly available.
With *PRISM*, the time spent on evaluating a model is minimized, providing developers with an advanced model analysis for a fraction of the time required by more traditional methods.

The API for *PRISM* was designed to work well for both simple and advanced projects, with every class being written as a base class, but also almost every user-method solely taking optional arguments.
This allows for the user to quickly get started with *PRISM*, while still being able to make adjustments to various routines with minimal effort.
Its ``Pipeline`` class features a user-friendly environment that connects all of *PRISM*'s methods together, whereas the ``ModelLink`` abstract base class helps users wrapping ('linking') their model to *PRISM*.
*PRISM* relies heavily on popular existing Python packages for its expensive computations, like ``NumPy`` [@NumPy], ``Scikit-learn`` [@Sklearn] and ``Mlxtend`` [@Mlxtend], making it more robust and future-proof.

Test applications of *PRISM* (see accompanying AAS submission) show that *PRISM* can provide a qualitative parameter estimation over $15$ times faster than stand-alone MCMC methods, while also being able to give insight into the model's behavior (which MCMC cannot provide).
In future work, *PRISM* will be used together with the MCMC package ``mhysa`` (Mutch et al. in prep.) to analyze and explore the parameter space of the semi-analytic model ``Meraxes`` [@Meraxes].
Also, several smaller application projects with *PRISM* are currently being planned.
The source code for *PRISM* can be found at https://github.com/1313e/PRISM


# Acknowledgements

EV would like to thank Alan Duffy, Darren Croton, Simon Mutch and Manodeep Sinha for being a great help with writing the code.
EV would also like to thank Chris Blake, Colin Jacobs and Theo Steininger for fruitful discussions and valuable suggestions.
Parts of this research were supported by the Australian Research Council Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D), through project number CE170100013.
Parts of this work were performed on the OzSTAR national facility at Swinburne University of Technology. OzSTAR is funded by Swinburne University of Technology and the National Collaborative Research Infrastructure Strategy (NCRIS).


# References

