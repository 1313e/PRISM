.. _hybrid_sampling:

Hybrid sampling
+++++++++++++++
A common problem when using MCMC methods is that it can often take a very long time for MCMC to find its way on the posterior probability distribution function, which is often referred to as the *burn-in phase*.
This is because, when considering a parameter set, there is usually no prior information that this parameter set is (un)likely to result into a desirable model realization.
This means that such a parameter set must first be evaluated in the model before any probabilities can be calculated.
However, by constructing an emulator of the model, one can use it as an additional prior for the posterior probability calculation.
Therefore, although *PRISM* is primarily designed to make analyzing models much more efficient and accessible than normal MCMC methods, it is also very capable of enhancing them.
This process is called *hybrid sampling*, which can be performed easily with the :mod:`~prism.utils` module and will be explained below.
Note that an interactive version of this section can be found in `the tutorials`_.

.. _the tutorials: https://github.com/1313e/PRISM/tree/master/tutorials/4_hybrid_sampling.ipynb

Algorithm
#########
Hybrid sampling allows one to use *PRISM* to first analyze a model's behavior, and later use the gathered information to speed up parameter estimations (by using the emulator as an additional prior in a Bayesian analysis).
Hybrid sampling works in the following way:

    1. Whenever an MCMC walker proposes a new sample, it is first passed to the emulator of the model;
    2. If the sample is not within the defined parameter space, it automatically receives a prior probability of zero (or :math:`-\infty` in case of logarithmic probabilities).
       Else, it will be evaluated in the emulator;
    3. If the sample is labeled as *implausible* by the emulator, it also receives a prior probability of zero.
       If it is plausible, the sample is evaluated in the same way as for normal sampling;
    4. Optionally, a scaled value of the first implausibility cut-off is used as an exploratory method by adding an additional (non-zero) prior probability.
       This can be enabled by using the `impl_prior` input argument for the :func:`~prism.utils.get_hybrid_lnpost_fn` function.

Since the emulator that *PRISM* makes of a model is not defined outside of the parameter space given by :attr:`~prism.modellink.ModelLink.par_rng`, the second step is necessary to make sure the results are valid.
There are several advantages of using hybrid sampling over normal sampling:

    - Acceptable samples are guaranteed to be within plausible space;
    - This in turn makes sure that the model is only evaluated for plausible samples, which heavily reduces the number of required evaluations;
    - No burn-in phase is required, as the starting positions of the MCMC walkers are chosen to be in plausible space;
    - As a consequence, varying the number of walkers tends to have a much lower negative impact on the convergence probability and speed;
    - Samples with low implausibility values can optionally be favored.

Usage
#####
In order to help the user with combining *PRISM* with MCMC to use hybrid sampling, the :mod:`~prism.utils` module provides two functions: :func:`~prism.utils.get_walkers` and :func:`~prism.utils.get_hybrid_lnpost_fn`.
The :func:`~prism.utils.get_walkers` function analyzes a set of proposed `init_walkers` and returns the positions that are plausible (and the number of positions that are plausible).
By default, it uses the available :attr:`~prism.Pipeline.impl_sam` of the last constructed iteration, but it can also be supplied with a custom set of proposed walkers or an integer stating how many proposed positions the function should check::

    >>> # Use impl_sam if it is available
    >>> n, p0 = get_walkers(pipe)

    >>> # Request 2000 proposed samples
    >>> n_walkers = 2000
    >>> n, p0 = get_walkers(pipe, init_walkers=n_walkers)

    >>> # Use custom init_walkers
    >>> from e13tools.sampling import lhd
    >>> init_walkers = lhd(n_walkers, pipe.modellink.n_par, pipe.modellink.par_rng)
    >>> n, p0 = get_walkers(pipe, init_walkers=init_walkers)

As *PRISM*'s sampling methods operate in parameter space, the :func:`~prism.utils.get_walkers` function automatically assumes that all starting positions are defined in parameter space.
However, as some sampling methods use unit space, normalized starting positions can be requested by setting the `unit_space` input argument to :pycode:`True`.
One has to keep in mind that, because of the way the emulator works, there is no guarantee for a specific number of plausible starting positions to be obtained.
Having the desired emulator iteration already analyzed may give an indication how many starting positions in total need to be proposed to be left with a specific number.

When the initial positions of the MCMC walkers have been determined, one can use them in an MCMC parameter estimation algorithm, avoiding the burn-in phase.
This in itself can already be very useful, but it does not allow for hybrid sampling yet.
Most MCMC methods require the definition of an `lnpost()` function, which takes a parameter set and returns the corresponding natural logarithm of the posterior probability.
In order to do hybrid sampling, this `lnpost()` function must have the algorithm described above implemented.

The :func:`~prism.utils.get_hybrid_lnpost_fn` function factory provides exactly that.
It takes a user-defined `lnpost()` function (as `lnpost_fn`) and a :obj:`~prism.Pipeline` object, and returns a function definition :pycode:`hybrid_lnpost(par_set, *args, **kwargs)`.
This `hybrid_lnpost()` function first analyzes a proposed `par_set` in the emulator, passes `par_set` (along with any additional arguments) to `lnpost()` if the sample is plausible, or returns :math:`-\infty` if it is not.
The return-value of the `lnpost()` function is then returned by the `hybrid_lnpost()` function as well.
To make sure that the `hybrid_lnpost()` function can be used in both execution modes (see :ref:`dual_nature`), all parallel calls to the :obj:`~prism.Pipeline` object are done with the :meth:`~prism.Pipeline._make_call` method.

The use of a function factory here allows for all input arguments to be validated once and then saved as local variables for the `hybrid_lnpost()` function.
Not only does this avoid that all arguments have to be provided and validated for every individual call, but it also ensures that the same arguments are used every time, as local variables of a function cannot be modified by anything.
Since users most likely use :func:`~prism.utils.get_walkers` and :func:`~prism.utils.get_hybrid_lnpost_fn` frequently together, the :func:`~prism.utils.get_walkers` function allows for the `lnpost_fn` argument to be supplied to it.
This will automatically call the :func:`~prism.utils.get_hybrid_lnpost_fn` function factory using the provided `lnpost_fn` and the same input arguments given to :func:`~prism.utils.get_walkers`, and return the obtained `hybrid_lnpost()` function in addition to the starting positions of the MCMC walkers.

Application
###########
Using the information above, using hybrid sampling on a model of choice can be done quite easily.
For performing the MCMC analysis, we will be using the `emcee`_ package in this example.

Assume that we want to first analyze and then optimize the Gaussian model given by the :class:`~prism.modellink.GaussianLink` class.
So, we first have to make an emulator of the model::

    >>> from prism import Pipeline
    >>> from prism.modellink import GaussianLink
    >>> model_data = {3: [3.0, 0.1], 5: [5.0, 0.1], 7: [3.0, 0.1]}
    >>> modellink_obj = GaussianLink(model_data=model_data)
    >>> pipe = Pipeline(modellink_obj)
    >>> pipe.construct()

Using the constructed emulator, we can perform a model parameter optimization using hybrid sampling.
For this, we need to define an `lnpost()` function, for which we will use a simple Gaussian probability function::

    def lnpost(par_set, pipe):
        # Create parameter dict for call_model
        par_dict = dict(zip(pipe.modellink.par_name, par_set))

        # Use wrapped model to obtain model output
        mod_out = pipe.modellink.call_model(pipe.emulator.emul_i,
                                            par_dict,
                                            pipe.modellink.data_idx)

        # Get the model and data variances
        # Since the value space is linear, the data error is centered
        md_var = pipe.modellink.get_md_var(pipe.emulator.emul_i,
                                           par_dict,
                                           pipe.modellink.data_idx)
        data_var = [err[0]**2 for err in pipe.modellink.data_err]

        # Calculate the posterior probability and return it
        sigma_2 = md_var+data_var
        diff = pipe.modellink.data_val-mod_out
        return(-0.5*(np.sum(diff**2/sigma2)))

Since the :obj:`~prism.Pipeline` object already has the model wrapped and linked, we used that to evaluate the model.
The :class:`~prism.modellink.GaussianLink` class has a centered data error, therefore we can take the upper bound for every error when calculating the variance.
However, for more complex models, this is probably not true.

Next, we have to obtain the starting positions for the MCMC walkers.
Since we want to do hybrid sampling, we can obtain the `hybrid_lnpost()` function at the same time as well::

    >>> from prism.utils import get_walkers
    >>> n, p0, hybrid_lnpost = get_walkers(pipe, unit_space=False,
                                           lnpost_fn=lnpost, impl_prior=True)

By setting `impl_prior` to :pycode:`True`, we use the implausibility cut-off value as an additional prior.
Now we only still need the :class:`~emcee.EnsembleSampler` class and NumPy (for the `lnpost()` function)::

    >>> import numpy as np
    >>> from emcee import EnsembleSampler

Now we have everything that is required to perform a hybrid sampling analysis.
In most cases, MCMC methods require to be executed on only a single MPI rank, so we will use the :attr:`~prism.Pipeline.worker_mode`::

    # Activate worker mode
    with pipe.worker_mode:
        if pipe.is_controller:
            # Create EnsembleSampler object
            sampler = EnsembleSampler(n, pipe.modellink.n_par,
                                      hybrid_lnpost, args=[pipe])

            # Run mcmc for 1000 iterations
            sampler.run_mcmc(p0, 1000)

            # Execute any custom operations here
            # For example, saving the chain data or plotting the results

And that is basically all that is required for using *PRISM* together with MCMC.
For a normal MCMC approach, the same code can be used, except that one has to use `lnpost()` instead of `hybrid_lnpost()` (and, obtain the starting positions of the walkers in a different way).

.. _emcee: http://dfm.io/emcee/current
