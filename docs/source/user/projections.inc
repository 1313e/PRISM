.. _projections:

Projections
+++++++++++
After having made an emulator of a given model, *PRISM* can show the user the knowledge it has about the behavior of this model by making *projections* of the active parameters in a specific emulator iteration.
These projections are created by the :meth:`~prism.Pipeline.project` method, which has many different properties and options.
For showing them below, the same emulator as the one in :ref:`minimal_example` is used.

Properties
##########
Projections (and their figures) are made by analyzing a large set of evaluations samples.
For 3D projections, this set is made up of a grid of :attr:`~prism.Pipeline.proj_res` x :attr:`~prism.Pipeline.proj_res` samples for the plotted (active) parameters, where the values for the remaining parameters in every individual grid point are given by an LHD of :attr:`~prism.Pipeline.proj_depth` samples.
This gives the total number of analyzed samples as :attr:`~prism.Pipeline.proj_res` x :attr:`~prism.Pipeline.proj_res` x :attr:`~prism.Pipeline.proj_depth`.

Every sample in the sample set is then analyzed in the emulator, saving whether or not this sample is plausible and what the implausibility value at the first cut-off is (the first value in :attr:`~prism.Pipeline.impl_cut`).
This yields :attr:`~prism.Pipeline.proj_depth` results per grid point, which can be used to determine the fraction of samples that is plausible and the minimum implausibility value at the first cut-off in this point.
Doing this for the entire grid and interpolating them, creates a map of results that is independent of the values of the non-plotted parameters.
For 2D projections, it works the same way, except that only a single active parameter is plotted.

.. note::
    When using a 2D model, the *projection depth* used to make a 2D projection will be :attr:`~prism.Pipeline.proj_depth`, which is to be expected.
    However, when using an nD model, the projection depth of a 2D projection is equal to :attr:`~prism.Pipeline.proj_res` x :attr:`~prism.Pipeline.proj_depth`.
    This is to make sure that for an nD model, the density of samples in a 2D projection is the same as in a 3D projection.

The :meth:`~prism.Pipeline.project` method solely takes optional arguments.
Calling it without any arguments will produce six projection figures: three 2D projections and three 3D projections.
One of each type is shown below.

.. figure:: images/1_proj__A1_.png
    :alt: 2D projection figure of model parameter A1.
    :width: 100%
    :align: center
    :name: 1_proj_A1

    2D projection figure of model parameter :math:`A_1`.
    The vertical dashed line shows the parameter estimate of :math:`A_1`, whereas the horizontal red line shows the first implausibility cut-off value.

.. figure:: images/1_proj__A1-B1_.png
    :alt: 3D projection figure of model parameters A1 and B1.
    :width: 100%
    :align: center
    :name: 1_proj_A1-B1

    3D projection figure of model parameters :math:`A_1` and :math:`B_1`.
    The dashed lines show the estimates of both parameters.

A projection figure is made up of two subplots.
The upper subplot shows a map of minimum implausibility values that can be reached for any given value (combination) of the plotted parameter(s).
The lower subplot gives a map of the fraction of samples that is plausible in a specified point on the grid (called "line-of-sight depth" due to the way it is calculated).
Another way of describing this map is that it gives the probability that a parameter set with given plotted value(s) is plausible.

Both projection types have a different purpose.
A 3D projection gives insight into what the dependencies (or correlations) are between the two plotted parameters, by showing where the best (top) and most (bottom) plausible samples can be found.
On the other hand, a 2D projection is quite similar in meaning to a *maximum likelihood* optimization performed by MCMC methods, with the difference being that the projection is based on expectations rather than real model output.
A combination of both subplots allows for many model properties to be derived, especially when they do not agree with each other.

Options
#######
The :meth:`~prism.Pipeline.project` method takes two (optional) arguments, `emul_i` and `proj_par`.
The first controls which emulator iteration should be used, while the latter provides the model parameters of which projections need to be made.
Since it only makes sense to make projections of active parameters, all passive parameters are filtered out of `proj_par`.
The remaining parameters are then used to determine which projections are required (which also depends on the requested projection types).
For example, if one wishes to only obtain projections of the :math:`A_1` and :math:`B_1` parameters (which are both active) in iteration 1, then this can be done with::

    >>> pipe.project(1, ('A1', 'B1'))

This would generate the figures shown above, as well as the 2D projection figure of :math:`B_1`.
By default, the last constructed emulator iteration and all model parameters are requested.

The remaining input arguments can only be given as keyword arguments, since they control many different aspects of the :meth:`~prism.Pipeline.project` method.
The `proj_type` argument controls which projection types to make.
For 2D models, this is always `'2D'` and cannot be modified.
However, for nD models, this can be set to `'2D'` (only 2D projections), `'3D'` (only 3D projections) or `'both'` (both 2D and 3D projections).
By default, it is set to `'both'`.

The `figure` argument is a bool, that determines whether or not the projection figures should be created after calculating the projection data.
If :pycode:`True`, the projection figures will be created and saved, which is done by default.
If :pycode:`False`, the data that is contained within the projection figures will be calculated and returned in a dict.
This allows the user to either let *PRISM* create the projection figures using the standard template or create the figures themselves.

The `align` argument controls the alignment of the subplots in every projection figure.
By default, it aligns the subplots in a column (`'col'`), as shown in the figures above.
Aligning the subplots in a row (`'row'`) would give :numref:`1_proj_A1` as the figure below.

.. figure:: images/1_proj__A1__row.png
    :alt: 2D projection figure of model parameter A1 with the 'row' alignment.
    :width: 100%
    :align: center
    :name: 1_proj_A1_row

    2D projection figure of model parameter :math:`A_1` with the `'row'` alignment.

The `smooth` argument is also a bool, that determines what to do if a grid point in the projection figure contains no plausible samples, but does contain a minimum implausibility value below the first non-wildcard cut-off.
If :pycode:`False`, which is the default, these values are kept in the figure, which may show up as artifact-like features.
If :pycode:`True`, these values are set to the first cut-off, basically removing them from the projection figure.
This may however also remove interesting features.
Below are two identical projections, one that is smoothed and one that is not, to showcase this difference (these projections are from the second iteration, since this effect rarely occurs in the first iteration).

.. figure:: images/2_proj__A1-B1_.png
    :alt: Non-smoothed 3D projection figure of model parameters A1 and B1.
    :width: 100%
    :align: center
    :name: 2_proj_A1-B1

    Non-smoothed 3D projection figure of model parameters :math:`A_1` and :math:`B_1`.

.. figure:: images/2_proj__A1-B1__s.png
    :alt: Smoothed 3D projection figure of model parameters A1 and B1.
    :width: 100%
    :align: center
    :name: 2_proj_A1-B1_s

    Smoothed 3D projection figure of model parameters :math:`A_1` and :math:`B_1`.

In these figures, one can see that the non-smoothed projection shows many features in the upper subplot that look like artifacts.
These features are however not artifacts, but caused by a sample (or samples) having its highest implausibility value being below the first implausibility cut-off, but still being implausible due to failing a later cut-off.
For example, if the implausibility cut-offs are :pycode:`[4.0, 3.7, 3.5]` and a sample has implausibility values :pycode:`[3.9, 3.8, 3.2]`, it is found implausible due to failing to meet the second cut-off.
However, since the first value is still the highest implausibility value, that value is used in the projection figure.
Smoothing figures usually allows for 3D projections (2D projections rarely show this) to become less crowded, but they do throw away information.
It should therefore only be used when necessary.

The `force` argument is a bool, which controls what to do if a projection is requested for which data already exists.
If :pycode:`False` (default), it will use the previously acquired projection data to create the projection figure if it does not exist, skip if it does or return the figure data if `figure` is :pycode:`False`.
If :pycode:`True`, the projection data and all associated projection figures will be deleted, and the projection will be recalculated.

The remaining six arguments are keyword argument dicts, that need to be passed to the various different plotting functions that are used for creating the projection figures.
The `fig_kwargs` dict is passed to the :func:`~matplotlib.pyplot.figure` function when creating the projection figure instance.
The `impl_kwargs_2D` and `los_kwargs_2D` dicts are passed to the :func:`~matplotlib.pyplot.plot` function when making the minimum implausibility and line-of-sight depth subplots, respectively, for the 2D projections.
Similarly, the `impl_kwargs_3D` and `los_kwargs_3D` dicts are passed to the :func:`~matplotlib.pyplot.hexbin` function for 3D projections.
And, finally, the `line_kwargs` dict is passed to the :func:`~matplotlib.pyplot.draw` function for drawing the parameter estimate lines.

