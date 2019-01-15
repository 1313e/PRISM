.. _md_var:

Model discrepancy variance (`md_var`)
+++++++++++++++++++++++++++++++++++++
Of the three different variances that are used for calculating the implausibility values of a parameter set, the *model discrepancy variance* is by far the most important.
The model discrepancy variance describes all uncertainty about the correctness of the model output that is caused by the model itself.
This includes the accuracy of the code implementation, completeness of the inclusion of the involved physics, made assumptions and the accuracy of the output itself, amongst others.
It therefore acts as a measure of the quality of the model that is being emulated by *PRISM*, and as with :meth:`~prism.modellink.ModelLink.call_model`, must be handled with great care.

Theory
######
When *PRISM* constructs an emulator, it attempts to make a perfect approximation of the model that covers the absolute plausible regions of parameter space.
This perfect approximation would be reached if the adjusted emulator variance (`adj_var`) is zero for all samples.
In this case, the emulator has the same variance associated with it as the model, which is given by the model discrepancy variance.
Therefore, if the model discrepancy variance is determined incorrectly, the emulator itself will be incorrect as well.

The reason for this is as follows.
The implausibility value of a parameter set states how many standard deviations the emulator system expects the model realization corresponding to this parameter set, to be away from explaining the model comparison data.
When the total variance increases, the implausibility value decreases (since less standard deviations fit in the total difference).
For an emulator system that is still very inaccurate (e.g., first iteration), the adjusted emulator variance dominates over the other two variances.
However, later on, the adjusted emulator variance becomes less and less dominant, causing the other two variances to start playing a role.
In most cases, it is safe to assume that the model discrepancy variance is higher than the observational variance, since a model would be fitting noise if this was not the case.
Therefore, there is going to be a moment when the model discrepancy variance starts being close to the adjusted emulator variance.

When this happens, the plausible region of parameter space starts being determined by the model discrepancy variance.
If the model discrepancy variance is generally higher than it should be, then this will often result into the emulator system not converging as far as it could have, since parts of parameter space are still marked as plausible.
The opposite however (the model discrepancy variance generally being lower than it should be) can mark parts of parameter space as implausible while they are not.
This means that these parts are removed from the emulator.

From the above, it becomes clear that overestimating the model discrepancy variance is much less costly than underestimating its value.
It is therefore important that this variance is properly described at all times.
However, since the description of the model discrepancy variance can take a large amount of time, *PRISM* uses its own default description in case none was provided, which is defined as :math:`\mathrm{Var}(\epsilon_{\mathrm{md}, i})=\left(z_i/6\right)^2`, where :math:`\mathrm{Var}(\epsilon_{\mathrm{md}, i})` is the model discrepancy variance of a specified model comparison data point :math:`i` and :math:`z_i` is the corresponding data value.
If one assumes that a model output within half of the data is considered to be acceptable, with acceptable being defined as the :math:`3\sigma`-interval, then the model discrepancy variance is obtained as:

.. math::

    \left[z_i-3\sigma, z_i+3\sigma\right] &= \left[\frac{1}{2}z_i, \frac{3}{2}z_i\right],\\
    6\sigma &= z_i,\\
    \sigma &= \frac{z_i}{6},\\
    \mathrm{Var}(\epsilon_{\mathrm{md}, i}) &= \sigma^2 = \left(\frac{z_i}{6}\right)^2.

This description of the model discrepancy variance usually works well for simple models, and acts as a starting point within *PRISM*.
When models become bigger and more complex, it is likely that such a description is not enough.
Given that the model discrepancy variance is unique to every model and might even be different for every model output, *PRISM* cannot possibly cover all scenarios.
It is therefore advised that the model discrepancy variance is provided externally by the user.

Implementation
##############
The model discrepancy variance is given by the :meth:`~prism.modellink.ModelLink.get_md_var` method.
This method is, like :meth:`~prism.modellink.ModelLink.call_model`, an abstract method and must be overridden by the :class:`~prism.modellink.ModelLink` subclass before it can be initialized.
The :meth:`~prism.modellink.ModelLink.get_md_var` method is called every time the implausibility value of an emulator evaluation sample is determined.
Unlike the :meth:`~prism.modellink.ModelLink.call_model` method, the :meth:`~prism.modellink.ModelLink.get_md_var` method is called by individual emulator systems, as they determine implausibility values individually.

For this reason, the :meth:`~prism.modellink.ModelLink.get_md_var` method is provided with the emulator iteration :pycode:`emul_i`, a single parameter set :pycode:`par_set` and the data identifiers requested by the emulator system :pycode:`data_idx`.
The :attr:`~prism.modellink.ModelLink.call_type` and :attr:`~prism.modellink.ModelLink.MPI_call` flags have no influence on the way the :meth:`~prism.modellink.ModelLink.get_md_var` method is used, as it is always called in serial for a single parameter set.
When it is called, *PRISM* expects an array-like container back with shape :pycode:`(n_data)` (if :math:`1\sigma`-interval is centered) or shape :pycode:`(n_data, 2)` (if :math:`1\sigma`-interval is given by upper and lower errors).
The default model discrepancy variance description given above is used if the :meth:`~prism.modellink.ModelLink.get_md_var` method raises a :pycode:`NotImplementedError`, but this is discouraged.

.. warning::
    Because the :meth:`~prism.modellink.ModelLink.get_md_var` method is always called for single parameter sets, it is important that it can be called without requiring any preparation of data or models.

