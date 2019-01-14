.. _call_model:

Wrapping a model (`call_model`)
+++++++++++++++++++++++++++++++
The :meth:`~prism.modellink.ModelLink.call_model` method is the most important method in the entire *PRISM* package.
It provides the :obj:`~prism.Pipeline` instance with a way to call the model that is wrapped in the user-defined :class:`~prism.modellink.ModelLink` subclass.
For *PRISM*, this method is a black box: it takes a parameter/sample set, performs a series of unknown operations and returns the values corresponding to the requested data points and sample(s).
Therefore, the :meth:`~prism.modellink.ModelLink.call_model` method must be written with great care.

Input arguments
###############
Depending on the values of the :attr:`~prism.modellink.ModelLink.multi_call` and :attr:`~prism.modellink.ModelLink.MPI_call` flags (where the first is set by the :attr:`~prism.modellink.ModelLink.call_type` flag), the :obj:`~prism.Pipeline` instance will use the :meth:`~prism.modellink.ModelLink.call_model` method differently.
As explained in :ref:`writing_modellink`, every model evaluation sample is requested individually in serial by default, which corresponds to :attr:`~prism.modellink.ModelLink.multi_call` is :pycode:`False` and :attr:`~prism.modellink.ModelLink.MPI_call` is :pycode:`False`.
When single-calling a model, *PRISM* expects an array-like container back with shape :pycode:`(n_data)`.
If we assume that we have an instance of the :class:`LineLink` class (introduced in `line_link.py`_) called :pycode:`modellink_obj` and want to evaluate the model three times for all data points, then the model would be called as (solely by the controller rank)::

    # Get emul_i, sam_set and data_idx
    emul_i = 1
    sam_set = np.random.rand(3, modellink_obj.n_par)
    data_idx = modellink_obj.data_idx

    # Evaluate model
    mod_set = np.zeros([sam_set.shape[0], len(data_idx)])
    for i, par_set in enumerate(sam_set):
        par_dict = dict(zip(modellink_obj.par_name, par_set))
        mod_set[i] = modellink_obj.call_model(emul_i, par_dict, data_idx)

Here, we looped through the entire sample set one-by-one, converted every individual sample to a dict and called the model with it.
The emulator iteration is given as a normal integer and the data identifiers :pycode:`data_idx` is provided as a list of individual data identifiers (which are either single elements or tuples of elements, as described in :ref:`data_idx`).
The requested data identifiers are not necessarily the same as those given in :attr:`~prism.modellink.ModelLink.data_idx`.
An individual sample provided in this way will be of the form::

    par_dict = {'par_name_1': par_1_val,
                'par_name_2': par_2_val,
                ...,
                'par_name_n': par_n_val}

An example of this would be :pycode:`par_dict = {'A': 1.0, 'B': 2.0}` for the :class:`LineLink` class.
This works very well for models that do not require any preparation before they can start evaluating and requires a minimal amount of effort to implement.
However, if the sample set is very large, then evaluating the model in this fashion can be inefficient due to many memory look-ups.

Therefore, the :class:`~prism.modellink.GaussianLink` class accepts both single and multi-calls.
When multi-calling a model, *PRISM* expects an array-like container back with shape :pycode:`(n_sam, n_data)`.
So, if we use the same example again, but this time have an instance of the :class:`~prism.modellink.GaussianLink` class with :attr:`~prism.modellink.ModelLink.multi_call` is :pycode:`True`, then the model would be called as (again solely by the controller rank)::

    # Get emul_i, sam_set and data_idx
    emul_i = 1
    sam_set = np.random.rand(3, modellink_obj.n_par)
    data_idx = modellink_obj.data_idx
    
    # Evaluate model
    sam_dict = dict(zip(modellink_obj.par_name, sam_set.T))
    mod_set = modellink_obj.call_model(emul_i, sam_dict, data_idx)

This call is roughly the same as before, but this time the entire sample set is provided as a dict instead of individual samples.
The lay-out of this sample dict is of the form::

    sam_dict = {'par_name_1': [par_1_val_1, par_1_val_2, ..., par_1_val_m],
                'par_name_2': [par_2_val_1, par_2_val_2, ..., par_2_val_m],
                ...,
                'par_name_n': [par_n_val_1, par_n_val_2, ..., par_n_val_m]}
    
Again, in the case of the :class:`~prism.modellink.GaussianLink` class, this sample dict could look like :pycode:`sam_dict = {'A1': [1.0, 5.5, 10.0], 'B1': [0.0, 5.0, 10.0], 'C1': [0.0, 2.5, 5.0]}`.
This can be used when the model requires some kind of preparation before being able to perform evaluations, or when it is simply more efficient to provide all requested samples at once (like for the :class:`~prism.modellink.GaussianLink` class).

.. note::
    If a model uses OpenMP parallelization, it is recommended to set :attr:`~prism.modellink.ModelLink.MPI_call` to :pycode:`False` in the :class:`~prism.modellink.ModelLink` subclass.
    This allows for all worker ranks to be used in OpenMP threads, while only the controller rank calls the model.

When the :attr:`~prism.modellink.ModelLink.MPI_call` flag is set to :pycode:`True`, the calls to the :meth:`~prism.modellink.ModelLink.call_model` method are almost the same as described above.
The only difference is that all ranks call the method (each providing the same :pycode:`emul_i`, :pycode:`par_dict`/:pycode:`sam_dict` and :pycode:`data_idx`) instead of just the controller rank.

Multi-calling 
#############
When the :attr:`~prism.modellink.ModelLink.multi_call` flag is set to :pycode:`False`, the :meth:`~prism.modellink.ModelLink.call_model` method is most likely nothing more than a simple function.
But, when :attr:`~prism.modellink.ModelLink.multi_call` is set to :pycode:`True`, :meth:`~prism.modellink.ModelLink.call_model` can be a lot more complex.
An example of this would be if we tried to make an emulator of an emulator (which is possible, but completely pointless).
In this case, it would be necessary for the "model" (as we are going to call the emulated emulator from now on) to be loaded into memory first before it can be evaluated.
Although loading an emulator into memory usually does not take that long, we do not want to do this for every single "model" evaluation.
Besides, evaluating an emulator is much quicker when all samples are evaluated at once (due to the way the :meth:`~prism.Pipeline._evaluate_sam_set` method is written).

So, therefore, it is necessary to use :attr:`~prism.modellink.ModelLink.multi_call` is :pycode:`True` for this "model".
If we assume that we have already made an emulator of the :class:`LineLink` class, then, the :meth:`~prism.modellink.ModelLink.call_model` method could be written as::

    def call_model(self, emul_i, par_set, data_idx):
        # Initialize Pipeline object as a model
        modellink_obj = LineLink()
        pipe_model = Pipeline(modellink_obj, working_dir='linelink_0')
        
        # Call pipe_model
        mod_set = pipe_model.evaluate(par_set, emul_i)['adj_exp_val']
        
        # Make sure only the requested data points are kept
        req_idx = [pipe_model.emulator._data_idx[emul_i].index(idx) for idx in data_idx]
        mod_set = mod_set[:, req_idx]

        # Return mod_set
        return(mod_set)

Here, we only initialize the "model" once per model call, and then evaluate all samples in it by using the :meth:`~prism.Pipeline.evaluate` method (which can take sample dicts as a valid input argument).
This returns a dict of the evaluation results, where we are only interested in the adjusted expectation values.
Note that making an emulator of an emulator is pointless, but used here as an example.

.. note::
    Due to the way *PRISM* is written, it is technically speaking not necessary to reinitialize the :class:`~prism.Pipeline` class every time that :meth:`~prism.modellink.ModelLink.call_model` is called.
    It is possible to initialize it when the corresponding :class:`~prism.modellink.ModelLink` subclass is initialized and keep it in memory.
    The code above would however be necessary if the "model" works in the same way as *PRISM*'s :attr:`~prism.Pipeline.worker_mode`, where all worker ranks are listening for calls until the "model" is finalized.
    This finalization would be required in order to give *PRISM* control back over all ranks.

