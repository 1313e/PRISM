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
When single-calling a model, *PRISM* expects an array-like container back with shape :pycode:`(n_data,)`, where the order of the elements is the same as the order of the requested :pycode:`data_idx`.
If we assume that we have an instance of the :class:`LineLink` class (introduced in `line_link.py`_) called :pycode:`modellink_obj` and want to evaluate the model three times for all data points, then the model would be called as (solely by the controller rank)::

    # Import SortedDict class
    from sortedcontainers import SortedDict as sdict

    # Get emul_i, sam_set and data_idx
    emul_i = 1
    sam_set = np.random.rand(3, modellink_obj.n_par)
    data_idx = modellink_obj.data_idx

    # Evaluate model
    mod_set = np.zeros([sam_set.shape[0], len(data_idx)])
    for i, par_set in enumerate(sam_set):
        par_dict = sdict(zip(modellink_obj.par_name, par_set))
        mod_set[i] = modellink_obj.call_model(emul_i=emul_i,
                                              par_set=par_dict,
                                              data_idx=data_idx)

Here, we looped through the entire sample set one-by-one, converted every individual sample to a (sorted) dict and called the model with it.
The emulator iteration is given as a normal integer and the data identifiers :pycode:`data_idx` is provided as a list of individual data identifiers (which are either single elements or tuples of elements, as described in :ref:`data_idx`).
The requested data identifiers are not necessarily the same as those given in :attr:`~prism.modellink.ModelLink.data_idx`.
An individual sample provided in this way will be of the form::

    par_dict = {'par_1_name': par_1_val,
                'par_2_name': par_2_val,
                ...,
                'par_n_name': par_n_val}

An example of this would be :pycode:`par_dict = {'A': 1.0, 'B': 2.0}` for the :class:`LineLink` class.
This works very well for models that do not require any preparation before they can start evaluating and requires a minimal amount of effort to implement.
However, if the sample set is very large, then evaluating the model in this fashion can be inefficient due to many memory look-ups.

Therefore, the :class:`~prism.modellink.GaussianLink` class accepts both single and multi-calls.
When multi-calling a model, *PRISM* expects an array-like container back with shape :pycode:`(n_sam, n_data)`, where the order of the columns is the same as the order of the requested :pycode:`data_idx`.
So, if we use the same example again, but this time have an instance of the :class:`~prism.modellink.GaussianLink` class with :attr:`~prism.modellink.ModelLink.multi_call` is :pycode:`True`, then the model would be called as (again solely by the controller rank)::

    # Get emul_i, sam_set and data_idx
    emul_i = 1
    sam_set = np.random.rand(3, modellink_obj.n_par)
    data_idx = modellink_obj.data_idx
    
    # Evaluate model
    sam_dict = sdict(zip(modellink_obj.par_name, sam_set.T))
    mod_set = modellink_obj.call_model(emul_i=emul_i,
                                       par_set=sam_dict,
                                       data_idx=data_idx)

This call is roughly the same as before, but this time the entire sample set is provided as a (sorted) dict instead of individual samples.
The lay-out of this sample dict is of the form::

    sam_dict = {'par_1_name': [par_1_val_1, par_1_val_2, ..., par_1_val_m],
                'par_2_name': [par_2_val_1, par_2_val_2, ..., par_2_val_m],
                ...,
                'par_n_name': [par_n_val_1, par_n_val_2, ..., par_n_val_m]}
    
Again, in the case of the :class:`~prism.modellink.GaussianLink` class, this sample dict could look like :pycode:`sam_dict = {'A1': [1.0, 5.5, 10.0], 'B1': [0.0, 5.0, 10.0], 'C1': [0.0, 2.5, 5.0]}`.
This can be used when the model requires some kind of preparation before being able to perform evaluations, or when it is simply more efficient to provide all requested samples at once (like for the :class:`~prism.modellink.GaussianLink` class).

.. note::
    If a model uses OpenMP parallelization, it is recommended to set :attr:`~prism.modellink.ModelLink.MPI_call` to :pycode:`False` in the :class:`~prism.modellink.ModelLink` subclass.
    This allows for all worker ranks to be used in OpenMP threads, while only the controller rank calls the model.

.. note::
    If one wishes to transform the received :pycode:`sam_dict` back into a normal NumPy array of shape :pycode:`(n_sam, n_par)`, this can be done quite easily by executing :pycode:`sam_set = np.array(par_set.values()).T`, where :pycode:`par_set` is the :pycode:`sam_dict` provided to the :meth:`~prism.modellink.ModelLink.call_model` method.
    Keep in mind that doing so means that the columns are sorted on the names of the model parameters.
    If one instead wishes to transform it into a generator, use :pycode:`sam_set=map(lambda *args: args, *par_set.values())`.

.. versionadded:: 1.1.2
    It is also possible to make :meth:`~prism.modellink.ModelLink.call_model` return a dict instead, where it has the identifiers in the requested :pycode:`data_idx` as its keys and scalars (single-call) or 1D array-likes of shape :pycode:`(n_sam)` (multi-call) as its values.
    *PRISM* will automatically convert the dict back to the array-like container format that is normally expected.

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


Backing up progress
###################
.. versionadded:: 1.1.1

.. warning::
    This feature is still experimental and it may see significant changes or be (re)moved in the future.

In *PRISM*, an emulator system is constructed by calculating all required components individually.
This means that the construction process of an emulator iteration can easily be interrupted and restored at a later time, only losing the progress that was made in the current step (e.g., interrupting construction during the calculation of the covariance matrix will lose progress made there, but not the already previously finished steps).
This system was implemented to accommodate for *PRISM* running on clusters, where the construction is more prone to interruptions due to, for example, jobs timing out, and to allow for *PRISM* to be loaded unto any number of MPI processes.

However, the biggest step in the construction of all emulator systems, is the evaluation of the model.
Since the evaluation of the model is carried out by the :meth:`~prism.modellink.ModelLink.call_model` method, *PRISM* has no control over what is happening until this method gives control back to the :obj:`~prism.Pipeline` instance (by returning the requested data points).
Therefore, automated backups of already calculated data points cannot be performed by *PRISM* itself, running the risk that many CPU hours are wasted if a job on a cluster takes longer than initially expected and times out.
While this could be avoided if the user writes its own backup system, this would require more work from the user, which clashes with *PRISM*'s ease-of-use policy.

Therefore, the :class:`~prism.modellink.ModelLink` class implements its own (experimental) backup system based on the `hickle`_ package, given by the :meth:`~prism.modellink.ModelLink._make_backup` and :meth:`~prism.modellink.ModelLink._read_backup` methods.
This backup system is best used for models that are multi-called (:attr:`~prism.modellink.ModelLink.multi_call` set to :pycode:`True`), as made backups will replace previous ones (of the same type).
The :meth:`~prism.modellink.ModelLink._make_backup` method is meant to be used from within the :meth:`~prism.modellink.ModelLink.call_model` method and will not work if called anywhere else.
Attempting to call it incorrectly (e.g., not from within :meth:`~prism.modellink.ModelLink.call_model` or with incorrect arguments), will raise a :class:`~prism._internal.RequestWarning` and simply return without doing anything, rather than raising a :class:`~prism._internal.RequestError`.
This is to make sure that using it incorrectly does not disrupt the :meth:`~prism.modellink.ModelLink.call_model` call, as that has the exact opposite effect of what the backup system tries to achieve.

The :meth:`~prism.modellink.ModelLink._make_backup` method takes two arguments, :pycode:`*args` and :pycode:`**kwargs`, of which at least one is required.
Calling it from within the :meth:`~prism.modellink.ModelLink.call_model` method will produce an HDF5-file containing the :pycode:`emul_i`, :pycode:`par_set` and :pycode:`data_idx` argument values that were used to call :meth:`~prism.modellink.ModelLink.call_model` with, and the supplied :pycode:`*args` and :pycode:`**kwargs`.
The name of the HDF5-file contains the values of :pycode:`emul_i` and :attr:`~prism.modellink.ModelLink.name`, and will be saved in the current working directory (NOT the emulator working directory, as the :obj:`~prism.modellink.ModelLink` instance has no access to its path).
The backup can be read in by passing the value of :pycode:`emul_i` to the :meth:`~prism.modellink.ModelLink._read_backup` method of the corresponding :obj:`~prism.modellink.ModelLink` instance, which will return a dict containing the values of the five arguments that were saved to the file.

Backups can be made at any point during the execution of :meth:`~prism.modellink.ModelLink.call_model`, and basically all types of objects are compatible and can be viewed freely in the HDF5-file.
It is possible that instances of certain custom classes may not be supported by the `hickle`_ package, in which case they will be pickled and saved as a string, causing them to not be able to be viewed freely (but they can still be backed up).
Depending on the size of the data provided, it can sometimes take a little while before a backup is made.
Therefore, it is probably best to trigger making backups at specified progress points in :meth:`~prism.modellink.ModelLink.call_model`.

.. _hickle: https://github.com/telegraphic/hickle

To illustrate how this backup system can be used, assume that we have written a :class:`~prism.modellink.ModelLink` subclass, which requires some preparation before it can start evaluating the wrapped model.
Here, we will assume that this preparation is provided by a function called :pycode:`prepare_model()`, which returns an instance of some class that can be used to evaluate the model after the preparation is completed.
Then, we could incorporate the backup system by writing a :meth:`~prism.modellink.ModelLink.call_model` method like this::

    def call_model(self, emul_i, par_set, data_idx):
        # Prepare the model for evaluation
        model = prepare_model()

        # Controller performs evaluations
        if model.is_controller:
            # Initialize empty array of results
            mod_set = np.zeros([len(par_set['par1']), len(data_idx)])
            
            # Unpack par_set into a NumPy array
            sam_set = np.array(par_set.values()).T

            # Call model for every individual sample in sam_set
            for i, sam in enumerate(sam_set):
                mod_set[i] = model.evaluate(sam, data_idx)
                
                # Make a backup every 500 evaluations
                if not((i+1) % 500):
                    self._make_backup(mod_set=mod_set[:i])

        # Finalize the model
        model.finalize()

        # Return the results on the controller
        if model.is_controller:
            return(mod_set)

The code above shows an example of a model that needs to be initialized before it can be multi-called in MPI, and needs to be finalized afterward.
Since such a model is probably quite complex, it may be a good idea to make a backup every once in a while.
Therefore, whenever 500 evaluations have been done, a backup is made of all results gained up to that point.
This means that whenever the model evaluation process is interrupted, a maximum of the last 500 evaluations is lost.
The evaluations that are not lost can be loaded back in by using the :meth:`~prism.modellink.ModelLink._read_backup` method, and potentially (after a bit of formatting) be passed to the :pycode:`ext_real_set` input argument of the :meth:`~prism.Pipeline.construct` method when attempting to construct the emulator iteration again.

Note that if :pycode:`model.evaluate()` was implemented such that it takes the entire sample set at once rather than one at a time, calling :meth:`~prism.modellink.ModelLink._make_backup` in :pycode:`model.evaluate()` works perfectly fine, as long as :pycode:`model.evaluate()` is always called by :meth:`~prism.modellink.ModelLink.call_model` or any other function for which this is true.
Put a little bit more simple: :meth:`~prism.modellink.ModelLink._make_backup` must be called either directly or indirectly by :meth:`~prism.modellink.ModelLink.call_model`, as shown in the following example.

.. admonition:: Example

    ::

        def call_model(self, emul_i, par_set, data_idx):
            # Call a function A and return its output
            # This function does not require emul_i, so do not provide it
            return(A(self, par_set, data_idx))

        def A(modellink_obj, par_set, data_idx):
            # Prepare model
            model = prepare_model()

            # Prepare par_set for evaluation
            sam_set = np.array(par_set.values()).T

            # Call a function B
            mod_set = B(modellink_obj, model, sam_set, data_idx)
            
            # Finalize the model
            model.finalize()
            
            # Return the results
            return(mod_set)

        def B(modellink_obj, model_obj, sam_set, data_idx):
            # Prepare mod_set
            mod_set = np.zeros([np.shape(sam_set)[0], len(data_idx)])
            
            # Call model for every individual sample in sam_set
            for i, sam in enumerate(sam_set):
                mod_set[i] = model_obj.evaluate(sam, data_idx)
                
                # Make a backup every 500 evaluations
                if not((i+1) % 500):
                    modellink_obj._make_backup(mod_set=mod_set[:i+1])

            # Return mod_set
            return(mod_set)


