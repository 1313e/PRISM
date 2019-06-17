.. _minimal_example:

Minimal example
+++++++++++++++
A minimal example on how to initialize and use the *PRISM* pipeline is shown here.
First, one has to import the :class:`~prism.Pipeline` class and a :class:`~prism.modellink.ModelLink` subclass::

    >>> from prism import Pipeline
    >>> from prism.modellink import GaussianLink

Normally, one would import a custom-made :class:`~prism.modellink.ModelLink` subclass, but for this example one of the two :class:`~prism.modellink.ModelLink` subclasses that come with the *PRISM* package is used (see :ref:`writing_modellink` for the basic structure of writing a custom :class:`~prism.modellink.ModelLink` subclass).

Next, the :class:`~prism.modellink.ModelLink` should be initialized, which is the :class:`~prism.modellink.GaussianLink` class in this case.
In addition to user-defined arguments, every :class:`~prism.modellink.ModelLink` subclass takes two optional arguments, `model_parameters` and `model_data`.
The use of either one will add the provided parameters/data to the default parameters/data defined in the class.
Since the :class:`~prism.modellink.GaussianLink` class does not have default data defined, it is required to supply it with some data during initialization (using an array, dict or external file)::

    >>> # f(3) = 3.0 +- 0.1, f(5) = 5.0 +- 0.1, f(7) = 3.0 +- 0.1
    >>> model_data = {3: [3.0, 0.1], 5: [5.0, 0.1], 7: [3.0, 0.1]}
    >>> modellink_obj = GaussianLink(model_data=model_data)

Here, the :class:`~prism.modellink.GaussianLink` class was initialized by giving it three custom data points and using its default parameters.
One can check this by looking at the representation of this :obj:`~prism.modellink.GaussianLink` object::

    >>> modellink_obj
    GaussianLink(model_parameters={'A1': [1.0, 10.0, 5.0], 'B1': [0.0, 10.0, 5.0],
                                   'C1': [0.0, 5.0, 2.0]},
                 model_data={7: [3.0, 0.1], 5: [5.0, 0.1], 3: [3.0, 0.1]})

The :class:`~prism.Pipeline` class takes several optional arguments, which are mostly paths and the type of :class:`~prism.emulator.Emulator` class that must be used.
It also takes one mandatory argument, which is an instance of the :class:`~prism.modellink.ModelLink` subclass to use.
Since it has already been initialized above, the :class:`~prism.Pipeline` class can be initialized::

    >>> pipe = pipeline(modellink_obj)
    >>> pipe
    Pipeline(GaussianLink(model_parameters={'A1': [1.0, 10.0, 5.0], 'B1': [0.0, 10.0, 5.0],
                                            'C1': [0.0, 5.0, 2.0]},
                          model_data={7: [3.0, 0.1], 5: [5.0, 0.1], 3: [3.0, 0.1]}),
             working_dir='prism_0')

Since no working directory was provided to the :class:`~prism.Pipeline` class and none already existed, it automatically created one (``prism_0``).

*PRISM* is now completely ready to start emulating the model.
The :class:`~prism.Pipeline` allows for all steps in a full cycle (see :ref:`PRISM pipeline <PRISM>`) to be executed automatically::

    >>> pipe.run()

which is equivalent to::

    >>> pipe.construct(analyze=False)
    >>> pipe.analyze()
    >>> pipe.project()

This will construct the next iteration (first in this case) of the emulator, analyze it to check if it contains plausible regions and make projections of all active parameters.
The current state of the :obj:`~prism.Pipeline` object can be viewed by calling the :meth:`~prism.Pipeline.details` method (called automatically after most user-methods), which gives an overview of many properties that the :obj:`~prism.Pipeline` object currently has.

This is all that is required to construct an emulator of the model of choice.
All user-methods, with one exception (:meth:`~prism.Pipeline.evaluate`), solely take optional arguments and perform the operations that make the most sense given the current state of the :obj:`~prism.Pipeline` object if no arguments are given.
These arguments allow for one to modify the performed operations, like reconstructing/reanalyzing previous iterations, projecting specific parameters, evaluating the emulator and more.
