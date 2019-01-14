.. _writing_modellink:

Writing a ModelLink subclass
++++++++++++++++++++++++++++
In :ref:`minimal_example`, a description is given of how to initialize the :class:`~prism.Pipeline` class using a default :class:`~prism.modellink.ModelLink` subclass.
Here, the basic steps for making a custom :class:`~prism.modellink.ModelLink` subclass are shown.

----

.. topic:: example_link.py
    :name: example_link.py

    .. include:: scripts/example_link.py
        :literal:
        :code: python

----

In the example_link.py_ file above, a minimal example of a :class:`~prism.modellink.ModelLink` subclass is shown.
It has two abstract methods that need to be overridden; :meth:`~prism.modellink.ModelLink.call_model` (wrapper function for calling the model) and :meth:`~prism.modellink.ModelLink.get_md_var` (calculates the model discrepancy variance).
A :class:`~prism.modellink.ModelLink` subclass cannot be initialized if either method has not been overridden.
Given the importance of both methods, detailed descriptions are given in :ref:`call_model` and :ref:`md_var`, respectively.

Since every model is different, with some requiring preparations in order to work properly, the :meth:`~prism.modellink.ModelLink.__init__` constructor method may be extended to include any custom code to be executed when the subclass is initialized.
The superclass version of the :meth:`~prism.modellink.ModelLink.__init__` method must always be called, as it sets several important flags and properties, but the time at which this is done does not matter.
During the initialization of the :class:`~prism.Emulator` class, it is checked whether or not the superclass constructor of a provided :obj:`~prism.modellink.ModelLink` instance was called (to avoid this from being forgotten).

Besides executing custom code, three properties/flags can be set in :meth:`~prism.modellink.ModelLink.__init__`, which have the following default values if the extended constructor does not set them::

    self.name = self.__class__.__name__ # Set instance name to the name of the class
    self.call_type = 'single'           # Request single model calls
    self.MPI_call = False               # Request only controller calls

The first property, :attr:`~prism.modellink.ModelLink.name`, defines the name of the :obj:`~prism.modellink.ModelLink` instance.
This name is used by the :class:`~prism.Emulator` class during initialization to check if a constructed emulator is linked to the proper :obj:`~prism.modellink.ModelLink` instance, in order to avoid causing mismatches.
If one wants to use the same :class:`~prism.modellink.ModelLink` subclass for different models (like, using different parameter spaces), it is recommended to add an identifier for this to this name.
An example of this can be found in the definition of the :class:`~prism.modellink.GaussianLink` class, which adds the number of Gaussians in the model to its :attr:`~prism.modellink.ModelLink.name` property.

The other two properties, :attr:`~prism.modellink.ModelLink.call_type` and :attr:`~prism.modellink.ModelLink.MPI_call`, are flags that tell *PRISM* how the :meth:`~prism.modellink.ModelLink.call_model` method should be used.
By default, every model evaluation sample is requested individually in serial, since this would be the most expected behavior.
However, this is most likely not enough for sophisticated models, as they can require some preparation (e.g., having to read in data files) or more than a single core (in MPI) to function.
Therefore, :attr:`~prism.modellink.ModelLink.call_type` can be set to accept solely individual samples (:pycode:`'single'`), solely entire sample sets (:pycode:`'multi'`) or both (:pycode:`'hybrid'`).
In the same way, :attr:`~prism.modellink.ModelLink.MPI_call` can be set to :pycode:`True` or :pycode:`False` to identify that the model needs to be executed in serial or in MPI.

.. note::
    If a model uses OpenMP parallelization, it is recommended to set :attr:`~prism.modellink.ModelLink.MPI_call` to :pycode:`False` in the :class:`~prism.modellink.ModelLink` subclass.
    This allows for all worker ranks to be used in OpenMP threads, while only the controller rank calls the model.

Finally, the :class:`~prism.modellink.ModelLink` class has three methods that can be overridden for adding utility to the class (of which two are shown in example_link.py_).
The :meth:`~prism.modellink.ModelLink.get_default_model_parameters` and :meth:`~prism.modellink.ModelLink.get_default_model_data` methods return dictionaries containing the default model parameters and model data to use in this class instance, respectively.
By overriding these methods, one can hard-code the use of specific parameters or comparison data, avoiding having to provide them when initializing the :class:`~prism.modellink.ModelLink` subclass.
Additionally, if a default parameter or data point is also provided during initialization, the provided information will override the defaults.

.. admonition:: Example

    The :class:`~prism.modellink.GaussianLink` class has default parameters defined::
	
        >>> from prism.modellink import GaussianLink
        >>> model_data = {3: [3.0, 0.1]}
        >>> modellink_obj = GaussianLink(model_data=model_data)
        >>> modellink_obj
        GaussianLink(model_parameters={'A1': [1.0, 10.0, 5.0], 'B1': [0.0, 10.0, 5.0],
                                       'C1': [0.0, 5.0, 2.0]},
                     model_data={3: [3.0, 0.1]})

    Providing a custom set of parameters will override the coded defaults::
	
        >>> model_parameters = {'A1': [-5, 7, 2]}
        >>> modellink_obj = GaussianLink(model_parameters=model_parameters, model_data=model_data)
        >>> modellink_obj
        GaussianLink(model_parameters={'A1': [-5.0, 7.0, 2.0], 'B1': [0.0, 10.0, 5.0],
                                       'C1': [0.0, 5.0, 2.0]},
                     model_data={3: [3.0, 0.1]})

The third method, :meth:`~prism.modellink.ModelLink.get_str_repr`, is a simple function that returns a list containing the representations of all non-default input arguments the :class:`~prism.modellink.ModelLink` subclass takes.
It can be overridden to add the missing input arguments to the full representation of the class, which is automatically called whenever the representation is requested.
The :class:`~prism.modellink.GaussianLink` class overrides this method to add its :attr:`~prism.modellink.GaussianLink.n_gaussians` input argument.

----

.. topic:: line_link.py
    :name: line_link.py

    .. include:: scripts/line_link.py
        :literal:
        :code: python

----

Using all the information above and the template given in example_link.py_, a :class:`~prism.modellink.ModelLink` subclass can be written for a straight line model, shown in the line_link.py_ file above.
Here, all methods discussed before (besides the :meth:`~prism.modellink.ModelLink.get_str_repr` method, since no additional input arguments are used) have been overridden.
Given that this model is very simple, no changes have been made to the instance constructor, :meth:`~prism.modellink.ModelLink.__init__`.
Therefore, only single evaluation samples in serial are requested.

*PRISM* provides the :func:`~prism.modellink.test_subclass` function that allows the user to check if a :class:`~prism.modellink.ModelLink` subclass is properly written.
It returns an instance of the subclass if the test passes, or raises a specific error if not.
We can use this function to initialize our newly written subclass::

    >>> from line_link import LineLink
    >>> from prism.modellink import test_subclass
    >>> modellink_obj = test_subclass(LineLink)
    >>> modellink_obj
    LineLink(model_parameters={'A': [-10.0, 10.0, 3.0], 'B': [0.0, 5.0, 1.5]},
             model_data={2.5: [6.8, 0.1], -2: [0.0, 0.1], 1: [4.5, 0.1]})

Since no errors were raised, we can now use the initialized :class:`~prism.modellink.ModelLink` subclass to initialize the :class:`~prism.Pipeline` class::

    >>> from prism import Pipeline
    >>> pipe = Pipeline(modellink_obj)

