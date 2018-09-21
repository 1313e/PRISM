Usage Notes
-----------
- Unless specified otherwise, all input arguments in the *PRISM* package that accept a bool (*True*/*False*) also accept 0/1 as a valid input;
- Unless specified otherwise, all input arguments in the *PRISM* package that accept *None* indicate a default value or operation for obtaining this input argument.
  In most of these cases, the default value depends on the current state of the *PRISM* pipeline, and therefore a small operation is required for obtaining this value.
  An example is providing *None* to :attr:`~prism.pipeline.Pipeline.pot_active_par`, where it indicates that all model parameters should be potentially active;
- Unless specified otherwise, all input arguments in the *PRISM* package that accept the names of model parameters also accept the internal indices of these model parameters.
  The index is the order in which the parameter names appear in the :attr:`~prism.modellink.modellink.ModelLink` list or as they appear in the output of the :meth:`~prism.pipeline.Pipeline.details` method.
- Depending on the used emulator type, state of loaded emulator and the *PRISM* parameter values, it is possible that providing values for certain *PRISM* parameters has no influence on the outcome of the pipeline.
  This can be either because they have non-changeable default values or are simply not used anywhere (given the current state of the pipeline).
  An example of this is when :attr:`~prism.emulator.Emulator.method` != ``'gaussian'``, which causes :attr:`~prism.emulator.Emulator.sigma` to have no use in the pipeline.
  An other example is switching the bool value for :attr:`~prism.emulator.Emulator.use_mock` while loading a constructed emulator, since the mock data is generated (or not) when constructing a new emulator and cannot be changed or swapped out afterward;
- All docstrings in *PRISM* are written in `RTF`_ (Rich Text Format) and are therefore best viewed in an editor that supports it (like `Spyder`_).

.. _RTF: https://en.wikipedia.org/wiki/Rich_Text_Format
.. _Spyder: https://pythonhosted.org/spyder/installation.html
