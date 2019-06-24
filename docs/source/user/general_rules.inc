.. _general_rules:

General usage rules
+++++++++++++++++++
Below is a list of general usage rules that apply to *PRISM*.

- Unless specified otherwise in the documentation, any input argument in the *PRISM* package that accepts...

    - a bool (:pycode:`True`/:pycode:`False`) also accepts 0/1 as a valid input;
    - :pycode:`None` indicates a default value or operation for obtaining this input argument.
      In most of these cases, the default value depends on the current state of the *PRISM* pipeline, and therefore a small operation is required for obtaining this value;

      .. admonition:: Example

          Providing :pycode:`None` to :attr:`~prism.Pipeline.pot_active_par`, where it indicates that all model parameters should be potentially active.

    - the names of model parameters also accepts the internal indices of these model parameters.
      The index is the order in which the parameter names appear in the :attr:`~prism.modellink.ModelLink.par_name` list or as they appear in the output of the :meth:`~prism.Pipeline.details` method;
    - a parameter/sample set will accept a 1D/2D array-like or a dict of sample(s).
      As with the previous rule, the columns in an array-like are in the order in which the parameter names appear in the :attr:`~prism.modellink.ModelLink.par_name` list;
    - a sequence of integers, floats and/or strings will accept (almost) any formatting including most special characters as separators as long as they do not have any meaning (like a dot for floats or valid escape sequences for strings).
      Keep in mind that providing :pycode:`'1e3'` (or equivalent) will be converted to :pycode:`1000.0`, as per Python standards;

      .. admonition:: Example

          The following sequences are equal:
              - :pycode:`A, 1, 20.0, B`;
              - :pycode:`[A,1,2e1,B]`;
              - :pycode:`"A 1 20. B"`;
              - :pycode:`"'[" (A / }| \n; <1{}) \,,">20.000000 !! \t< )?%\\B '`.
    - the path to a data file (*PRISM* parameters, model parameters, model data) will read in all the data from that file as a Python dict, with a colon :pycode:`:` acting as the separator between the key and value.


- Depending on the used emulator type, state of loaded emulator and the *PRISM* parameter values, it is possible that providing values for certain *PRISM* parameters has no influence on the outcome of the pipeline.
  This can be either because they have non-changeable default values or are simply not used anywhere (given the current state of the pipeline);

  .. admonition:: Examples

      - If :attr:`~prism.emulator.Emulator.method` != :pycode:`'gaussian'`, it causes :attr:`~prism.emulator.Emulator.sigma` to have no use in the pipeline;
      - Switching the bool value for :attr:`~prism.emulator.Emulator.use_mock` while loading a constructed emulator has no effect, since the mock data is generated (or not) when constructing a new emulator and cannot be changed or swapped out afterward.

- All docstrings in *PRISM* are written in `RTF`_ (Rich Text Format) and are therefore best viewed in an editor that supports it (like `Spyder`_).

.. _RTF: https://en.wikipedia.org/wiki/Rich_Text_Format
.. _Spyder: https://www.spyder-ide.org
