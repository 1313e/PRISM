.. role:: pycode(code)
    :language: python3
    :class: highlight

Below are some bigger ideas/improvements that may be added to *PRISM* if there is demand:

    - Low-level MPI implementation (probably by using D2O);
    
      With 6 emulator systems and 4 processes, the three different MPI levels would be:

        - No level: 6-0-0-0;
        - High-level: 2-2-1-1;
        - Low-level: 1.5-1.5-1.5-1.5.

    - Dynamic implausibility cut-offs;
    - Allow for a master projection figure to be made (kind of like a double corner plot);
    - Set :ref:`parameters` properties manually (using a property setter) instead of only reading it in from file.
      This will additionally allow for external functions to be executed that determine the values of certain properties;
    - Allow for user-provided methods in the :class:`~prism.modellink.ModelLink` subclass to be executed at specific points in the emulator construction;
    - Implement multi-variate implausibilities;
    - Allow for no :obj:`~prism.modellink.ModelLink` object to be provided, which blocks construction but enables everything emulator-only related;
    - Allow for old *PRISM* master files to be provided when making a new emulator, recycling work done previously;
    - If :attr:`~prism.modellink.ModelLink.MPI_call` is :pycode:`False` for the :class:`~prism.modellink.ModelLink` subclass, use all MPI ranks to evaluate a part of :pycode:`sam_set` in the model simultaneously.
      This will require a check or flag that the model can be called in multiple instances simultaneously (to accommodate for models that, for example, need to read files during evaluations).
      Added benefit of this is that it would become possible to add the option for the user to set a preferred number of MPI processes calling the model (in MPI), allowing *PRISM* to split up the available processes if more efficient;
    - GPU acceleration;
    - Adding the theory behind *PRISM* to the docs;
    - Adding the possibility to evaluate the derivatives of the emulated model outputs, which could be used as approximations of the gradient field of a model for certain MCMC methods;
    - Replace the *list of lists* data system with a *list of dicts* system.
      This would remove the need for converting global indices to/from local indices in several cases, and make it easier for users to understand.
      However, as indexing dicts is more complicated, this may require a lot of rewriting;
    - Code objects can be made pickleable by importing the :mod:`~ipykernel.codeutil` module.
      This package could be added to the requirements or an equivalent function could be written, which is then automatically imported/executed upon importing *PRISM*;

