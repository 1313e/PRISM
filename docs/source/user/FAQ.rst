.. _FAQ:

FAQ
===
How do I contribute?
--------------------
Contributing to *PRISM* is done through pull requests in the `repository`_.
If you have an idea on what to contribute, it is recommended to open a `GitHub issue`_ about it, such that the maintainers can give their advice or help.
If you want to contribute but do not really know what, then you can take a look at the ToDos that are scattered throughout the code.
When making a contribution, please keep in mind that it must be compatible with all Python versions that *PRISM* supports (3.5+), and preferably with all operating systems as well.

.. _repository: https://github.com/1313e/PRISM
.. _GitHub issue: https://github.com/1313e/PRISM/issues


How do I report a bug/problem?
------------------------------
By opening a `GitHub issue`_ and using the `Bug report` template.

What does the term `...` mean?
------------------------------
A list of the most commonly used terms in *PRISM* can be found on the :ref:`terminology` page.

Where can I get PRISM's colormaps?
----------------------------------
The *rainforest* and *freeze* colormaps that are used for drawing *PRISM*'s projection figures, are freely available in the `e13Tools`_ package.
Importing e13Tools will automatically add both colormaps (and their reverses) to the list of available colormaps in Matplotlib.
One can then access them directly in the :mod:`~matplotlib.cm` module or by using the :func:`~matplotlib.cm.get_cmap` function.

Which OSs are supported?
------------------------
*PRISM* should be compatible with all Windows, Mac OS and UNIX-based machines, as long as they support one of the required Python versions.
Compatibility is currently tested for Linux 16.04 (Xenial)/Mac OS-X using `Travis CI`_, Windows 32-bit/64-bit using `AppVeyor`_ and all OSs using `Azure Pipelines`_.

.. _e13Tools: https://github.com/1313e/e13Tools
.. _Travis CI: https://travis-ci.com/1313e/PRISM
.. _AppVeyor: https://ci.appveyor.com/project/1313e/PRISM
.. _Azure Pipelines: https://dev.azure.com/1313e/PRISM/_build/latest?definitionId=1
