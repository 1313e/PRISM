.. _FAQ:

FAQ
===
What does the term `...` mean?
------------------------------
A list of the most commonly used terms in *PRISM* can be found on the :ref:`terminology` page.

What OS are supported?
----------------------
*PRISM* should be compatible with all Windows, MacOS and UNIX-based machines, as long as they support one of the required Python versions.
Compatibility is currently tested for Linux 16.04 (Xenial) using `Travis CI`_ and Windows 32-bit/64-bit using `AppVeyor`_.
MacOS is not explicitly tested for, which may be added in the future, but *PRISM* should be compatible with it.

.. _Travis CI: https://travis-ci.com/1313e/PRISM
.. _AppVeyor: https://ci.appveyor.com/project/1313e/PRISM

Why does `...` affect the construction speed/evaluation rate?
-------------------------------------------------------------
There are several different aspects in the *PRISM* code that affect its performance, which are discussed on the :ref:`speed_optimization` page.
