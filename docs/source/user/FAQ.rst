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

Why does increasing the number of MPI processes not increase the evaluation rate?
---------------------------------------------------------------------------------
Currently, *PRISM* uses a high-level MPI implementation.
This means that the evaluation rate of the emulator scales with the highest number of emulator systems (data points) that are assigned to a single MPI process.
For example, having `16` emulator systems will roughly yield the same evaluation rate on `8` processes and `15` processes (and everything in between).
Low-level MPI is planned to be implemented in the future, removing this limitation.
Currently, it is advised to make sure that the number of emulator systems is divisable by the number of MPI processes.

Why does the evaluation rate decrease with increasing number of MPI processes?
------------------------------------------------------------------------------
Many of the calculations in *PRISM* require NumPy's `lin_alg` functions, which use OpenMP for calculations.
On some architectures, it is possible that NumPy spawns much more OpenMP threads than there are MPI processes available.
Setting the number of OpenMP threads to `1` (``export OMP_NUM_THREADS=1`` on UNIX) will remove this problem.