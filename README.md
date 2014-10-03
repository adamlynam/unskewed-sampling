Unskewed Sampling
=================

A WEKA compatible implementation of the Unskewed Sampling meta classification technique

This technique allows for only the two most basic transformations of a skewed dataset. These transformations are under-sampling of the majority class, the same as One-sided Selection discussed by Kubat and Matwin[1] and over-sampling of the minority class. Collectively these two will be known as Unskewed Sampling. With this basic meta classifier it is possible to reduce the number of majority class examples and/or duplicate minority class examples in order to get a more even class distribution in two class problems.

1) M. Kubat and S. Matwin. Addressing the Curse of Imbalanced Training Sets: One-Sided Selection. In Proceedings of the 14th International Conference on Machine Learning, pages 179-186. Morgan Kaufmann, 1997
