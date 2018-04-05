/*
Package sparse provides implementations of selected sparse matrix formats.  Matrices and linear algebra are used extensively in scientific computing and machine learning applications.  Large datasets are analysed comprising vectors of numerical features that represent some object.  The nature of feature encoding schemes, especially those like "one hot", tends to lead to vectors with mostly zero values for many of the features.  In text mining applications, where features are typically terms from a vocabulary, it is not uncommon for 99% of the elements within these vectors to contain zero values.

Sparse matrix formats take advantage of this fact to optimise memory usage and processing performance by only storing and processing non-zero values.

Sparse matrix formats can broadly be divided into 3 main categories:

1. Creational - Sparse matrix formats suited to construction and building of matrices.  Matrix formats in this category include DOK (Dictionary Of Keys) and COO (COOrdinate aka triplet).

2. Operational - Sparse matrix formats suited to arithmetic operations e.g. multiplication.  Matrix formats in this category include CSR (Compressed Sparse Row aka CRS - Compressed Row Storage) and CSC (Compressed Sparse Column aka CCS - Compressed Column Storage)

3. Specialised - Specialised matrix formats suiting specific sparsity patterns.  Matrix formats in this category include DIA (DIAgonal) for efficiently storing and manipulating symmetric diagonal matrices.

A common practice is to construct sparse matrices using a creational format e.g. DOK or COO and then convert them to an operational format e.g. CSR for arithmetic operations.

All sparse matrix implementations in this package implement the Matrix interface defined within the gonum/mat package and so may be used interchangeably with matrix types defined within the package e.g. mat.Dense, mat.VecDense, etc.
*/
package sparse
