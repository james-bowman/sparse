# Sparse matrix formats
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GoDoc](https://godoc.org/github.com/james-bowman/sparse?status.svg)](https://godoc.org/github.com/james-bowman/sparse)
[![Build Status](https://travis-ci.org/james-bowman/sparse.svg?branch=master)](https://travis-ci.org/james-bowman/sparse)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/sparse)](https://goreportcard.com/report/github.com/james-bowman/sparse)
[![codecov](https://codecov.io/gh/james-bowman/sparse/branch/master/graph/badge.svg)](https://codecov.io/gh/james-bowman/sparse)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/avelino/awesome-go)
[![Sourcegraph](https://sourcegraph.com/github.com/james-bowman/sparse/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/sparse?badge)


Implementations of selected sparse matrix formats for linear algebra supporting scientific and machine learning applications compatible with the APIs in the [Gonum](http://www.gonum.org/) library.

Machine learning applications typically model entities as vectors of numerical features so that they may be compared and analysed quantitively.  Typically the majority of the elements in these vectors are zeros. In the case of text mining applications, each document within a corpus is represented as a vector and its features represent the vocabulary of unique words.  A corpus of several thousand documents might utilise a vocabulary of hundreds of thousands (or perhaps even millions) of unique words but each document will typically only contain a couple of hundred unique words.  This means the number of non-zero values in the matrix might only be around 1%.

Sparse matrix formats capitalise on this premise by only storing the non-zero values thereby reducing both storage/memory requirements and processing effort for manipulating the data.  Sparse matrices can effectively be divided into 3 main categories:

1. Creational - Sparse matrix formats suited to construction and building of matrices.  Matrix formats in this category include DOK (Dictionary Of Keys) and COO (COOrdinate aka triplet).

2. Operational - Sparse matrix formats suited to arithmetic operations e.g. multiplication.  Matrix formats in this category include CSR (Compressed Sparse Row aka CRS - Compressed Row Storage) and CSC (Compressed Sparse Column aka CCS - Compressed Column Storage)

3. Specialised - Specialised matrix formats to efficiently store and manipulate specific sparsity patterns or data types.  Matrix formats in this category include DIA (DIAgonal) for diagonal matrices (where all the non-zero elements are situated along the diagonal) and Binary (bit) vectors and matrices for binary digits (bits - 1 or 0).

A common practice is to construct sparse matrices using a creational format e.g. DOK or COO and then convert them to an operational format e.g. CSR for arithmetic operations.

## Implemented Features

* Implemented Formats:
    * Sparse Matrix Formats:
        * [DOK (Dictionary Of Keys)](https://en.wikipedia.org/wiki/Sparse_matrix#Dictionary_of_keys_(DOK)) format
        * [COO (COOrdinate)](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) format (sometimes referred to as 'triplet')
        * [CSR (Compressed Sparse Row)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format
        * [CSC (Compressed Sparse Column)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)) format
        * [DIA (DIAgonal)](https://en.wikipedia.org/wiki/Sparse_matrix#Diagonal) format
        * sparse vectors
    * Other Formats:
        * [Binary (Bit) vectors](https://en.wikipedia.org/wiki/Bit_array) and matrices
* CSR matrix product from multiplication of 2 matrices (any implementations of the [Gonum mat.Matrix](https://godoc.org/gonum.org/v1/gonum/mat#Matrix) interface) with specific optimisations for the following scenarios:
    * CSR * DIA
    * DIA * CSR
    * DIA * mat.Matrix
    * mat.Matrix * DIA
    * CSR * CSR
    * CSR * CSC
    * CSR * mat.Matrix
* CSR result from addition of 2 matrices (any implementations of the [Gonum mat.Matrix](https://godoc.org/gonum.org/v1/gonum/mat#Matrix) interface) with specific optimisations for the following scenarios:
    * CSR + CSR
    * CSR + mat.Matrix
    * mat.Matrix + CSR
* Row and column slicing.
* Binary Vector and Matrix types for efficient storage and processing of binary/bit vectors (elements are 0 or 1) requiring a single bit of memory to represent each element.
* sparse optimised implementations of scalar vector dot product and normalisation (e.g. L2 norm).
* Implements standard Gonum API for iterating over non-zero elements

## Planned

* Further optimisations of CSR dot product for sparse matrix type operands (only considering non-zero values as with CSR operands currently), even as RHS operand ((AB)^T = B^T A^T)
* Consider implicitly converting sparse matrix operands to CSR/CSC for arithmetic operations
* Implement parallel/fast matrix multiplication algorithm for sparse matrices
* Implement further arithmetic operations e.g. subtract, divide, element wise multiplication, etc.
* Consider using native libraries to perform matrix arithmetic (as an option if available on host).
* Improve memory allocation for matrix multiplication - pre-calculating sparsity pattern for product and allocate storage in advance rather than incrementally.

