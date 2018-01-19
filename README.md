# Sparse matrix formats
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/sparse?status.svg)](https://godoc.org/github.com/james-bowman/sparse) 
[![Build Status](https://travis-ci.org/james-bowman/sparse.svg?branch=master)](https://travis-ci.org/james-bowman/sparse)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/sparse)](https://goreportcard.com/report/github.com/james-bowman/sparse)
[![GoCover](https://gocover.io/_badge/github.com/james-bowman/sparse)](https://gocover.io/github.com/james-bowman/sparse) 
<!--[![wercker status](https://app.wercker.com/status/33d6c1400cca054635f46a8f44c14c42/s/master "wercker status")](https://app.wercker.com/project/byKey/33d6c1400cca054635f46a8f44c14c42) 
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp) [![Sourcegraph Badge](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)-->

Implementations of selected sparse matrix formats for linear algebra supporting scientific and machine learning applications.

Machine learning applications typically model entities as vectors of numerical features so that they may be compared and analysed quantitively.  Typically the majority of the elements in these vectors are zeros. In the case of text mining applications, each document within a corpus is represented as a vector and its features represent the vocabulary of unique words.  A corpus of several thousand documents might utilise a vocabulary of hundreds of thousands (or perhaps even millions) of unique words but each document will typically only contain a couple of hundred unique words.  This means the number of non-zero values in the matrix might only be around 1%.

Sparse matrix formats capitalise on this premise by only storing the non-zero values thereby reducing both storage/memory requirements and processing effort for manipulating the data.  Sparse matrices can effectively be divided into 3 main categories:

1. Creational - Sparse matrix formats suited to construction and building of matrices.  Matrix formats in this category include DOK (Dictionary Of Keys) and COO (COOrdinate aka triplet).

2. Operational - Sparse matrix formats suited to arithmetic operations e.g. multiplication.  Matrix formats in this category include CSR (Compressed Sparse Row aka CRS - Compressed Row Storage) and CSC (Compressed Sparse Column aka CCS - Compressed Column Storage)

3. Specialised - Specialised matrix formats to efficiently store and manipulate specific sparsity patterns or data types.  Matrix formats in this category include DIA (DIAgonal) for diagonal matrices (where all the non-zero elements are situated along the diagonal) and Binary (bit) vectors and matrices for binary digits (bits - 1 or 0).

A common practice is to construct sparse matrices using a creational format e.g. DOK or COO and then convert them to an operational format e.g. CSR for arithmetic operations.

## Implemented Features

* Implemented Formats:
    * Sparse Matrix Formats:
        * DOK (Dictionary Of Keys) format
        * COO (COOrdinate) format (sometimes referred to as 'triplet')
        * CSR (Compressed Sparse Row) format
        * CSC (Compressed Sparse Column) format
        * DIA (DIAgonal) format
        * sparse vectors
    * Other Formats:
        * Binary (Bit) vectors and matrices
* CSR dot product (matrix multiplication) of 2 matrices (with optimisations for operands of type DIA (as LHS or RHS operand), CSC (RHS operand only when LHS operand is CSR) and CSR (LHS operand only) but supporting any implementation of [Matrix](https://github.com/gonum/gonum/blob/d7342e68fbbe64d7dbbdc0feb4ecf60500444cdc/mat/matrix.go) interface from [gonum](https://github.com/gonum/gonum)).
* CSR addition of 2 matrices (with optimisations for operands of type CSR but supporting any implementation of [Matrix](https://github.com/gonum/gonum/blob/d7342e68fbbe64d7dbbdc0feb4ecf60500444cdc/mat/matrix.go) interface from [gonum](https://github.com/gonum/gonum)).
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

