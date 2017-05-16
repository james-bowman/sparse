# Sparse matrix formats
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/sparse?status.svg)](https://godoc.org/github.com/james-bowman/sparse) 

<!--[![wercker status](https://app.wercker.com/status/33d6c1400cca054635f46a8f44c14c42/s/master "wercker status")](https://app.wercker.com/project/byKey/33d6c1400cca054635f46a8f44c14c42) 
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/nlp)](https://goreportcard.com/report/github.com/james-bowman/nlp) [![Sourcegraph Badge](https://sourcegraph.com/github.com/james-bowman/nlp/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/nlp?badge)-->

Implementation of selected sparse matrix formats to support scientific and machine learning applications.  

Machine learning applications typically model entities as mathematical vectors of features so that they may be compared and analysed quantitively.  Typically the majority of the elements in these vectors are zeros. In the case of text mining applications, each document within a corpus is represented as a vector and its features represent the vocabulary of unique words.  A corpus of several thousand documents might utilise a vocabulary of hundreds of thousands (or perhaps even millions) of unique words but each document will typically only contain a couple of hundred unique words.  This means the number of non-zero values in the matrix might only be around 1%.

Sparse matrix formats capitalise on this by only storing the non-zero values thereby reducing both storage/memory requirements and processing effort for manipulating the data.  Sparse matrices can effectively be divided into 3 main categories:

1. Creational - Sparse matrix formats suited to construction and building of matrices.  Matrix formats in this category include DOK (Dictionary Of Keys) and COO (COOrdinate).

2. Operational - Sparse matrix formats suited to arithmetic operations e.g. multiplication.  Matrix formats in this category include CSR (Compact Sparse Row aka CRS - Compact Row Storage) and CSC (Compact Sparse Column aka CCS - Compact Column Storage)

3. Specialised - Specialised matrix formats suiting specific sparsity patterns.  Matrix formats in this category include DIA (DIAgonal) for efficiently storing and manipulating symmetric diagonal matrices.

A common practice is to construct sparse matrices using a creational format e.g. DOK or COO and then converting to an operational format e.g. CSR for arithmetic operations.

## Features

* DOK (Dictionary Of Keys) format
* COO (COOrdinate) format
* CSR (Compressed Sparse Row) format
* CSC (Compressed Sparse Column) format
* DIA (DIAgonal) format
* CSR multiplication of matrices (with optimisations for DIA and CSR matrix types)

## Planned

* Further optimisations of CSR multiplication for sparse matrix types, even for RHS operand ((AB)^T = A^T * B^T)
* Consider implicitly converting sparse matrix type operands to CSR/CSC for arithmetic operations
* Further arithmetic operations e.g. add, subtract, divide, etc.
* Further optimisation of arithmetic operations for improved speed and efficient storage
* Improve memory allocation for matrix multiplication - pre-calculating sparsity pattern for product and allocate storage in advance.
