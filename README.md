# Sparse matrix formats
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GoDoc](https://godoc.org/github.com/james-bowman/sparse?status.svg)](https://godoc.org/github.com/james-bowman/sparse)
[![Build Status](https://travis-ci.org/james-bowman/sparse.svg?branch=master)](https://travis-ci.org/james-bowman/sparse)
[![Go Report Card](https://goreportcard.com/badge/github.com/james-bowman/sparse)](https://goreportcard.com/report/github.com/james-bowman/sparse)
[![codecov](https://codecov.io/gh/james-bowman/sparse/branch/master/graph/badge.svg)](https://codecov.io/gh/james-bowman/sparse)
[![Mentioned in Awesome Go](https://awesome.re/mentioned-badge-flat.svg)](https://github.com/avelino/awesome-go)
[![Sourcegraph](https://sourcegraph.com/github.com/james-bowman/sparse/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/sparse?badge)

Implementations of selected sparse matrix formats for linear algebra supporting scientific and machine learning applications.  Compatible with the APIs in the [Gonum](http://www.gonum.org/) package and interoperable with Gonum dense matrix types.

## Overview

Machine learning applications typically model entities as vectors of numerical features so that they may be compared and analysed quantitively.  Typically the majority of the elements in these vectors are zeros. In the case of text mining applications, each document within a corpus is represented as a vector and its features represent the vocabulary of unique words.  A corpus of several thousand documents might utilise a vocabulary of hundreds of thousands (or perhaps even millions) of unique words but each document will typically only contain a couple of hundred unique words.  This means the number of non-zero values in the matrix might only be around 1%.

Sparse matrix formats capitalise on this premise by only storing the non-zero values thereby reducing both storage/memory requirements and processing effort for manipulating the data.

## Features

* Implementations of [Sparse BLAS](http://www.netlib.org/blas/blast-forum/chapter3.pdf) standard routines.
* Compatible with [Gonum's APIs](https://godoc.org/gonum.org/v1/gonum/mat) and interoperable with Gonum's dense matrix types.
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
* Matrix multiplication, addition and subtraction and vector dot products.

## Usage

The sparse matrices in this package implement the Gonum `Matrix` interface and so are fully interoperable and mutually compatible with the Gonum APIs and dense matrix types.

``` go
// Construct a new 3x2 DOK (Dictionary Of Keys) matrix
dokMatrix := sparse.NewDOK(3, 2)

// Populate it with some non-zero values
dokMatrix.Set(0, 0, 5)
dokMatrix.Set(2, 1, 7)

// Demonstrate accessing values (could use Gonum's mat.Formatted()
// function to pretty print but this demonstrates element access)
m, n := dokMatrix.Dims()
for i := 0; i < m; i++ {
    for j := 0; j < n; j++ {
        fmt.Printf("%.0f,", dokMatrix.At(i, j))
    }
    fmt.Printf("\n")
}

// Convert DOK matrix to CSR (Compressed Sparse Row) matrix
// just for fun (not required for upcoming multiplication operation)
csrMatrix := dokMatrix.ToCSR()

// Create a random 2x3 COO (COOrdinate) matrix with
// density of 0.5 (half the elements will be non-zero)
cooMatrix := sparse.Random(sparse.COOFormat, 2, 3, 0.5)

// Convert CSR matrix to Gonum mat.Dense matrix just for fun
// (not required for upcoming multiplication operation)
// then transpose so it is the right shape/dimensions for
// multiplication with the original CSR matrix
denseMatrix := csrMatrix.ToDense().T()

// Multiply the 2 matrices together and store the result in the
// sparse receiver (multiplication with sparse product)
var csrProduct sparse.CSR
csrProduct.Mul(csrMatrix, cooMatrix)

// As an alternative, use the sparse BLAS routines for efficient
// sparse matrix multiplication with a Gonum mat.Dense product
// (multiplication with dense product)
denseProduct := sparse.MulMatMat(false, 1, csrMatrix, denseMatrix, nil)
```

## Installation

With Go installed, package installation is performed using go get.

```
go get -u github.com/james-bowman/sparse/...
```

## Acknowledgements

* [Gonum](http://www.gonum.org/)
* [Netlib. BLAS. Chapter 3: Sparse BLAS](http://www.netlib.org/blas/blast-forum/chapter3.pdf)
* J.R. Gilbert, C. Moler, and R. Schreiber. Sparse matrices in
MATLAB: Design and implementation. SIAM Journal on Matrix Analysis and
Applications, 13:333–356, 1992.
* F.G. Gustavson. Some basic techniques for solving sparse systems
of linear equations. In D.J. Rose and R.A. Willoughby, eds., Sparse Matrices and
Their Applications, 41–52, New York: Plenum Press, 1972.
* F.G. Gustavson. Efficient algorithm to perform sparse matrix
multiplication. IBM Technical Disclosure Bulletin, 20:1262–1264, 1977.
* [Wikipedia. Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix)
* [A. Fog. 2. Optimizing subroutines in assembly language An optimization guide for x86 platforms, 1996.](https://www.agner.org/optimize/optimizing_assembly.pdf)

## See Also

* [gonum/gonum](https://github.com/gonum/gonum)

## License

MIT