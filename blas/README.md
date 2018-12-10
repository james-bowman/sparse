# Sparse matrix formats
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![GoDoc](https://godoc.org/github.com/james-bowman/sparse/blas?status.svg)](https://godoc.org/github.com/james-bowman/sparse/blas) 
[![Sourcegraph](https://sourcegraph.com/github.com/james-bowman/sparse/blas/-/badge.svg)](https://sourcegraph.com/github.com/james-bowman/sparse/blas?badge)

Implementation of sparse BLAS (Basic Linear Algebra Subprograms) routines in Go for 
sparse matrix arithmetic.  See http://www.netlib.org/blas/blast-forum/chapter3.pdf for more details.  Includes optimised assembler implementations of key kernel operations (vector dot product and Axpy operations).

## Indicative Benchmarks

| Operation                        |  Pure Go     |   Assembler  |
| -------------------------------- | ------------ | ------------ |
| Dusdot (with increment/stride)   |  1340 ns/op  |   978 ns/op  |
| Dusdot (unitary)                 |  1215 ns/op  |   662 ns/op  |
| Dusaxpy (with increment/stride)  |  1944 ns/op  |  1769 ns/op  |
| Dusaxpy (unitary)                |  1091 ns/op  |   979 ns/op  |
