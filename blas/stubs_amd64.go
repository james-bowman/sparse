//+build !noasm,!appengine,!safe

package blas

// Dusdot (Sparse dot product (r <- x^T * y)) calculates the dot product of
// sparse vector x and dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Dusdot(x []float64, indx []int, y []float64, incy int) (dot float64)

// Dusaxpy (Sparse update (y <- alpha * x + y)) scales the sparse vector x by
// alpha and adds the result to the dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Dusaxpy(alpha float64, x []float64, indx []int, y []float64, incy int)
