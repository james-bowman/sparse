// +build !amd64 noasm appengine safe

package blas

// Dusaxpy (Sparse update (y <- alpha * x + y)) scales the sparse vector x by
// alpha and adds the result to the dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Dusaxpy(alpha float64, x []float64, indx []int, y []float64, incy int) {
	for i, index := range indx {
		y[index*incy] += alpha * x[i]
	}
}
