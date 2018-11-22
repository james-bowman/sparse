// +build !amd64 noasm appengine safe

package blas

// Dusdot (Sparse dot product (r <- x^T * y)) calculates the dot product of
// sparse vector x and dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Dusdot(x []float64, indx []int, y []float64, incy int) (dot float64) {
	for i, index := range indx {
		dot += x[i] * y[index*incy]
	}
	return
}
