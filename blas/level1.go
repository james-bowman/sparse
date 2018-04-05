package blas

// Usdot (Sparse dot product (r <- x^T * y)) calculates the dot product of
// sparse vector x and dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Usdot(x []float64, indx []int, y []float64, incy int) float64 {
	var dot float64
	for i, index := range indx {
		dot += x[i] * y[index*incy]
	}
	return dot
}

// Usaxpy (Sparse update (y <- a * x + y)) scales the sparse vector x by
// alpha and adds the result to the dense vector y.  indx is used as the index
// values to gather and incy as the stride for y.
func Usaxpy(alpha float64, x []float64, indx []int, y []float64, incy int) {
	if alpha == 0 {
		return
	}
	for i, index := range indx {
		y[index*incy] += alpha * x[i]
	}
}

// Usga (Sparse gather (x <- y|x)) gathers entries from the dense vector
// y into the sparse vector x using indx as the index values to gather
// and incy as the stride for y.
func Usga(y []float64, incy int, x []float64, indx []int) {
	for i, index := range indx {
		x[i] = y[index*incy]
	}
}

// Usgz (Sparse gather and zero (x <- y|x, y|x <- 0)) gathers entries
// from the dense vector y into the sparse vector x
// (as Usga()) and then sets the corresponding elements of y (y[indx[i]])
// to 0.  indx is used as the index values to gather and incy as the stride
// for y.
func Usgz(y []float64, incy int, x []float64, indx []int) {
	for i, index := range indx {
		x[i] = y[index*incy]
		y[index*incy] = 0
	}
}

// Ussc (Sparse scatter (y|x <- x)) scatters enries into the dense vector y
// from the sparse vector x using indx as the index values to scatter to
// and incy as the stride for y. The function will panic if x and idx are
// different lengths.
func Ussc(x []float64, y []float64, incy int, indx []int) {
	for i, index := range indx {
		y[index*incy] = x[i]
	}
}
