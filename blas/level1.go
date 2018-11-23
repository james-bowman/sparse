package blas

// Dusga (Sparse gather (x <- y|x)) gathers entries from the dense vector
// y into the sparse vector x using indx as the index values to gather
// and incy as the stride for y.
func Dusga(y []float64, incy int, x []float64, indx []int) {
	for i, index := range indx {
		x[i] = y[index*incy]
	}
}

// Dusgz (Sparse gather and zero (x <- y|x, y|x <- 0)) gathers entries
// from the dense vector y into the sparse vector x
// (as Usga()) and then sets the corresponding elements of y (y[indx[i]])
// to 0.  indx is used as the index values to gather and incy as the stride
// for y.
func Dusgz(y []float64, incy int, x []float64, indx []int) {
	for i, index := range indx {
		x[i] = y[index*incy]
		y[index*incy] = 0
	}
}

// Dussc (Sparse scatter (y|x <- x)) scatters enries into the dense vector y
// from the sparse vector x using indx as the index values to scatter to
// and incy as the stride for y. The function will panic if x and idx are
// different lengths.
func Dussc(x []float64, y []float64, incy int, indx []int) {
	for i, index := range indx {
		y[index*incy] = x[i]
	}
}
