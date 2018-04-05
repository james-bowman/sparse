package blas

// Usmv (sparse matrix / vector multiply (y <- alpha * A * x + y Or y <- alpha * A^T * x + y))
// multiplies a dense vector x by sparse matrix a (or its transpose), and adds it
// to the dense vector y.  transA is a boolean indicating whether to transpose (true) a.
// alpha is used to scale a and incx and incy represent the span to be used for indexing into
// vectors x and y respectively.
func Usmv(transA bool, alpha float64, a *SparseMatrix, x []float64, incx int, y []float64, incy int) {
	r := a.I

	if alpha == 0 {
		return
	}

	if transA {
		for i := 0; i < r; i++ {
			begin, end := a.Indptr[i], a.Indptr[i+1]
			Usaxpy(alpha*x[i*incx], a.Data[begin:end], a.Ind[begin:end], y, incy)
		}
	} else {
		for i := 0; i < r; i++ {
			begin, end := a.Indptr[i], a.Indptr[i+1]
			y[i*incy] += alpha * Usdot(a.Data[begin:end], a.Ind[begin:end], x, incx)
		}
	}
}
