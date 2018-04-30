package blas

// Dusmm (Sparse matrix multiply (C <- alpha * A * B + C Or C <- alpha * A^T * B + C))
// multiplies a dense matrix B by a sparse matrix A (or its transpose), and
// adds it to a dense matrix operand C.  C is modified to hold the result of
// operation.  k Represents the number of columns in matrices B and C and ldb and ldc
// are the spans to be used for indexing into matrices B and C respectively.
func Dusmm(transA bool, k int, alpha float64, a *SparseMatrix, b []float64, ldb int, c []float64, ldc int) {
	// A is m x n, B is n x k, C is m x k
	if alpha == 0 {
		return
	}

	// Perform k matvecs: i-th column of C gets A*(i-th column of B)
	for i := 0; i < k; i++ {
		Dusmv(transA, alpha, a, b[i:], ldb, c[i:], ldc)
	}
}
