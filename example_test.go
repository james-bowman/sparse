package sparse_test

import (
	"fmt"

	"github.com/james-bowman/sparse"
)

func Example() {
	// Construct a new 3x2 DOK (Dictionary Of Keys) matrix
	dokMatrix := sparse.NewDOK(3, 2)

	// Populate it with some non-zero values
	dokMatrix.Set(0, 0, 5)
	dokMatrix.Set(2, 1, 7)

	// Demonstrate accessing values (could use mat.Formatted() to
	// pretty print but this demonstrates element access)
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
	rows, cols := denseProduct.Dims()
	if rows != 2 && cols != 3 {
		fmt.Printf("Expected product 2x3 but received %dx%d\n", rows, cols)
	}

	// Output: 5,0,
	//0,0,
	//0,7,
}
