package sparse

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

func Example() {
	// Construct a new DOK (Dictionary Of Keys) matrix
	dokMatrix := NewDOK(3, 2)

	// Populate it with some non-zero values
	dokMatrix.Set(0, 0, 5)
	dokMatrix.Set(2, 1, 7)

	// Demonstrate accessing values (could use mat64.Formatted() to
	// pretty print but this demonstrates element access)
	m, n := dokMatrix.Dims()
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			fmt.Printf("%.0f", dokMatrix.At(i, j))
			if j < n-1 {
				fmt.Printf(" ")
			}
		}
		fmt.Printf("\n")
	}

	// Convert DOK matrix to mat64.Dense just for fun
	// (not required for upcoming multiplication operation)
	denseMatrix := dokMatrix.ToDense()

	// Confirm the two matrices in different formats are equal
	// Using the mat64.Equal function
	if !mat64.Equal(dokMatrix, denseMatrix) {
		fmt.Println("DOK and converted Dense are not equal")
	}

	// Create a random 10x25 CSR (Compressed Sparse Row) matrix with
	// density of 0.5 (half the elements will be non-zero)
	csrMatrix := Random(CSRFormat, 2, 3, 0.5)

	// Create a new CSR (Compressed Sparse Row) matrix
	csrProduct := &CSR{}

	// Multiply the 2 matrices together and store the result in the
	// receiver
	csrProduct.Mul(csrMatrix, denseMatrix)

	// Output: 5 0
	//0 0
	//0 7
}
