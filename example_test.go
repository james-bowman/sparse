package sparse_test

import (
	"fmt"
	"math"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
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

func ExampleLU() {
	// ExampleLU created on example from:
	// https://www.gnu.org/software/gsl/doc/html/splinalg.html#examples

	size := 2000                 // amount intermediate points
	n := size - 2                // without boundary condition
	A := sparse.NewDOK(n, n)     // sparse matrix
	f := mat.NewDense(n, 1, nil) // right hand size vector
	d := mat.NewDense(n, 1, nil) // solution vector

	for i := 0; i < n; i++ {
		xi := float64(i+1) / float64(size-1)
		f.Set(i, 0, -math.Pow(math.Pi, 2.0)*math.Sin(math.Pi*xi))

		dh := math.Pow(float64(size-1), 2.0)
		A.Set(i, i, -2*dh)
		if i+1 < n {
			A.Set(i, i+1, 1*dh)
			A.Set(i+1, i, 1*dh)
		}
	}

	// LU decomposition
	var lu mat.LU
	lu.Factorize(A)

	err := lu.Solve(d, false, f)
	if err != nil {
		fmt.Printf("err = %v", err)
		return
	}

	// compare with except result
	fmt.Printf("%-8s %-12s %-12s %-8s\n",
		"x,rad.", "Solved", "Expect", "Delta, %")
	for i := 0; i < n; i += 100 {
		h := 1.0 / float64(size-1)
		xi := float64(i+1) * h
		expect := math.Sin(math.Pi * xi)
		delta := (d.At(i, 0) - expect) / expect * 100
		fmt.Printf("%-8f %-12f %-12f %-8e\n",
			xi, d.At(i, 0), expect, delta)
	}

	// Output:
	// x,rad.   Solved       Expect       Delta, %
	// 0.000500 0.001572     0.001572     2.058220e-05
	// 0.050525 0.158064     0.158064     2.058220e-05
	// 0.100550 0.310661     0.310661     2.058220e-05
	// 0.150575 0.455600     0.455600     2.058220e-05
	// 0.200600 0.589310     0.589310     2.058221e-05
	// 0.250625 0.708495     0.708495     2.058222e-05
	// 0.300650 0.810216     0.810216     2.058222e-05
	// 0.350675 0.891968     0.891968     2.058222e-05
	// 0.400700 0.951734     0.951734     2.058223e-05
	// 0.450725 0.988042     0.988042     2.058223e-05
	// 0.500750 0.999997     0.999997     2.058223e-05
	// 0.550775 0.987305     0.987304     2.058223e-05
	// 0.600800 0.950277     0.950276     2.058223e-05
	// 0.650825 0.889826     0.889826     2.058223e-05
	// 0.700850 0.807444     0.807444     2.058223e-05
	// 0.750875 0.705160     0.705159     2.058222e-05
	// 0.800900 0.585494     0.585494     2.058222e-05
	// 0.850925 0.451398     0.451398     2.058223e-05
	// 0.900950 0.306176     0.306176     2.058223e-05
	// 0.950975 0.153407     0.153407     2.058223e-05
}
