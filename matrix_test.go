package sparse

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMulMatMat(t *testing.T) {
	// A:
	// 1, 0, 2, 0,
	// 0, 0, 0, 0,
	// 0, 3, 4, 5,
	permsA := []struct {
		name   string
		matrix BlasCompatibleSparser
	}{
		{
			name: "CSR",
			matrix: NewCSR(
				3, 4,
				[]int{0, 2, 2, 5},
				[]int{0, 2, 1, 2, 3},
				[]float64{1, 2, 3, 4, 5}),
		},
		{
			name: "CSC",
			matrix: NewCSC(
				3, 4,
				[]int{0, 1, 2, 4, 5},
				[]int{0, 2, 0, 2, 2},
				[]float64{1, 3, 2, 4, 5}),
		},
		{
			name: "COO",
			matrix: NewCOO(
				3, 4,
				[]int{0, 2, 0, 2, 2},
				[]int{0, 3, 2, 1, 2},
				[]float64{1, 5, 2, 3, 4}),
		},
	}

	permsB := []struct {
		name   string
		matrix mat.Matrix
	}{
		{
			name: "Dense",
			matrix: mat.NewDense(4, 5, []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 1,
				2, 3, 4, 5, 6,
				7, 8, 9, 0, 1,
			}),
		},
		{
			name: "CSR",
			matrix: NewCSR(4, 5,
				[]int{0, 5, 10, 15, 19},
				[]int{
					0, 1, 2, 3, 4,
					0, 1, 2, 3, 4,
					0, 1, 2, 3, 4,
					0, 1, 2, 4,
				},
				[]float64{
					1, 2, 3, 4, 5,
					6, 7, 8, 9, 1,
					2, 3, 4, 5, 6,
					7, 8, 9, 1,
				}),
		},
	}

	tests := []struct {
		alpha  float64
		c      *mat.Dense
		er, ec int
		eData  []float64
	}{
		{ // C = 0 * A * B
			alpha: 0,
			c: mat.NewDense(3, 5, []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			}),
			er: 3, ec: 5,
			eData: []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			},
		},
		{ // C = 1 * A * B
			alpha: 1,
			c: mat.NewDense(3, 5, []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			}),
			er: 3, ec: 5,
			eData: []float64{
				5, 8, 11, 14, 17,
				0, 0, 0, 0, 0,
				61, 73, 85, 47, 32,
			},
		},
		{ // C = 2 * A * B
			alpha: 2,
			c: mat.NewDense(3, 5, []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			}),
			er: 3, ec: 5,
			eData: []float64{
				10, 16, 22, 28, 34,
				0, 0, 0, 0, 0,
				122, 146, 170, 94, 64,
			},
		},
		{ // C = 1 * A * B + C
			alpha: 1,
			c: mat.NewDense(3, 5, []float64{
				1, 0, 5, 0, 8,
				2, 0, 3, 0, 4,
				0, 0, 0, 7, 0,
			}),
			er: 3, ec: 5,
			eData: []float64{
				6, 8, 16, 14, 25,
				2, 0, 3, 0, 4,
				61, 73, 85, 54, 32,
			},
		},
	}

	for ti, test := range tests {
		for _, transA := range []bool{false, true} {
			for _, b := range permsB {
				for _, a := range permsA {
					amat := a.matrix
					var transInd string
					if transA {
						amat = amat.T().(BlasCompatibleSparser)
						transInd = "^T"
					}
					ccopy := mat.DenseCopyOf(test.c)
					c := MulMatMat(transA, test.alpha, amat, b.matrix, ccopy)

					cr, cc := c.Dims()
					if cr != test.er || cc != test.ec {
						t.Errorf("Test %d (%s%s x %s): Incorrect dimensions: expected %d x %d but received %d x %d", ti+1, a.name, transInd, b.name, test.er, test.ec, cr, cc)
					}

					craw := c.RawMatrix()

					for i, v := range test.eData {
						if v != craw.Data[i] {
							e := mat.NewDense(test.er, test.ec, test.eData)
							t.Errorf("Test %d (%s%s x %s): Failed, expected\n%v\n but received \n%v", ti+1, a.name, transInd, b.name, mat.Formatted(e), mat.Formatted(c))
							break
						}
					}
				}
			}
		}
	}
}
