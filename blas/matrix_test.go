package blas

import (
	"testing"
)

func TestSparseMatrixSet(t *testing.T) {
	var tests = []struct {
		r, c   int
		data   []float64
		i, j   int
		v      float64
		result []float64
	}{
		{ // 0 at start of matrix set to non-zero
			r: 3, c: 4,
			data: []float64{
				0, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 6,
			},
			i: 0, j: 0,
			v: 5,
			result: []float64{
				5, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 6,
			},
		},
		{ // 0 as first element of row set to non-zero
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 6,
			},
			i: 2, j: 0,
			v: 5,
			result: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				5, 0, 3, 6,
			},
		},
		{ // 0 as first non-zero element of row set to non-zero
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 6,
			},
			i: 2, j: 1,
			v: 5,
			result: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				0, 5, 3, 6,
			},
		},
		{ // 0 as non-zero element in middle of row/col set to non-zero
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 7,
				0, 0, 3, 6,
			},
			i: 1, j: 2,
			v: 5,
			result: []float64{
				1, 0, 0, 0,
				0, 2, 5, 7,
				0, 0, 3, 6,
			},
		},
		{ // non-zero value updated
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
			i: 2, j: 2,
			v: 5,
			result: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 5, 6,
			},
		},
		{ // 0 at end of row set to non-zero
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 0,
			},
			i: 2, j: 3,
			v: 5,
			result: []float64{
				1, 0, 0, 0,
				0, 2, 1, 0,
				0, 0, 3, 5,
			},
		},
		{ // 0 on all zero row/column set to non-zero
			r: 3, c: 4,
			data: []float64{
				1, 0, 2, 0,
				0, 0, 0, 0,
				0, 0, 3, 6,
			},
			i: 1, j: 1,
			v: 5,
			result: []float64{
				1, 0, 2, 0,
				0, 5, 0, 0,
				0, 0, 3, 6,
			},
		},
	}

	for ti, test := range tests {
		matrix := SparseMatrix{
			I: test.r, J: test.c,
		}
		matrix.Indptr = make([]int, test.r+1)
		for i := 0; i < test.r; i++ {
			for j := 0; j < test.c; j++ {
				v := test.data[i*test.c+j]
				if v != 0 {
					matrix.Ind = append(matrix.Ind, j)
					matrix.Data = append(matrix.Data, v)
				}
			}
			matrix.Indptr[i+1] = len(matrix.Data)
		}

		matrix.Set(test.i, test.j, test.v)

		for i := 0; i < test.r; i++ {
			for j := 0; j < test.c; j++ {
				if matrix.At(i, j) != test.result[i*test.c+j] {
					t.Errorf("Test %d: Expected %f at %d,%d but found %f", ti+1, test.result[i*test.c+j], i, j, matrix.At(i, j))
				}
			}
		}
	}
}
