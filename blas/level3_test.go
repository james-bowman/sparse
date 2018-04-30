package blas

import (
	"testing"
)

// A is m x n, B is n x k, C is m x k
func TestDusmm(t *testing.T) {
	tests := []struct {
		transA bool
		alpha  float64
		a      *SparseMatrix
		br, bc int
		bData  []float64
		cr, cc int
		cData  []float64
		er, ec int
		eData  []float64
	}{
		{
			transA: false,
			alpha:  1,
			a: &SparseMatrix{
				I: 3, J: 4,
				Ind:    []int{0, 2, 1, 2, 3},
				Indptr: []int{0, 2, 2, 5},
				Data:   []float64{1, 2, 3, 4, 5},
			},
			br: 4, bc: 5,
			bData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 1,
				2, 3, 4, 5, 6,
				7, 8, 9, 0, 1,
			},
			cr: 3, cc: 5,
			cData: []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			},
			er: 3, ec: 5,
			eData: []float64{
				5, 8, 11, 14, 17,
				0, 0, 0, 0, 0,
				61, 73, 85, 47, 32,
			},
		},
		{
			transA: true,
			alpha:  1,
			a: &SparseMatrix{
				I: 4, J: 3,
				Ind:    []int{0, 2, 0, 2, 2},
				Indptr: []int{0, 1, 2, 4, 5},
				Data:   []float64{1, 3, 2, 4, 5},
			},
			br: 4, bc: 5,
			bData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 1,
				2, 3, 4, 5, 6,
				7, 8, 9, 0, 1,
			},
			cr: 3, cc: 5,
			cData: []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			},
			er: 3, ec: 5,
			eData: []float64{
				5, 8, 11, 14, 17,
				0, 0, 0, 0, 0,
				61, 73, 85, 47, 32,
			},
		},
		{
			transA: false,
			alpha:  2,
			a: &SparseMatrix{
				I: 3, J: 4,
				Ind:    []int{0, 2, 1, 2, 3},
				Indptr: []int{0, 2, 2, 5},
				Data:   []float64{1, 2, 3, 4, 5},
			},
			br: 4, bc: 5,
			bData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 1,
				2, 3, 4, 5, 6,
				7, 8, 9, 0, 1,
			},
			cr: 3, cc: 5,
			cData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			},
			er: 3, ec: 5,
			eData: []float64{
				11, 18, 25, 32, 39,
				6, 7, 8, 9, 10,
				133, 158, 183, 108, 79,
			},
		},
		{
			transA: true,
			alpha:  2,
			a: &SparseMatrix{
				I: 4, J: 3,
				Ind:    []int{0, 2, 0, 2, 2},
				Indptr: []int{0, 1, 2, 4, 5},
				Data:   []float64{1, 3, 2, 4, 5},
			},
			br: 4, bc: 5,
			bData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 1,
				2, 3, 4, 5, 6,
				7, 8, 9, 0, 1,
			},
			cr: 3, cc: 5,
			cData: []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			},
			er: 3, ec: 5,
			eData: []float64{
				11, 18, 25, 32, 39,
				6, 7, 8, 9, 10,
				133, 158, 183, 108, 79,
			},
		},
	}

	for ti, test := range tests {
		Dusmm(test.transA, test.ec, test.alpha, test.a, test.bData, test.bc, test.cData, test.cc)

		for i, v := range test.eData {
			if v != test.cData[i] {
				t.Errorf("Test %d: Failed at index %d, expected %f but receied %f", ti+1, i, v, test.cData[i])
			}
		}
	}
}
