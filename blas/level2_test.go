package blas

import (
	"testing"
)

func TestDusmv(t *testing.T) {
	tests := []struct {
		transA   bool
		alpha    float64
		a        *SparseMatrix
		x        []float64
		incx     int
		y        []float64
		incy     int
		expected []float64
	}{
		{
			transA: false,
			alpha:  1,
			a: &SparseMatrix{
				I: 3, J: 4,
				Indptr: []int{0, 2, 2, 5},
				Ind:    []int{0, 2, 0, 1, 3},
				Data:   []float64{1, 2, 3, 4, 5},
			},
			// 1, 0, 2, 0,
			// 0, 0, 0, 0,
			// 3, 4, 0, 5,
			x:        []float64{1, 2, 3, 4},
			incx:     1,
			y:        []float64{0, 0, 0},
			incy:     1,
			expected: []float64{7, 0, 31},
		},
		{
			transA: true,
			alpha:  1,
			a: &SparseMatrix{
				I: 4, J: 3,
				Indptr: []int{0, 2, 3, 4, 5},
				Ind:    []int{0, 2, 2, 0, 2},
				Data:   []float64{1, 3, 4, 2, 5},
			},
			// 1	0	3
			// 0	0	4
			// 2	0	0
			// 0	0	5
			x:        []float64{1, 2, 3, 4},
			incx:     1,
			y:        []float64{0, 0, 0},
			incy:     1,
			expected: []float64{7, 0, 31},
		},
		{
			transA: false,
			alpha:  2,
			a: &SparseMatrix{
				I: 3, J: 4,
				Indptr: []int{0, 2, 2, 5},
				Ind:    []int{0, 2, 0, 1, 3},
				Data:   []float64{1, 2, 3, 4, 5},
			},
			// 1, 0, 2, 0,
			// 0, 0, 0, 0,
			// 3, 4, 0, 5,
			x: []float64{
				1, 5,
				2, 5,
				3, 5,
				4, 5,
			},
			incx: 2,
			y: []float64{
				1, 5, 5, 5,
				2, 5, 5, 5,
				3, 5, 5, 5,
			},
			incy: 4,
			expected: []float64{
				15, 5, 5, 5,
				2, 5, 5, 5,
				65, 5, 5, 5,
			},
		},
		{
			transA: true,
			alpha:  2,
			a: &SparseMatrix{
				I: 4, J: 3,
				Indptr: []int{0, 2, 3, 4, 5},
				Ind:    []int{0, 2, 2, 0, 2},
				Data:   []float64{1, 3, 4, 2, 5},
			},
			// 1	0	3
			// 0	0	4
			// 2	0	0
			// 0	0	5
			x: []float64{
				1, 5,
				2, 5,
				3, 5,
				4, 5,
			},
			incx: 2,
			y: []float64{
				1, 5, 5, 5,
				2, 5, 5, 5,
				3, 5, 5, 5,
			},
			incy: 4,
			expected: []float64{
				15, 5, 5, 5,
				2, 5, 5, 5,
				65, 5, 5, 5,
			},
		},
	}

	for ti, test := range tests {
		Dusmv(test.transA, test.alpha, test.a, test.x, test.incx, test.y, test.incy)

		for i, v := range test.expected {
			if v != test.y[i] {
				t.Errorf("Test %d: Expected %f at %d but received %f", ti, v, i, test.y[i])
			}
		}
	}
}
