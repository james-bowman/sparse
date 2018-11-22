package blas

import (
	"testing"
)

func TestDusdot(t *testing.T) {
	tests := []struct {
		x        []float64
		indx     []int
		y        []float64
		incy     int
		expected float64
	}{
		{
			x:        []float64{1, 3, 4},
			indx:     []int{0, 2, 3},
			y:        []float64{1, 2, 3, 4},
			incy:     1,
			expected: 26,
		},
		{
			x:        []float64{1, 3, 4, 5},
			indx:     []int{0, 2, 3, 4},
			y:        []float64{1, 2, 3, 4, 5},
			incy:     1,
			expected: 51,
		},
		{
			x:    []float64{1, 3, 4},
			indx: []int{0, 2, 3},
			y: []float64{
				1, 5, 5, 5,
				2, 5, 5, 5,
				3, 5, 5, 5,
				4, 5, 5, 5,
			},
			incy:     4,
			expected: 26,
		},
	}

	for ti, test := range tests {
		dot := Dusdot(test.x, test.indx, test.y, test.incy)

		if dot != test.expected {
			t.Errorf("Test %d: Wanted %f but received %f", ti+1, test.expected, dot)
		}
	}
}

func TestDusaxpy(t *testing.T) {
	tests := []struct {
		alpha    float64
		x        []float64
		indx     []int
		y        []float64
		incy     int
		expected []float64
	}{
		{
			alpha:    1,
			x:        []float64{1, 3, 4},
			indx:     []int{0, 2, 3},
			y:        []float64{0, 0, 0, 0},
			incy:     1,
			expected: []float64{1, 0, 3, 4},
		},
		{
			alpha:    1,
			x:        []float64{1, 3, 4},
			indx:     []int{0, 2, 3},
			y:        []float64{1, 2, 3, 4},
			incy:     1,
			expected: []float64{2, 2, 6, 8},
		},
		{
			alpha: 2,
			x:     []float64{1, 3, 4},
			indx:  []int{0, 2, 3},
			y: []float64{
				1, 5, 5, 5,
				2, 5, 5, 5,
				3, 5, 5, 5,
				4, 5, 5, 5,
			},
			incy: 4,
			expected: []float64{
				3, 5, 5, 5,
				2, 5, 5, 5,
				9, 5, 5, 5,
				12, 5, 5, 5,
			},
		},
	}

	for ti, test := range tests {
		Dusaxpy(test.alpha, test.x, test.indx, test.y, test.incy)

		for i, y := range test.y {
			if y != test.expected[i] {
				t.Errorf("Test %d: Wanted %f at %d but received %f", ti+1, test.expected[i], i, y)
			}
		}
	}
}

func TestDusga(t *testing.T) {
	tests := []struct {
		x        []float64
		indx     []int
		y        []float64
		incy     int
		expected []float64
	}{
		{
			x:        []float64{0, 0, 0},
			indx:     []int{0, 2, 3},
			y:        []float64{1, 2, 3, 4},
			incy:     1,
			expected: []float64{1, 3, 4},
		},
		{
			x:    []float64{5, 5, 5},
			indx: []int{0, 2, 3},
			y: []float64{
				1, 5, 5, 5,
				5, 5, 5, 5,
				3, 5, 5, 5,
				4, 5, 5, 5,
			},
			incy:     4,
			expected: []float64{1, 3, 4},
		},
	}

	for ti, test := range tests {
		Dusga(test.y, test.incy, test.x, test.indx)

		for i := range test.indx {
			if test.x[i] != test.expected[i] {
				t.Errorf("Test %d: Wanted %f at %d but received %f", ti+1, test.expected[i], i, test.x[i])
			}
		}
	}
}

func TestDusgz(t *testing.T) {
	tests := []struct {
		x        []float64
		indx     []int
		y        []float64
		incy     int
		expected []float64
	}{
		{
			x:        []float64{0, 0, 0},
			indx:     []int{0, 2, 3},
			y:        []float64{1, 2, 3, 4},
			incy:     1,
			expected: []float64{1, 3, 4},
		},
		{
			x:    []float64{5, 5, 5},
			indx: []int{0, 2, 3},
			y: []float64{
				1, 5, 5, 5,
				5, 5, 5, 5,
				3, 5, 5, 5,
				4, 5, 5, 5,
			},
			incy:     4,
			expected: []float64{1, 3, 4},
		},
	}

	for ti, test := range tests {
		Dusgz(test.y, test.incy, test.x, test.indx)

		for i, v := range test.indx {
			if test.x[i] != test.expected[i] {
				t.Errorf("Test %d: Wanted %f at %d but received %f", ti+1, test.expected[i], i, test.x[i])
			}
			if test.y[v*test.incy] != 0 {
				t.Errorf("Test %d: Expected %d element zeroed", ti+1, i)
			}
		}
	}
}

func TestDussc(t *testing.T) {
	tests := []struct {
		x        []float64
		indx     []int
		y        []float64
		incy     int
		expected []float64
	}{
		{
			x:        []float64{1, 3, 4},
			indx:     []int{0, 2, 3},
			y:        []float64{0, 0, 0, 0},
			incy:     1,
			expected: []float64{1, 0, 3, 4},
		},
		{
			x:    []float64{1, 3, 4},
			indx: []int{0, 2, 3},
			y: []float64{
				5, 5, 5, 5,
				5, 5, 5, 5,
				5, 5, 5, 5,
				5, 5, 5, 5,
			},
			incy: 4,
			expected: []float64{
				1, 5, 5, 5,
				5, 5, 5, 5,
				3, 5, 5, 5,
				4, 5, 5, 5,
			},
		},
	}

	for ti, test := range tests {
		Dussc(test.x, test.y, test.incy, test.indx)

		for i, y := range test.y {
			if y != test.expected[i] {
				t.Errorf("Test %d: Wanted %f at %d but received %f", ti+1, test.expected[i], i, y)
			}
		}
	}
}
