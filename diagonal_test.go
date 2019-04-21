package sparse

import (
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestDIARowColView(t *testing.T) {
	var tests = []struct {
		r, c int
		data []float64
	}{
		{
			r: 4, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
		},
		{
			r: 4, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 5,
			},
		},
		{
			r: 4, c: 5,
			data: []float64{
				1, 0, 0, 0, 0,
				0, 2, 0, 0, 0,
				0, 0, 3, 0, 0,
				0, 0, 0, 4, 0,
			},
		},
		{
			r: 5, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 5,
				0, 0, 0, 0,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		dense := mat.NewDense(test.r, test.c, test.data)
		dia := CreateDIA(test.r, test.c, test.data).(*DIA)

		for i := 0; i < test.r; i++ {
			row := dia.RowView(i)
			for k := 0; k < row.Len(); k++ {
				if row.At(k, 0) != test.data[i*test.c+k] {
					t.Logf("ROW: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat.Formatted(row), k, row.At(k, 0), i, k, mat.Formatted(dense))
					t.Fail()
				}
			}
		}

		for j := 0; j < test.c; j++ {
			col := dia.ColView(j)
			for k := 0; k < col.Len(); k++ {
				if col.At(k, 0) != test.data[k*test.c+j] {
					t.Logf("COL: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat.Formatted(col), k, col.At(k, 0), k, j, mat.Formatted(dense))
					t.Fail()
				}
			}
		}
	}
}

func TestDIADoNonZero(t *testing.T) {
	var tests = []struct {
		r, c int
		data []float64
	}{
		{
			r: 3, c: 3,
			data: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
			},
		},
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
			},
		},
		{
			r: 4, c: 3,
			data: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		matrix := CreateDIA(test.r, test.c, test.data).(*DIA)

		var nnz int
		matrix.DoNonZero(func(i, j int, v float64) {
			if testv := test.data[i*test.c+j]; testv == 0 || testv != v {
				t.Logf("Expected %f at (%d, %d) but received %f\n", v, i, j, testv)
				t.Fail()
			}
			nnz++
		})

		if nnz != matrix.NNZ() {
			t.Logf("Expected %d Non Zero elements but found %d", nnz, matrix.NNZ())
			t.Fail()
		}
	}
}

func TestDIATrace(t *testing.T) {
	var tests = []struct {
		s int
	}{
		{s: 8},
		{s: 32},
		{s: 100},
		{s: 123},
	}
	for _, test := range tests {
		dia := RandomDIA(test.s, test.s)
		tr := mat.Trace(dia)
		var checkTr float64
		for i := 0; i < test.s; i++ {
			checkTr += dia.At(i, i)
		}
		if !floats.EqualWithinAbs(tr, checkTr, 1e-13) {
			t.Logf("trace mismatch: %f vs %f", tr, checkTr)
			t.Fail()
		}

	}
}
