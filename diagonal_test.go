package sparse

import (
	"testing"

	"github.com/gonum/gonum/mat"
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
