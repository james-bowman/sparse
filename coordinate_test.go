package sparse

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestCOOConversion(t *testing.T) {
	r, c := 3, 4
	data := []float64{
		1, 0, 0, 7,
		0, 2, 4, 0,
		3, 0, 3, 6,
	}

	var tests = []struct {
		desc    string
		create  MatrixCreator
		convert func(a TypeConverter) Sparser
	}{
		{
			"COO -> DOK",
			CreateCOO,
			func(a TypeConverter) Sparser { return a.ToDOK() },
		},
		{
			"COO -> CSR",
			CreateCOO,
			func(a TypeConverter) Sparser { return a.ToCSR() },
		},
		{
			"COO -> CSC",
			CreateCOO,
			func(a TypeConverter) Sparser { return a.ToCSC() },
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d. %s\n", ti+1, test.desc)

		d := mat64.NewDense(r, c, data)

		a := test.create(r, c, data)

		if !mat64.Equal(d, a) {
			t.Logf("A : %v\n", a)
			t.Logf("Expected:\n%v\n but created:\n%v\n", mat64.Formatted(d), mat64.Formatted(a))
			t.Fail()
		}

		sa, ok := a.(Sparser)
		if !ok {
			t.Fatalf("Created matrix type does not implement Sparser")
		}

		b := test.convert(sa.(TypeConverter))

		if !mat64.Equal(a, b) {
			t.Logf("A : %v\n", a)
			t.Logf("B : %v\n", b)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat64.Formatted(a), mat64.Formatted(b))
			t.Fail()
		}
	}
}

func TestCOORowColView(t *testing.T) {
	var tests = []struct {
		r, c int
		data []float64
	}{
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
		},
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 3, 0,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		dense := mat64.NewDense(test.r, test.c, test.data)
		coo := CreateCOO(test.r, test.c, test.data).(*COO)

		for i := 0; i < test.r; i++ {
			row := coo.RowView(i)
			for k := 0; k < row.Len(); k++ {
				if row.At(k, 0) != test.data[i*test.c+k] {
					t.Logf("ROW: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat64.Formatted(row), k, row.At(k, 0), i, k, mat64.Formatted(dense))
					t.Fail()
				}
			}
		}

		for j := 0; j < test.c; j++ {
			col := coo.ColView(j)
			for k := 0; k < col.Len(); k++ {
				if col.At(k, 0) != test.data[k*test.c+j] {
					t.Logf("COL: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat64.Formatted(col), k, col.At(k, 0), k, j, mat64.Formatted(dense))
					t.Fail()
				}
			}
		}
	}
}
