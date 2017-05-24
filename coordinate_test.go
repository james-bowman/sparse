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
