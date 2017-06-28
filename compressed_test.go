package sparse

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestCSRCSCTranspose(t *testing.T) {
	var tests = []struct {
		r, c   int
		data   []float64
		er, ec int
		result []float64
	}{
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
			er: 4, ec: 3,
			result: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 6,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat64.NewDense(test.er, test.ec, test.result)

		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		t.Logf("CSR: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csr.i, csr.j, csr.ind, csr.indptr, csr.data)
		t.Logf("CSC: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csc.i, csc.j, csc.ind, csc.indptr, csc.data)

		if !mat64.Equal(expected, csr.T()) {
			t.Logf("CSR is:\n%v\n", mat64.Formatted(csr))
			t.Logf("For CSR^T, Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csr.T()))
			t.Fail()
		}
		if !mat64.Equal(expected, csc.T()) {
			t.Logf("CSC is:\n%v\n", mat64.Formatted(csc))
			t.Logf("For CSC^T, Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csc.T()))
			t.Fail()
		}
	}
}

func TestCSRCSCConversion(t *testing.T) {
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
			"CSR -> CSC",
			CreateCSR,
			func(a TypeConverter) Sparser { return a.ToCSC() },
		},
		{
			"CSC -> CSR",
			CreateCSC,
			func(a TypeConverter) Sparser { return a.ToCSR() },
		},
		{
			"CSR -> COO",
			CreateCSR,
			func(a TypeConverter) Sparser { return a.ToCOO() },
		},
		{
			"CSC -> COO",
			CreateCSC,
			func(a TypeConverter) Sparser { return a.ToCOO() },
		},
		{
			"CSR -> DOK",
			CreateCSR,
			func(a TypeConverter) Sparser { return a.ToDOK() },
		},
		{
			"CSC -> DOK",
			CreateCSC,
			func(a TypeConverter) Sparser { return a.ToDOK() },
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d. %s\n", ti+1, test.desc)

		d := mat64.NewDense(r, c, data)

		a := test.create(r, c, data)
		sa, ok := a.(Sparser)
		if !ok {
			t.Fatalf("Created matrix type does not implement Sparser")
		}
		b := test.convert(sa.(TypeConverter))

		if !mat64.Equal(d, b) {
			t.Logf("d : %v\n", a)
			t.Logf("B : %v\n", b)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat64.Formatted(d), mat64.Formatted(b))
			t.Fail()
		}
	}
}

func TestCSRCSCSet(t *testing.T) {
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
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat64.NewDense(test.r, test.c, test.result)

		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		t.Logf("CSR: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csr.i, csr.j, csr.ind, csr.indptr, csr.data)
		t.Logf("CSC: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csc.i, csc.j, csc.ind, csc.indptr, csc.data)

		csr.Set(test.i, test.j, test.v)
		csc.Set(test.i, test.j, test.v)

		t.Logf("CSR: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csr.i, csr.j, csr.ind, csr.indptr, csr.data)
		t.Logf("CSC: r: %d, c: %d, ind: %v, indptr: %v, data: %v", csc.i, csc.j, csc.ind, csc.indptr, csc.data)

		if !mat64.Equal(expected, csr) {
			t.Logf("For CSR.Set(), Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csr))
			t.Fail()
		}
		if !mat64.Equal(expected, csc) {
			t.Logf("For CSC.Set(), Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csc))
			t.Fail()
		}
	}
}
