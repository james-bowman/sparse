package sparse

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TestCSRMul(t *testing.T) {
	var tests = []struct {
		atype  MatrixCreator
		am, an int
		adata  []float64
		btype  MatrixCreator
		bm, bn int
		bdata  []float64
		cm, cn int
		cdata  []float64
	}{
		{
			atype: CreateDIA,
			am:    4, an: 4,
			adata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			btype: CreateCSR,
			bm:    4, bn: 3,
			bdata: []float64{
				1, 0, 2,
				3, 4, 5,
				0, 0, 0,
				6, 0, 7,
			},
			cm: 4, cn: 3,
			cdata: []float64{
				1, 0, 2,
				6, 8, 10,
				0, 0, 0,
				24, 0, 28,
			},
		},
		{
			atype: CreateCSR,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateDIA,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 3, cn: 4,
			cdata: []float64{
				1, 6, 0, 24,
				0, 8, 0, 0,
				2, 10, 0, 28,
			},
		},
		{
			atype: CreateDIA,
			am:    4, an: 4,
			adata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			btype: CreateDIA,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 4, cn: 4,
			cdata: []float64{
				1, 0, 0, 0,
				0, 4, 0, 0,
				0, 0, 9, 0,
				0, 0, 0, 16,
			},
		},
		{
			atype: CreateDIA,
			am:    4, an: 4,
			adata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			btype: CreateDense,
			bm:    4, bn: 3,
			bdata: []float64{
				1, 0, 2,
				3, 4, 5,
				0, 0, 0,
				6, 0, 7,
			},
			cm: 4, cn: 3,
			cdata: []float64{
				1, 0, 2,
				6, 8, 10,
				0, 0, 0,
				24, 0, 28,
			},
		},
		{
			atype: CreateDense,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateDIA,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 3, cn: 4,
			cdata: []float64{
				1, 6, 0, 24,
				0, 8, 0, 0,
				2, 10, 0, 28,
			},
		},
		{
			atype: CreateCSR,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateDense,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 3, cn: 4,
			cdata: []float64{
				1, 6, 0, 24,
				0, 8, 0, 0,
				2, 10, 0, 28,
			},
		},
		{
			atype: CreateDense,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateCSR,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 3, cn: 4,
			cdata: []float64{
				1, 6, 0, 24,
				0, 8, 0, 0,
				2, 10, 0, 28,
			},
		},
		{
			atype: CreateCSR,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateCSR,
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4,
			},
			cm: 3, cn: 4,
			cdata: []float64{
				1, 6, 0, 24,
				0, 8, 0, 0,
				2, 10, 0, 28,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat64.NewDense(test.cm, test.cn, test.cdata)

		a := test.atype(test.am, test.an, test.adata)
		b := test.btype(test.bm, test.bn, test.bdata)

		csr := NewCSR(0, 0, nil, nil, nil)
		csr.Mul(a, b)

		if !mat64.Equal(expected, csr) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csr))
			t.Fail()
		}
	}
}

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
