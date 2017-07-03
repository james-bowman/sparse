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
		{
			atype: CreateCSR,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateCSC,
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
			btype: CreateCSC,
			bm:    4, bn: 3,
			bdata: []float64{
				1, 0, 6,
				0, 2, 0,
				7, 0, 3,
				0, 8, 0,
			},
			cm: 3, cn: 3,
			cdata: []float64{
				1, 54, 6,
				0, 8, 0,
				2, 66, 12,
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

func TestCSRAdd(t *testing.T) {
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
			bm:    4, bn: 4,
			bdata: []float64{
				1, 0, 2, 2,
				3, 4, 5, 0,
				0, 0, 0, 1,
				6, 0, 7, 0,
			},
			cm: 4, cn: 4,
			cdata: []float64{
				2, 0, 2, 2,
				3, 6, 5, 0,
				0, 0, 3, 1,
				6, 0, 7, 4,
			},
		},
		{
			atype: CreateCSR,
			am:    4, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
				1, 0, 0, 1,
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
				2, 3, 0, 6,
				0, 6, 0, 0,
				2, 5, 3, 7,
				1, 0, 0, 5,
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
				2, 0, 0, 0,
				0, 4, 0, 0,
				0, 0, 6, 0,
				0, 0, 0, 8,
			},
		},
		{
			atype: CreateCSR,
			am:    4, an: 3,
			adata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
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
				2, 0, 2,
				3, 6, 5,
				0, 0, 3,
				6, 0, 7,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat64.NewDense(test.cm, test.cn, test.cdata)

		a := test.atype(test.am, test.an, test.adata)
		b := test.btype(test.bm, test.bn, test.bdata)

		csr := NewCSR(0, 0, nil, nil, nil)
		csr.Add(a, b)

		if !mat64.Equal(expected, csr) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csr))
			t.Fail()
		}
	}
}
