package sparse

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
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
			atype: CreateCSR,
			am:    3, an: 4,
			adata: []float64{
				1, 3, 0, 6,
				0, 4, 0, 0,
				2, 5, 0, 7,
			},
			btype: CreateDIA,
			bm:    4, bn: 3,
			bdata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
			},
			cm: 3, cn: 3,
			cdata: []float64{
				1, 6, 0,
				0, 8, 0,
				2, 10, 0,
			},
		},
		{
			atype: CreateDIA,
			am:    3, an: 4,
			adata: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
			},
			btype: CreateCSR,
			bm:    4, bn: 3,
			bdata: []float64{
				1, 3, 0,
				0, 4, 0,
				2, 5, 3,
				5, 0, 1,
			},
			cm: 3, cn: 3,
			cdata: []float64{
				1, 3, 0,
				0, 8, 0,
				6, 15, 9,
			},
		},
		{
			atype: CreateCSC,
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
			btype: CreateCSC,
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
		{ // 11
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
		{ // 11
			atype: CreateDense,
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
			bm:    4, bn: 5,
			bdata: []float64{
				1, 0, 6, 7, 3,
				0, 2, 0, 2, 0,
				7, 0, 3, 0, 1,
				0, 8, 0, 0, 1,
			},
			cm: 3, cn: 5,
			cdata: []float64{
				1, 54, 6, 13, 9,
				0, 8, 0, 8, 0,
				2, 66, 12, 24, 13,
			},
		},
	}

	for ti, test := range tests {
		expected := mat.NewDense(test.cm, test.cn, test.cdata)

		a := test.atype(test.am, test.an, test.adata)
		b := test.btype(test.bm, test.bn, test.bdata)

		var csr CSR
		csr.Mul(a, b)

		if !mat.Equal(expected, &csr) {
			t.Logf("Test %d:\n%v\n", ti+1, csr)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(&csr))
			t.Fail()
		}
	}
}

func TestCSRMatVec(t *testing.T) {
	var tests = []struct {
		am, an        int
		aind, aindptr []int
		adata         []float64
		rhs           []float64
		out           []float64
	}{
		{
			am: 5, an: 5,
			aind: []int{
				2, 4, 1, 1, 3, 2, 3,
			},
			aindptr: []int{
				0, 2, 3, 4, 5, 7,
			},
			adata: []float64{
				0.142866817922, 0.0564115790271, 0.099974915818, 0.650888472949, 0.721998772267, 0.333708611139, 0.459248891966,
			},
			rhs: []float64{
				1.0, 2.0, 3.0, 4.0, 5.0,
			},
			out: []float64{
				0.710658348901, 0.199949831636, 1.3017769459, 2.88799508907, 2.83812140128,
			},
		},

		{
			am: 5, an: 4,
			aind: []int{
				0, 1, 0, 2, 3, 1,
			},
			aindptr: []int{
				0, 2, 5, 5, 5, 6,
			},
			adata: []float64{
				0.23277134043, 0.0466656632136, 0.973755518841, 0.0906064345328, 0.618386009333, 0.382461991267,
			},
			rhs: []float64{
				1.0, 2.0, 3.0, 4.0,
			},
			out: []float64{
				0.326102666858, 3.71911885977, 0.0, 0.0, 0.764923982534,
			},
		},

		{
			am: 5, an: 3,
			aind: []int{
				0, 0, 1, 0,
			},
			aindptr: []int{
				0, 0, 0, 1, 3, 4,
			},
			adata: []float64{
				0.0650515929853, 0.607544851901, 0.170524123687, 0.948885537253,
			},
			rhs: []float64{
				1.0, 2.0, 3.0,
			},
			out: []float64{
				0.0, 0.0, 0.0650515929853, 0.948593099276, 0.948885537253,
			},
		},

		{
			am: 5, an: 2,
			aind: []int{
				0, 1, 1,
			},
			aindptr: []int{
				0, 0, 1, 1, 2, 3,
			},
			adata: []float64{
				0.0159662522202, 0.241025466026, 0.230893825622,
			},
			rhs: []float64{
				1.0, 2.0,
			},
			out: []float64{
				0.0, 0.0159662522202, 0.0, 0.482050932052, 0.461787651244,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)
		lhs := NewCSR(test.am, test.an, test.aindptr, test.aind, test.adata)
		have := make([]float64, test.am)
		MulMatRawVec(lhs, test.rhs, have)

		for row := 0; row < test.am; row++ {
			// NOTE: can only use precision 1e-11 b/c of printing output in numpy
			if math.Abs(have[row]-test.out[row]) > 1e-11 {
				t.Logf("Expected:\n%v\n but received:\n%v\n", test.out[row], have[row])
				t.Fail()
			}
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
		{
			atype: CreateCSR,
			am:    4, an: 3,
			adata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
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
				2, 0, 2,
				3, 6, 5,
				0, 0, 3,
				6, 0, 7,
			},
		},
		{
			atype: CreateDense,
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
		{
			atype: CreateDense,
			am:    4, an: 3,
			adata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
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
				2, 0, 2,
				3, 6, 5,
				0, 0, 3,
				6, 0, 7,
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
			btype: CreateCSC,
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
		{
			atype: CreateCSC,
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

		expected := mat.NewDense(test.cm, test.cn, test.cdata)

		a := test.atype(test.am, test.an, test.adata)
		b := test.btype(test.bm, test.bn, test.bdata)

		var csr CSR
		csr.Add(a, b)

		if !mat.Equal(expected, &csr) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(&csr))
			t.Fail()
		}
	}
}

func TestCSRSub(t *testing.T) {
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
				0, 0, 1, 1,
				6, 0, 7, 0,
			},
			cm: 4, cn: 4,
			cdata: []float64{
				0, 0, -2, -2,
				-3, -2, -5, 0,
				0, 0, 2, -1,
				-6, 0, -7, 4,
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
				0, 3, 0, 6,
				0, 2, 0, 0,
				2, 5, -3, 7,
				1, 0, 0, -3,
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
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 0, 0,
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
				0, 0, -2,
				-3, -2, -5,
				0, 0, 3,
				-6, 0, -7,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat.NewDense(test.cm, test.cn, test.cdata)

		a := test.atype(test.am, test.an, test.adata)
		b := test.btype(test.bm, test.bn, test.bdata)

		var csr CSR
		csr.Sub(a, b)

		if !mat.Equal(expected, &csr) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(&csr))
			t.Fail()
		}
	}
}
