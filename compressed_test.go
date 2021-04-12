package sparse

import (
	"testing"

	"gonum.org/v1/gonum/floats/scalar"
	"gonum.org/v1/gonum/mat"
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

		expected := mat.NewDense(test.er, test.ec, test.result)

		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		if !mat.Equal(expected, csr.T()) {
			t.Logf("CSR is:\n%v\n", mat.Formatted(csr))
			t.Logf("For CSR^T, Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csr.T()))
			t.Fail()
		}
		if !mat.Equal(expected, csc.T()) {
			t.Logf("CSC is:\n%v\n", mat.Formatted(csc))
			t.Logf("For CSC^T, Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csc.T()))
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
		r, c    int
		data    []float64
		create  MatrixCreator
		convert func(a TypeConverter) mat.Matrix
	}{
		{
			"CSR -> CSC",
			r, c,
			data,
			CreateCSR,
			func(a TypeConverter) mat.Matrix { return a.ToCSC() },
		},
		{
			"CSC -> CSR",
			r, c,
			data,
			CreateCSC,
			func(a TypeConverter) mat.Matrix { return a.ToCSR() },
		},
		{
			"CSR -> COO",
			r, c,
			data,
			CreateCSR,
			func(a TypeConverter) mat.Matrix { return a.ToCOO() },
		},
		{
			"CSC -> COO",
			r, c,
			data,
			CreateCSC,
			func(a TypeConverter) mat.Matrix { return a.ToCOO() },
		},
		{
			"CSR -> DOK",
			r, c,
			data,
			CreateCSR,
			func(a TypeConverter) mat.Matrix { return a.ToDOK() },
		},
		{
			"CSC -> DOK",
			r, c,
			data,
			CreateCSC,
			func(a TypeConverter) mat.Matrix { return a.ToDOK() },
		},
		{
			"CSR -> Dense",
			r, c,
			data,
			CreateCSR,
			func(a TypeConverter) mat.Matrix { return a.ToDense() },
		},
		{
			"CSC -> Dense",
			r, c,
			data,
			CreateCSC,
			func(a TypeConverter) mat.Matrix { return a.ToDense() },
		},
		{
			"CSR -> CSC 2",
			5, 4,
			[]float64{
				1, 0, 0, 7,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 7, 0,
				0, 0, 0, 0,
			},
			CreateCSR,
			func(a TypeConverter) mat.Matrix { return a.ToCSC() },
		},
		{
			"CSC -> CSR 2",
			5, 4,
			[]float64{
				1, 0, 0, 7,
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 7, 0,
				0, 0, 0, 0,
			},
			CreateCSC,
			func(a TypeConverter) mat.Matrix { return a.ToCSR() },
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d. %s\n", ti+1, test.desc)

		d := mat.NewDense(r, c, data)

		a := test.create(r, c, data)
		b := test.convert(a.(TypeConverter))

		if !mat.Equal(d, b) {
			t.Logf("d : %v\n", a)
			t.Logf("B : %v\n", b)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(d), mat.Formatted(b))
			t.Fail()
		}
		// check has not mutated original matrix
		if !mat.Equal(a, b) {
			t.Logf("A : %v\n", a)
			t.Logf("B : %v\n", b)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(a), mat.Formatted(b))
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

		expected := mat.NewDense(test.r, test.c, test.result)

		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		csr.Set(test.i, test.j, test.v)
		csc.Set(test.i, test.j, test.v)

		if !mat.Equal(expected, csr) {
			t.Logf("For CSR.Set(), Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csr))
			t.Fail()
		}
		if !mat.Equal(expected, csc) {
			t.Logf("For CSC.Set(), Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csc))
			t.Fail()
		}
	}
}

func TestCSRCSCRowColView(t *testing.T) {
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

		dense := mat.NewDense(test.r, test.c, test.data)
		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		for i := 0; i < test.r; i++ {
			row := csr.RowView(i)
			for k := 0; k < row.Len(); k++ {
				if row.At(k, 0) != test.data[i*test.c+k] {
					t.Logf("ROW: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat.Formatted(row), k, row.At(k, 0), i, k, mat.Formatted(dense))
					t.Fail()
				}
			}
		}

		for j := 0; j < test.c; j++ {
			col := csc.ColView(j)
			for k := 0; k < col.Len(); k++ {
				if col.At(k, 0) != test.data[k*test.c+j] {
					t.Logf("COL: Vector = \n%v\nElement %d = %f was not element %d, %d from \n%v\n", mat.Formatted(col), k, col.At(k, 0), k, j, mat.Formatted(dense))
					t.Fail()
				}
			}
		}
	}
}

func TestCSRCSCDoNonZero(t *testing.T) {
	var tests = []struct {
		r, c int
		data []float64
	}{
		{
			r: 3, c: 3,
			data: []float64{
				1, 3, 6,
				0, 2, 0,
				1, 0, 3,
			},
		},
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 4,
				0, 0, 0, 0,
				1, 0, 3, 8,
			},
		},
		{
			r: 4, c: 3,
			data: []float64{
				1, 0, 0,
				0, 0, 0,
				0, 0, 3,
				4, 0, 8,
			},
		},
	}
	creatorFuncs := map[string]MatrixCreator{
		"csr": CreateCSR,
		"csc": CreateCSC,
	}

	for creatorKey, creator := range creatorFuncs {
		for ti, test := range tests {
			t.Logf("**** Test Run %d. using %s\n", ti+1, creatorKey)

			matrix := creator(test.r, test.c, test.data).(Sparser)

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
}

func TestCSTrace(t *testing.T) {
	var tests = []struct {
		s       int
		theType MatrixType
		density float32
	}{
		{
			s:       8,
			theType: CSRFormat,
			density: 0.1,
		},
		{
			s:       8,
			theType: CSCFormat,
			density: 0.1,
		},
		{
			s:       80,
			theType: CSRFormat,
			density: 0.75,
		},
		{
			s:       80,
			theType: CSCFormat,
			density: 0.75,
		},
	}
	for _, test := range tests {
		m := Random(test.theType, test.s, test.s, test.density)
		tr := mat.Trace(m)
		var checkTr float64
		for i := 0; i < test.s; i++ {
			checkTr += m.At(i, i)
		}
		if !scalar.EqualWithinAbs(tr, checkTr, 1e-13) {
			t.Logf("trace mismatch: %f vs %f", tr, checkTr)
			t.Fail()
		}

	}
}

func TestCSRCSCCull(t *testing.T) {
	var tests = []struct {
		r, c     int
		data     []float64
		nnz      int
		epsilon  float64
		nnzE     int
		expected []float64
	}{
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
			nnz:     4,
			epsilon: 0.0,
			nnzE:    4,
			expected: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
		},
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 6,
			},
			nnz:     4,
			epsilon: 2.5,
			nnzE:    2,
			expected: []float64{
				0, 0, 0, 0,
				0, 0, 0, 0,
				0, 0, 3, 6,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		expected := mat.NewDense(test.r, test.c, test.expected)
		csr := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc := CreateCSC(test.r, test.c, test.data).(*CSC)

		nnzCSR := csr.NNZ()
		nnzCSC := csc.NNZ()

		if nnzCSR != test.nnz {
			t.Logf("CSR NNZ is %d vs %d", nnzCSR, test.nnz)
			t.Fail()
		}
		if nnzCSC != test.nnz {
			t.Logf("CSC NNZ is %d vs %d", nnzCSC, test.nnz)
			t.Fail()
		}

		csrLen := len(csr.matrix.Data)
		cscLen := len(csc.matrix.Data)
		if csrLen != test.nnz {
			t.Logf("CSR data length incorrect: %d, %d", csrLen, test.nnz)
			t.Fail()
		}
		if cscLen != test.nnz {
			t.Logf("CSC data length incorrect: %d, %d", cscLen, test.nnz)
			t.Fail()
		}

		csr.Cull(test.epsilon)
		csc.Cull(test.epsilon)

		if !mat.Equal(csr, expected) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csr))
			t.Fail()
		}
		if !mat.Equal(csc, expected) {
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(expected), mat.Formatted(csc))
			t.Fail()
		}

		nnzECSR := csr.NNZ()
		nnzECSC := csc.NNZ()
		if nnzECSR != test.nnzE {
			t.Logf("CSR NNZE is %d vs %d", nnzECSR, test.nnzE)
			t.Fail()
		}
		if nnzECSC != test.nnzE {
			t.Logf("CSC NNZE is %d vs %d", nnzECSC, test.nnzE)
			t.Fail()
		}

		csrELen := len(csr.matrix.Data)
		cscELen := len(csc.matrix.Data)
		if csrELen != test.nnzE {
			t.Logf("CSR data length incorrect: %d, %d", csrELen, test.nnzE)
			t.Fail()
		}
		if cscELen != test.nnzE {
			t.Logf("CSC data length incorrect: %d, %d", cscELen, test.nnzE)
			t.Fail()
		}

		csr2 := CreateCSR(test.r, test.c, test.data).(*CSR)
		csc2 := CreateCSC(test.r, test.c, test.data).(*CSC)
		cscT := csc2.T().(*CSR)
		csrT := csr2.T().(*CSC)
		nnzCSRT := csrT.NNZ()
		nnzCSCT := cscT.NNZ()

		csrT.Cull(test.epsilon)
		cscT.Cull(test.epsilon)

		nnzECSRT := csrT.NNZ()
		nnzECSCT := cscT.NNZ()

		if nnzCSRT != test.nnz {
			t.Logf("CSRT NNZ is %d vs %d", nnzCSRT, test.nnz)
			t.Fail()
		}
		if nnzECSRT != test.nnzE {
			t.Logf("CSRT NNZE is %d vs %d", nnzECSRT, test.nnzE)
			t.Fail()
		}
		if nnzCSCT != test.nnz {
			t.Logf("CSCT NNZ is %d vs %d", nnzCSCT, test.nnz)
			t.Fail()
		}
		if nnzECSCT != test.nnzE {
			t.Logf("CSCT NNZE is %d vs %d", nnzECSCT, test.nnzE)
			t.Fail()
		}
	}
}
