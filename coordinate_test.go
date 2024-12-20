package sparse

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func CreateCOOWithDupes(m, n int, data []float64) mat.Matrix {
	coo := CreateCOO(m, n, data).(*COO)
	for k := 0; k < rand.Intn(m*n-1)+1; k++ {
		i := rand.Intn(m)
		j := rand.Intn(n)
		coo.Set(i, j, 5)
		coo.Set(i, j, -5)
	}
	return coo
}

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
			"COO -> CSR (With Dupes)",
			CreateCOOWithDupes,
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

		d := mat.NewDense(r, c, data)

		a := test.create(r, c, data)

		if !mat.Equal(d, a) {
			t.Logf("A : %v\n", a)
			t.Logf("Expected:\n%v\n but created:\n%v\n", mat.Formatted(d), mat.Formatted(a))
			t.Fail()
		}

		sa, ok := a.(Sparser)
		if !ok {
			t.Fatalf("Created matrix type does not implement Sparser")
		}

		b := test.convert(sa.(TypeConverter))

		if !mat.Equal(a, b) {
			t.Logf("A : %v\n", a)
			t.Logf("B : %v\n", b)
			t.Logf("Expected:\n%v\n but received:\n%v\n", mat.Formatted(a), mat.Formatted(b))
			t.Fail()
		}

		if !mat.Equal(d, a) {
			t.Logf("D : %v\n", d)
			t.Logf("A : %v\n", a)
			t.Logf("Original matrix changed - Expected:\n%v\n but received:\n%v\n", mat.Formatted(d), mat.Formatted(a))
			t.Fail()
		}
	}
}

func TestCOODoNonZero(t *testing.T) {
	var tests = []struct {
		r, c int
		data []float64
	}{
		{
			r: 3, c: 3,
			data: []float64{
				1, 0, 3,
				0, 2, 0,
				1, 0, 3,
			},
		},
		{
			r: 3, c: 4,
			data: []float64{
				1, 0, 5, 8,
				0, 0, 0, 0,
				6, 0, 3, 8,
			},
		},
		{
			r: 4, c: 3,
			data: []float64{
				1, 0, 8,
				0, 0, 0,
				3, 0, 3,
				2, 0, 0,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)

		matrix := CreateCOO(test.r, test.c, test.data).(*COO)

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

func TestCOOTranspose(t *testing.T) {
	tests := []struct {
		m     *COO
		r     int
		c     int
		data  []float64
		er    int
		ec    int
		edata []float64
	}{
		{
			m: NewCOO(
				3, 4,
				[]int{0, 1, 2, 2},
				[]int{0, 1, 2, 3},
				[]float64{1, 2, 3, 6},
			),
			er: 4, ec: 3,
			edata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 6},
		},
		{
			m: NewCOO(
				3, 4,
				[]int{0, 2, 1, 2},
				[]int{0, 2, 1, 3},
				[]float64{1, 3, 2, 6},
			),
			er: 4, ec: 3,
			edata: []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 6},
		},
	}

	for ti, test := range tests {
		orig := mat.DenseCopyOf(test.m)

		e := mat.NewDense(test.er, test.ec, test.edata)

		tr := test.m.T()

		if !mat.Equal(e, tr) {
			t.Errorf("Test %d: Expected\n%v\nBut received\n%v\n", ti, mat.Formatted(e), mat.Formatted(tr))
		}

		nt := tr.T()

		if !mat.Equal(orig, nt) {
			t.Errorf("Test %d: Transpose back Expected\n%v\nBut received\n%v\n", ti, mat.Formatted(orig), mat.Formatted(nt))
		}
	}
}

func TestPermutations(t *testing.T) {
	for _, test := range []struct {
		m           *COO
		permutation []int
		want        *mat.Dense
		desc        string
		permuteCol  bool
	}{
		{
			m:           NewCOO(2, 3, []int{0, 0, 1}, []int{0, 1, 0}, []float64{1.0, 2.0, 3.0}),
			permutation: []int{1, 0},
			want:        mat.NewDense(2, 3, []float64{3.0, 0.0, 0.0, 1.0, 2.0, 0.0}),
			desc:        "2x3 matrix row permutation",
			permuteCol:  false,
		},
		{
			m:           NewCOO(2, 3, []int{0, 0, 1}, []int{0, 1, 0}, []float64{1.0, 2.0, 3.0}),
			permutation: []int{1, 0, 2},
			want:        mat.NewDense(2, 3, []float64{2.0, 1.0, 0.0, 0.0, 3.0, 0.0}),
			desc:        "2x3 matrix col permutation",
			permuteCol:  true,
		},
		{
			m:           NewCOO(1, 1, []int{0}, []int{0}, []float64{1.0}),
			permutation: []int{0},
			want:        mat.NewDense(1, 1, []float64{1.0}),
			desc:        "1x1 matrix row permutation",
			permuteCol:  false,
		},
	} {
		if test.permuteCol {
			test.m.PermuteCols(test.permutation)
		} else {
			test.m.PermuteRows(test.permutation)
		}

		result := test.m.ToDense()

		if !mat.Equal(result, test.want) {
			t.Errorf("Test: %s: Expected\n%v\ngot%v\n", test.desc, test.want, result)
		}

	}
}

func TestIsPermutation(t *testing.T) {
	for _, test := range []struct {
		N           int
		permutation []int
		valid       bool
		desc        string
	}{
		{
			N:           2,
			permutation: []int{0, 1},
			valid:       true,
			desc:        "Valid 2xN matrix permutation",
		},
		{
			N:           3,
			permutation: []int{0, 1},
			valid:       false,
			desc:        "Invalid: missing 2",
		},
		{
			N:           3,
			permutation: []int{0, 0, 1},
			valid:       false,
			desc:        "Invalid: duplicates",
		},
		{
			N:           3,
			permutation: []int{1, 0, 4},
			valid:       false,
			desc:        "Invalid: value out of bounds",
		},
	} {
		if result := isPermutation(test.permutation, test.N); result != test.valid {
			t.Errorf("Test %s: expected %v got %v\n", test.desc, test.valid, result)
		}
	}
}
