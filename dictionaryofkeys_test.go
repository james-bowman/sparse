package sparse

import (
	"math/rand"
	"testing"

	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

func TestDOKConversion(t *testing.T) {
	var tests = []struct {
		m, n   int
		data   map[key]float64
		output []float64
	}{
		{
			m: 11, n: 11,
			data: map[key]float64{
				key{0, 3}:   1,
				key{1, 1}:   2,
				key{2, 2}:   3,
				key{5, 8}:   4,
				key{10, 10}: 5,
				key{1, 5}:   6,
				key{3, 5}:   7,
			},
			output: []float64{
				0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
				0, 2, 0, 0, 0, 6, 0, 0, 0, 0, 0,
				0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5,
			},
		},
		{
			m: 5, n: 4,
			data: map[key]float64{
				key{0, 3}: 1,
				key{1, 1}: 2,
				key{2, 2}: 3,
				key{4, 2}: 4,
				key{0, 0}: 5,
				key{1, 3}: 6,
				key{3, 3}: 7,
			},
			output: []float64{
				5, 0, 0, 1,
				0, 2, 0, 6,
				0, 0, 3, 0,
				0, 0, 0, 7,
				0, 0, 4, 0,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)
		expected := mat64.NewDense(test.m, test.n, test.output)

		dok := NewDOK(test.m, test.n)
		for k, v := range test.data {
			dok.Set(k.i, k.j, v)
		}

		coo := dok.ToCOO()
		if !(mat64.Equal(expected, coo)) {
			t.Logf("Expected:\n%v \nbut found COO matrix:\n%v\n", mat64.Formatted(expected), mat64.Formatted(coo))
		}

		csr := dok.ToCSR()
		if !(mat64.Equal(expected, csr)) {
			t.Logf("Expected:\n%v \nbut found CSR matrix:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csr))
		}

		csc := dok.ToCSC()
		if !(mat64.Equal(expected, csc)) {
			t.Logf("Expected:\n%v \nbut found CSC matrix:\n%v\n", mat64.Formatted(expected), mat64.Formatted(csc))
		}
	}

}

func TestDOKTranspose(t *testing.T) {
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

		dok := CreateDOK(test.r, test.c, test.data)

		if !mat64.Equal(expected, dok.T()) {
			t.Logf("Expected:\n %v\n but received:\n %v\n", mat64.Formatted(expected), mat64.Formatted(dok.T()))
		}
	}
}

func TestOldCSRMul(t *testing.T) {
	var tests = []struct {
		target MatrixCreator
		atype  MatrixCreator
		am, an int
		a      []float64
		btype  MatrixCreator
		bm, bn int
		b      []float64
	}{
		{
			target: CreateCSR,
			atype:  CreateCSR,
			am:     5, an: 4,
			a: []float64{
				7, 0, 0, 1,
				0, 2, 0, 1,
				6, 0, 3, 0,
				0, 5, 0, 0,
				0, 0, 0, 2,
			},
			btype: CreateDOK,
			bm:    4, bn: 5,
			b: []float64{
				7, 0, 0, 1, 5,
				0, 2, 0, 1, 5,
				6, 0, 3, 0, 0,
				0, 5, 0, 0, 7,
			},
		},
	}

	for ti, test := range tests {
		t.Logf("**** Test Run %d.\n", ti+1)
		expected := mat64.NewDense(test.am, test.bn, nil)
		expected.Mul(mat64.NewDense(test.am, test.an, test.a), mat64.NewDense(test.bm, test.bn, test.b))

		target := test.target(0, 0, nil)

		target.(*CSR).Mul(test.atype(test.am, test.an, test.a), test.btype(test.bm, test.bn, test.b))

		if !mat64.Equal(expected, target) {
			t.Logf("Expected:\n%v\nbut received:\n%v\n", mat64.Formatted(expected), mat64.Formatted(target))
			t.Fail()
		}
	}
}

type MatrixCreator func(m, n int, data []float64) mat64.Matrix

func CreateDOK(m, n int, data []float64) mat64.Matrix {
	dok := NewDOK(m, n)
	if data != nil {
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				if data[i*n+j] != 0 {
					dok.Set(i, j, data[i*n+j])
				}
			}
		}
	}

	return dok
}

func CreateCOO(m, n int, data []float64) mat64.Matrix {
	return CreateDOK(m, n, data).(*DOK).ToCOO()
}

func CreateCSR(m, n int, data []float64) mat64.Matrix {
	return CreateDOK(m, n, data).(*DOK).ToCSR()
}

func CreateCSC(m, n int, data []float64) mat64.Matrix {
	return CreateDOK(m, n, data).(*DOK).ToCSC()
}

func CreateDIA(m, n int, data []float64) mat64.Matrix {
	if m != n {
		panic((matrix.ErrRowAccess))
	}
	c := make([]float64, m)
	for i := 0; i < m; i++ {
		c[i] = data[i*n+i]
	}
	return NewDIA(m, c)
}

func CreateDense(m, n int, data []float64) mat64.Matrix {
	return mat64.NewDense(m, n, data)
}

func createMatrix(creator MatrixCreator, r int, c int, density float32) mat64.Matrix {
	data := make([]float64, r*c)

	for i := 0; i < len(data); i++ {
		prob := rand.Float32()
		if prob < density {
			data[i] = rand.Float64()
		}
	}

	return creator(r, c, data)
}

func createMatrices(tc MatrixCreator, ac MatrixCreator, bc MatrixCreator) (target, a, b mat64.Matrix) {
	data := []float64{
		7, 0, 0, 1, 6,
		0, 2, 0, 1, 4,
		6, 0, 3, 0, 3,
		0, 5, 0, 0, 8,
		0, 0, 0, 2, 4,
	}

	target = tc(0, 0, nil)
	a = ac(5, 5, data)
	b = bc(5, 5, data)

	return
}

type MatMultiplyer interface {
	Mul(a, b mat64.Matrix)
}

func benchmarkMatrixMultiplication(target MatMultiplyer, lhs mat64.Matrix, rhs mat64.Matrix, b *testing.B) {
	for n := 0; n < b.N; n++ {
		target.Mul(lhs, rhs)
	}
}

func BenchmarkMulSmallDenseDenseDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateDense, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDOKDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateDOK, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDOKDOK(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateDOK, 5, 6, 0.4)
	rhs := createMatrix(CreateDOK, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseDOK(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateDense, 5, 6, 0.4)
	rhs := createMatrix(CreateDOK, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCSRDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCSR, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseCSR(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCSR, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCSRCSR(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCSR, 5, 6, 0.4)
	rhs := createMatrix(CreateCSR, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCOODense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCOO, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseCOO(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCOO, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCOOCOO(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := createMatrix(CreateCOO, 5, 6, 0.4)
	rhs := createMatrix(CreateCOO, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulSmallCSRDenseDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateDense, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDOKDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateDOK, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDenseDOK(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateDOK, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDOKDOK(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateDOK, 5, 6, 0.4)
	rhs := createMatrix(CreateDOK, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRCSRDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateCSR, 5, 6, 0.4)
	rhs := createMatrix(CreateDense, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDenseCSR(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateDense, 5, 6, 0.4)
	rhs := createMatrix(CreateCSR, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRCSRCSR(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := createMatrix(CreateCSR, 5, 6, 0.4)
	rhs := createMatrix(CreateCSR, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeDenseDenseDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDOKDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDOKDOK(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDenseDOK(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCSRDense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDenseCSR(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCSRCSR(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCOODense(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(COOFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

/*
func BenchmarkMulLargeDenseDenseCOO(b *testing.B) {
	t := mat64.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(COOFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
*/
//func BenchmarkMulLargeDenseCOOCOO(b *testing.B) {
//	t := mat64.NewDense(0, 0, nil)
//	lhs := createMatrix(CreateCOO, 500, 600, 0.4)
//	rhs := createMatrix(CreateCOO, 600, 500, 0.4)
//	benchmarkMatrixMultiplication(t, lhs, rhs, b)
//}

func BenchmarkMulLargeCSRDenseDense(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRDOKDense(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRDenseDOK(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRDOKDOK(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRCSRDense(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRDenseCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeCSRCSRCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMul2CSRDenseDense(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateDense, CreateDense)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2CSRDOKDense(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateDOK, CreateDense)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2CSRDOKDOK(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateDOK, CreateDOK)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2CSRCSRDense(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateCSR, CreateDense)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2CSRDenseCSR(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateDense, CreateCSR)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2CSRCSRCSR(b *testing.B) {
	t, l, r := createMatrices(CreateCSR, CreateCSR, CreateCSR)
	for n := 0; n < b.N; n++ {
		t.(*CSR).Mul2(l, r)
	}
}
func BenchmarkMul2LargeCSRCSRCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	l := Random(CSRFormat, 500, 600, 0.01)
	r := Random(CSRFormat, 600, 500, 0.01)
	for n := 0; n < b.N; n++ {
		t.Mul2(l, r)
	}
}
