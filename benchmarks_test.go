package sparse

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

type MatMultiplyer interface {
	Mul(a, b mat.Matrix)
}

func benchmarkMatrixMultiplication(target MatMultiplyer, lhs mat.Matrix, rhs mat.Matrix, b *testing.B) {
	for n := 0; n < b.N; n++ {
		target.Mul(lhs, rhs)
	}
}

func BenchmarkMulSmallDenseDenseDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDOKDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDOKDOK(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 5, 6, 0.4)
	rhs := Random(DOKFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseDOK(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 5, 6, 0.4)
	rhs := Random(DOKFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCSRDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseCSR(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCSRCSR(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 5, 6, 0.4)
	rhs := Random(CSRFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCOODense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(COOFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseDenseCOO(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(COOFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallDenseCOOCOO(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(COOFormat, 5, 6, 0.4)
	rhs := Random(COOFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulSmallCSRDenseDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(DenseFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDOKDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(DOKFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDenseDOK(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(DOKFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDOKDOK(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(DOKFormat, 5, 6, 0.4)
	rhs := Random(DOKFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRCSRDense(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(CSRFormat, 5, 6, 0.4)
	rhs := Random(DenseFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRDenseCSR(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(DenseFormat, 5, 6, 0.4)
	rhs := Random(CSRFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulSmallCSRCSRCSR(b *testing.B) {
	t := CreateCSR(6, 5, nil).(*CSR)
	lhs := Random(CSRFormat, 5, 6, 0.4)
	rhs := Random(CSRFormat, 6, 5, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeDenseDenseDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDOKDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDOKDOK(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DOKFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDenseDOK(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(DOKFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCSRDense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseDenseCSR(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCSRCSR(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenseCOODense(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(COOFormat, 500, 600, 0.01)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

/*
func BenchmarkMulLargeDenseDenseCOO(b *testing.B) {
	t := mat.NewDense(0, 0, nil)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := Random(COOFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
*/
//func BenchmarkMulLargeDenseCOOCOO(b *testing.B) {
//	t := mat.NewDense(0, 0, nil)
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
func BenchmarkMulLargeCSRCSRCSC(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(CSCFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenserCSRCSRDense(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.4)
	rhs := Random(DenseFormat, 600, 500, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenserCSRCSRCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.4)
	rhs := Random(CSRFormat, 600, 500, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenserCSRCSRDOK(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.4)
	rhs := Random(CSRFormat, 600, 500, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}
func BenchmarkMulLargeDenserCSRCSRCSC(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.4)
	rhs := Random(CSCFormat, 600, 500, 0.4)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkAddLargeCSRCSRCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := Random(CSRFormat, 500, 600, 0.01)
	benchmarkMatrixAddition(t, lhs, rhs, b)
}

func BenchmarkAddLargeDenserCSRCSRCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.4)
	rhs := Random(CSRFormat, 500, 600, 0.4)
	benchmarkMatrixAddition(t, lhs, rhs, b)
}

func BenchmarkAddLargeDenserCSRDenseCSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DenseFormat, 500, 600, 0.4)
	rhs := Random(CSRFormat, 500, 600, 0.4)
	benchmarkMatrixAddition(t, lhs, rhs, b)
}

func benchmarkMatrixAddition(target *CSR, lhs mat.Matrix, rhs mat.Matrix, b *testing.B) {
	for n := 0; n < b.N; n++ {
		target.Add(lhs, rhs)
	}
}

func BenchmarkBLASMulMatMat(b *testing.B) {
	ar, ac := 500, 600
	br, bc := 600, 500

	benchmarks := []struct {
		name    string
		transA  bool
		alpha   float64
		a       MatrixType
		b       MatrixType
		density float32
	}{
		{
			name:    "CSRxDense",
			transA:  false,
			a:       CSRFormat,
			b:       DenseFormat,
			density: 0.01,
		},
		{
			name:    "CSCxDense",
			transA:  false,
			a:       CSCFormat,
			b:       DenseFormat,
			density: 0.01,
		},
		// {
		// 	name:    "CSRxDense",
		// 	transA:  true,
		// 	alpha:   1,
		// 	a:       CSRFormat,
		// 	b:       DenseFormat,
		// 	density: 0.1,
		// },
		// {
		// 	name:    "CSCxDense",
		// 	transA:  true,
		// 	alpha:   1,
		// 	a:       CSCFormat,
		// 	b:       DenseFormat,
		// 	density: 0.1,
		// },
		{
			name:    "COOxDense",
			transA:  false,
			a:       COOFormat,
			b:       DenseFormat,
			density: 0.01,
		},
		{
			name:    "DOKxDense",
			transA:  false,
			a:       DOKFormat,
			b:       DenseFormat,
			density: 0.01,
		},
		{
			name:    "CSRxCSC",
			transA:  false,
			a:       CSRFormat,
			b:       CSCFormat,
			density: 0.01,
		},
		{
			name:    "CSRxCSR",
			transA:  false,
			a:       CSRFormat,
			b:       CSRFormat,
			density: 0.01,
		},
		{
			name:    "CSRxCOO",
			transA:  false,
			a:       CSRFormat,
			b:       COOFormat,
			density: 0.01,
		},
		{
			name:    "CSCxCSC",
			transA:  false,
			a:       CSCFormat,
			b:       CSCFormat,
			density: 0.01,
		},
		{
			name:    "CSCxCSR",
			transA:  false,
			a:       CSCFormat,
			b:       CSRFormat,
			density: 0.01,
		},
		{
			name:    "CSCxCOO",
			transA:  false,
			a:       CSCFormat,
			b:       COOFormat,
			density: 0.01,
		},
	}

	cMat := mat.NewDense(ar, bc, nil)

	for _, bench := range benchmarks {

		aMat := Random(bench.a, ar, ac, bench.density).(BlasCompatibleSparser)
		bMat := Random(bench.b, br, bc, bench.density)

		c := cMat.RawMatrix()
		for i := range c.Data {
			c.Data[i] = 0
		}

		b.Run(bench.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cMat = MulMatMat(bench.transA, 1, aMat, bMat, cMat)
			}
		})
	}
}

// dot is a package level variable to hold the result of dot benchmark to prevent
// compiler from optimising out the call.
var dot float64

func BenchmarkDot(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	population := 0.01
	dim := 100000

	adata := make([]float64, dim)
	bdata := make([]float64, dim)

	pop := int(float64(dim) * population)
	for i := 1; i <= pop; i++ {
		adata[rnd.Intn(dim)] = float64(i)
		bdata[rnd.Intn(dim)] = float64(i)
	}

	benchmarks := []struct {
		name string
		af   vector
		bf   vector
		fn   func(mat.Vector, mat.Vector) float64
	}{
		{name: "Mat Dense Dense", af: denseVec, bf: denseVec, fn: mat.Dot},
		{name: "Mat Sparse Sparse", af: sparseVec, bf: sparseVec, fn: mat.Dot},
		{name: "Mat Dense Sparse", af: denseVec, bf: sparseVec, fn: mat.Dot},
		{name: "Mat Sparse Dense", af: denseVec, bf: sparseVec, fn: mat.Dot},

		{name: "Sparse Sparse Sparse", af: sparseVec, bf: sparseVec, fn: Dot},
		{name: "Sparse Sparse Dense", af: sparseVec, bf: denseVec, fn: Dot},
		{name: "Sparse Dense Sparse", af: denseVec, bf: sparseVec, fn: Dot},
		{name: "Sparse Dense Dense", af: denseVec, bf: denseVec, fn: Dot},
	}

	for _, bench := range benchmarks {
		av := bench.af(adata)
		bv := bench.bf(bdata)

		b.Run(bench.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				dot = bench.fn(av, bv)
			}
		})
	}
}

type vector func([]float64) mat.Vector

func sparseVec(s []float64) mat.Vector {
	var data []float64
	var ind []int

	for i, v := range s {
		if v != 0 {
			data = append(data, v)
			ind = append(ind, i)
		}
	}
	return NewVector(len(s), ind, data)
}

func denseVec(s []float64) mat.Vector {
	return mat.NewVecDense(len(s), s)
}
