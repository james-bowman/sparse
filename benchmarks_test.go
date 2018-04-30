package sparse

import (
	"math/rand"
	"testing"

	"github.com/james-bowman/sparse/blas"
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

func BenchmarkMulBLASLargeDenseCSRDense(b *testing.B) {
	ar, ac := 500, 600
	t := mat.NewDense(ar, ar, nil)
	lhs := Random(CSRFormat, ar, ac, 0.01).(*CSR)
	rhs := Random(DenseFormat, ac, ar, 0.01).(*mat.Dense)

	a := lhs.RawMatrix()

	rawB := rhs.RawMatrix()
	rawC := t.RawMatrix()

	for n := 0; n < b.N; n++ {
		blas.Dusmm(false, ar, 1, a, rawB.Data, rawB.Stride, rawC.Data, rawC.Stride)
	}
}

func BenchmarkMulBLASLargeDenseCSRCSC(b *testing.B) {
	ar, ac := 500, 600
	t := mat.NewDense(ar, ar, nil)
	lhs := Random(CSRFormat, ar, ac, 0.01).(*CSR)
	rhs := Random(CSCFormat, ac, ar, 0.01).(*CSC)

	a := lhs.RawMatrix()
	br := rhs.RawMatrix()

	rawC := t.RawMatrix()
	y := make([]float64, ac)

	for n := 0; n < b.N; n++ {
		for i := 0; i < ar; i++ {
			ind := br.Ind[br.Indptr[i]:br.Indptr[i+1]]
			blas.Dussc(br.Data[br.Indptr[i]:br.Indptr[i+1]], y, 1, ind)
			blas.Dusmv(false, 1, a, y, 1, rawC.Data[i:], rawC.Stride)
			for _, v := range ind {
				y[v] = 0
			}
		}
	}
}

func BenchmarkMulBLASLargeCSCCSRCSC(b *testing.B) {
	ar, ac := 500, 600

	lhs := Random(CSRFormat, ar, ac, 0.01).(*CSR)
	rhs := Random(CSCFormat, ac, ar, 0.01).(*CSC)

	a := lhs.RawMatrix()
	br := rhs.RawMatrix()

	indptr := make([]int, ar+1)
	indptr[0] = 0
	var indx []int
	var data []float64

	y := make([]float64, ac)
	z := make([]float64, ar)

	for n := 0; n < b.N; n++ {
		for i := 0; i < ar; i++ {
			ind := br.Ind[br.Indptr[i]:br.Indptr[i+1]]
			blas.Dussc(br.Data[br.Indptr[i]:br.Indptr[i+1]], y, 1, ind)
			blas.Dusmv(false, 1, a, y, 1, z, 1)

			for k, v := range z {
				if v != 0 {
					data = append(data, v)
					ind = append(indx, k)
					z[k] = 0
				}
			}
			indptr[i+1] = len(data)
		}
	}
	NewCSC(ar, ar, indptr, indx, data)
}

// dot is a package level variable to hold the result of dot benchmark to prevent
// compiler from optimising out the call.
var dot float64

func BenchmarkDot(b *testing.B) {
	rnd := rand.New(rand.NewSource(0))
	population := 0.3
	dim := 100000000

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
			dot = bench.fn(av, bv)
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
