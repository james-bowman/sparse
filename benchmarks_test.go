package sparse

import (
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
