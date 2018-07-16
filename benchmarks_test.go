package sparse

import (
	"fmt"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

type Product interface {
	Mul(a, b mat.Matrix)
	Add(a, b mat.Matrix)
}

var products = []Product{
	&mat.Dense{},
	&CSR{},
}

var densities = []float32{
	0.01,
	//0.1,
	0.4,
	//	0.6,
}

type MatMultiplyer interface {
	Mul(a, b mat.Matrix)
}

func benchmarkMatrixMultiplication(target MatMultiplyer, lhs mat.Matrix, rhs mat.Matrix, b *testing.B) {
	for n := 0; n < b.N; n++ {
		target.Mul(lhs, rhs)
	}
}

// DIAgonal Multiplication

func BenchmarkMulLargeCSRDIACSR(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := RandomDIA(500, 600)
	rhs := Random(CSRFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeCSRCSRDIA(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSRFormat, 500, 600, 0.01)
	rhs := RandomDIA(600, 500)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeCSRDIACSC(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := RandomDIA(500, 600)
	rhs := Random(CSCFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeCSRCSCDIA(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(CSCFormat, 500, 600, 0.01)
	rhs := RandomDIA(600, 500)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeCSRDIADense(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := RandomDIA(500, 600)
	rhs := Random(DenseFormat, 600, 500, 0.01)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func BenchmarkMulLargeCSRDenseDIA(b *testing.B) {
	t := CreateCSR(0, 0, nil).(*CSR)
	lhs := Random(DenseFormat, 500, 600, 0.01)
	rhs := RandomDIA(600, 500)
	benchmarkMatrixMultiplication(t, lhs, rhs, b)
}

func RandomDIA(r, c int) *DIA {
	var min int
	if r < c {
		min = r
	} else {
		min = c
	}
	data := make([]float64, min)
	for i := range data {
		data[i] = rand.Float64()
	}
	return NewDIA(r, c, data)
}

func BenchmarkAdd(b *testing.B) {
	var dimensions = []struct {
		ar, ac, br, bc int
	}{
		//{ar: 5, ac: 6, br: 5, bc: 6},
		{ar: 500, ac: 600, br: 500, bc: 600},
	}

	benchmarks := []struct {
		name string
		a    MatrixType
		b    MatrixType
	}{
		{
			name: "CSR+CSR",
			a:    CSRFormat,
			b:    CSRFormat,
		},
		{
			name: "CSR+Dense",
			a:    CSRFormat,
			b:    DenseFormat,
		},
		{
			name: "Dense+CSR",
			a:    DenseFormat,
			b:    CSRFormat,
		},
		{
			name: "Dense+Dense",
			a:    DenseFormat,
			b:    DenseFormat,
		},
		{
			name: "CSR+CSC",
			a:    CSRFormat,
			b:    CSCFormat,
		},
		{
			name: "CSC+CSC",
			a:    CSCFormat,
			b:    CSCFormat,
		},
	}

	for _, dims := range dimensions {
		for _, density := range densities {
			for _, c := range products {
				for _, bench := range benchmarks {
					if cs, resetable := c.(mat.Reseter); resetable {
						cs.Reset()
					}
					aMat := Random(bench.a, dims.ar, dims.ac, density)
					bMat := Random(bench.b, dims.br, dims.bc, density)

					b.Run(fmt.Sprintf("%dx%d (%.2f) %T=%s", dims.ar, dims.bc, density, c, bench.name), func(b *testing.B) {
						for i := 0; i < b.N; i++ {
							c.Add(aMat, bMat)
						}
					})
				}
			}
		}
	}
}

func BenchmarkMul(b *testing.B) {
	var dimensions = []struct {
		ar, ac, br, bc int
	}{
		//{ar: 5, ac: 6, br: 6, bc: 5},
		{ar: 500, ac: 600, br: 600, bc: 500},
	}

	benchmarks := []struct {
		name string
		a    MatrixType
		b    MatrixType
	}{
		{
			name: "Dense*Dense",
			a:    DenseFormat,
			b:    DenseFormat,
		},
		{
			name: "Dense*CSR",
			a:    CSRFormat,
			b:    DenseFormat,
		},
		{
			name: "CSR*Dense",
			a:    CSRFormat,
			b:    DenseFormat,
		},
		{
			name: "CSR*CSR",
			a:    CSRFormat,
			b:    CSRFormat,
		},
		{
			name: "CSR*CSC",
			a:    CSRFormat,
			b:    CSCFormat,
		},
		{
			name: "CSC*CSR",
			a:    CSCFormat,
			b:    CSRFormat,
		},
		{
			name: "CSC*CSC",
			a:    CSCFormat,
			b:    CSCFormat,
		},
		{
			name: "CSR*DOK",
			a:    CSRFormat,
			b:    DOKFormat,
		},
		{
			name: "Dense*DOK",
			a:    DenseFormat,
			b:    DOKFormat,
		},
		{
			name: "DOK*DOK",
			a:    DOKFormat,
			b:    DOKFormat,
		},
		// {
		// 	name: "CSR*COO",
		// 	a:    CSRFormat,
		// 	b:    COOFormat,
		// },
	}

	for _, dims := range dimensions {
		for _, density := range densities {
			for _, c := range products {
				for _, bench := range benchmarks {
					if cs, resetable := c.(mat.Reseter); resetable {
						cs.Reset()
					}
					aMat := Random(bench.a, dims.ar, dims.ac, density)
					bMat := Random(bench.b, dims.br, dims.bc, density)

					b.Run(fmt.Sprintf("%dx%d (%.2f) %T=%s", dims.ar, dims.bc, density, c, bench.name), func(b *testing.B) {
						for i := 0; i < b.N; i++ {
							c.Mul(aMat, bMat)
						}
					})
				}
			}
		}
	}
}

func BenchmarkBLASMulMatMat(b *testing.B) {
	var dimensions = []struct {
		ar, ac, br, bc int
	}{
		//{ar: 5, ac: 6, br: 6, bc: 5},
		{ar: 500, ac: 600, br: 600, bc: 500},
	}
	benchmarks := []struct {
		name   string
		transA bool
		alpha  float64
		a      MatrixType
		b      MatrixType
	}{
		{
			name:   "CSRxDense",
			transA: false,
			alpha:  1,
			a:      CSRFormat,
			b:      DenseFormat,
		},
		{
			name:   "CSCxDense",
			transA: false,
			alpha:  1,
			a:      CSCFormat,
			b:      DenseFormat,
		},
		{
			name:   "COOxDense",
			transA: false,
			alpha:  1,
			a:      COOFormat,
			b:      DenseFormat,
		},
		{
			name:   "DOKxDense",
			transA: false,
			alpha:  1,
			a:      DOKFormat,
			b:      DenseFormat,
		},
		{
			name:   "CSRxCSC",
			transA: false,
			alpha:  1,
			a:      CSRFormat,
			b:      CSCFormat,
		},
		{
			name:   "CSRxCSR",
			transA: false,
			alpha:  1,
			a:      CSRFormat,
			b:      CSRFormat,
		},
		{
			name:   "CSRxCOO",
			transA: false,
			alpha:  1,
			a:      CSRFormat,
			b:      COOFormat,
		},
		{
			name:   "CSCxCSC",
			transA: false,
			alpha:  1,
			a:      CSCFormat,
			b:      CSCFormat,
		},
		{
			name:   "CSCxCSR",
			transA: false,
			alpha:  1,
			a:      CSCFormat,
			b:      CSRFormat,
		},
		{
			name:   "CSCxCOO",
			transA: false,
			alpha:  1,
			a:      CSCFormat,
			b:      COOFormat,
		},
	}

	for _, dims := range dimensions {
		for _, density := range densities {
			for _, bench := range benchmarks {
				cMat := mat.NewDense(dims.ar, dims.bc, nil)
				aMat := Random(bench.a, dims.ar, dims.ac, density).(BlasCompatibleSparser)
				bMat := Random(bench.b, dims.br, dims.bc, density)

				c := cMat.RawMatrix()
				for i := range c.Data {
					c.Data[i] = 0
				}

				b.Run(fmt.Sprintf("%dx%d (%.2f) %s", dims.ar, dims.bc, density, bench.name), func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						cMat = MulMatMat(bench.transA, 1, aMat, bMat, cMat)
					}
				})
			}
		}
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

func BenchmarkNorm(b *testing.B) {
	ind := []int{0, 100, 200, 300, 400, 500, 600, 700, 800, 900}
	data := []float64{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	vec := NewVector(1000, ind, data)

	benchmarks := []struct {
		name string
		f    func(float64) float64
	}{
		{name: "norm", f: vec.Norm},
	}

	var v float64
	for _, bench := range benchmarks {
		b.Run(bench.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				v = bench.f(2)
			}
		})
	}
	_ = v
}
