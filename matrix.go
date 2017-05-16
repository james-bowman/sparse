package sparse

import (
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

type Sparser interface {
	mat64.Matrix
	NNZ() int

	ToDense() *mat64.Dense
	ToDOK() *DOK
	ToCOO() *COO
	ToCSR() *CSR
	ToCSC() *CSC
	ToType(matType MatrixType) mat64.Matrix
}

type MatrixType interface {
	Convert(from Sparser) mat64.Matrix
}

type DenseType int

func (d DenseType) Convert(from Sparser) mat64.Matrix {
	return from.ToDense()
}

type DOKType int

func (s DOKType) Convert(from Sparser) mat64.Matrix {
	return from.ToDOK()
}

type COOType int

func (s COOType) Convert(from Sparser) mat64.Matrix {
	return from.ToCOO()
}

type CSRType int

func (s CSRType) Convert(from Sparser) mat64.Matrix {
	return from.ToCSR()
}

type CSCType int

func (s CSCType) Convert(from Sparser) mat64.Matrix {
	return from.ToCSC()
}

const (
	DenseFormat DenseType = iota
	DOKFormat   DOKType   = iota
	COOFormat   COOType   = iota
	CSRFormat   CSRType   = iota
	CSCFormat   CSCType   = iota
)

func Random(t MatrixType, r int, c int, density float32) mat64.Matrix {
	d := int(density * float32(r) * float32(c))

	m := make([]int, d)
	n := make([]int, d)
	data := make([]float64, d)

	for i := 0; i < d; i++ {
		data[i] = rand.Float64()
		m[i] = rand.Intn(r)
		n[i] = rand.Intn(c)
	}

	return NewCOO(r, c, m, n, data).ToType(t)
}
