package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type key struct {
	i, j int
}

// DOK is a Dictionary Of Keys sparse matrix implementation and implements the Matrix interface from gonum/matrix.
// This allows large sparse (mostly zero values) matrices to be stored efficiently in memory (only storing
// non-zero values).
type DOK struct {
	r        int
	c        int
	elements map[key]float64
}

func NewDOK(r, c int) *DOK {
	if uint(r) < 0 {
		panic(matrix.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(matrix.ErrColAccess)
	}

	return &DOK{r: r, c: c, elements: make(map[key]float64)}
}

func (d *DOK) Dims() (r, c int) {
	return d.r, d.c
}

func (d *DOK) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(d.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.c) {
		panic(matrix.ErrColAccess)
	}

	return d.elements[key{i, j}]
}

func (d *DOK) T() mat64.Matrix {
	return mat64.Transpose{d}
}

func (d *DOK) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(d.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.c) {
		panic(matrix.ErrColAccess)
	}

	d.elements[key{i, j}] = v
}

func (d *DOK) NNZ() int {
	return len(d.elements)
}

func (d *DOK) ToDense() *mat64.Dense {
	mat := mat64.NewDense(d.r, d.c, nil)

	for k, v := range d.elements {
		mat.Set(k.i, k.j, v)
	}

	return mat
}

func (d *DOK) ToDOK() *DOK {
	return d
}

func (d *DOK) ToCOO() *COO {
	nnz := d.NNZ()
	rows := make([]int, nnz)
	cols := make([]int, nnz)
	data := make([]float64, nnz)

	i := 0
	for k, v := range d.elements {
		rows[i], cols[i], data[i] = k.i, k.j, v
		i++
	}

	coo := NewCOO(d.r, d.c, rows, cols, data)

	return coo
}

func (d *DOK) ToCSR() *CSR {
	return d.ToCOO().ToCSR()
}

func (d *DOK) ToCSC() *CSC {
	return d.ToCOO().ToCSC()
}

func (d *DOK) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(d)
}
