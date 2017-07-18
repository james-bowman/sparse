package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// key is used to specify the row and column of elements within the matrix.
type key struct {
	i, j int
}

// DOK is a Dictionary Of Keys sparse matrix implementation and implements the Matrix interface from gonum/matrix.
// This allows large sparse (mostly zero values) matrices to be stored efficiently in memory (only storing
// non-zero values).  DOK matrices are good for incrementally constructing sparse matrices but poor for arithmetic
// operations or other operations that require iterating over elements of the matrix sequentially.  As this type
// implements the gonum mat64.Matrix interface, it may be used with any of the Gonum mat64 functions that accept
// Matrix types as parameters in place of other matrix types included in the Gonum mat64 package e.g. mat64.Dense.
type DOK struct {
	r        int
	c        int
	elements map[key]float64
}

// NewDOK creates a new Dictionary Of Keys format sparse matrix initialised to the size of the specified r * c
// dimensions (rows * columns)
func NewDOK(r, c int) *DOK {
	if uint(r) < 0 {
		panic(matrix.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(matrix.ErrColAccess)
	}

	return &DOK{r: r, c: c, elements: make(map[key]float64)}
}

// Dims returns the size of the matrix as the number of rows and columns
func (d *DOK) Dims() (r, c int) {
	return d.r, d.c
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.
func (d *DOK) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(d.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.c) {
		panic(matrix.ErrColAccess)
	}

	return d.elements[key{i, j}]
}

// T transposes the matrix.  This is an implicit transpose, wrapping the matrix in a mat64.Transpose type.
func (d *DOK) T() mat64.Matrix {
	return mat64.Transpose{d}
}

// Set sets the element of the matrix located at row i and column j to equal the specified value, v.  Set
// will panic if specified values for i or j fall outside the dimensions of the matrix.
func (d *DOK) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(d.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.c) {
		panic(matrix.ErrColAccess)
	}

	d.elements[key{i, j}] = v
}

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (d *DOK) NNZ() int {
	return len(d.elements)
}

// ToDense returns a mat64.Dense dense format version of the matrix.  The returned mat64.Dense
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (d *DOK) ToDense() *mat64.Dense {
	mat := mat64.NewDense(d.r, d.c, nil)

	for k, v := range d.elements {
		mat.Set(k.i, k.j, v)
	}

	return mat
}

// ToDOK returns the receiver
func (d *DOK) ToDOK() *DOK {
	return d
}

// ToCOO returns a COOrdinate sparse format version of the matrix.  The returned COO matrix will
// not share underlying storage with the receiver nor is the receiver modified by this call.
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

// ToCSR returns a CSR (Compressed Sparse Row)(AKA CRS (Compressed Row Storage)) sparse format
// version of the matrix.  The returned CSR matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (d *DOK) ToCSR() *CSR {
	return d.ToCOO().ToCSR()
}

// ToCSC returns a CSC (Compressed Sparse Column)(AKA CCS (Compressed Column Storage)) sparse format
// version of the matrix.  The returned CSC matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (d *DOK) ToCSC() *CSC {
	return d.ToCOO().ToCSC()
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (d *DOK) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(d)
}

// RowView slices the matrix and returns a Vector containing a copy of elements
// of row i.
func (d *DOK) RowView(i int) *mat64.Vector {
	return mat64.NewVector(d.c, d.RawRowView(i))
}

// ColView slices the matrix and returns a Vector containing a copy of elements
// of column i.
func (d *DOK) ColView(j int) *mat64.Vector {
	return mat64.NewVector(d.r, d.RawColView(j))
}

// RawRowView returns a slice representing row i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func (d *DOK) RawRowView(i int) []float64 {
	return rawRowView(d, i)
}

// RawColView returns a slice representing col i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func (d *DOK) RawColView(j int) []float64 {
	return rawColView(d, j)
}
