package sparse

import (
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser       = (*DOK)(nil)
	_ TypeConverter = (*DOK)(nil)
	_ mat.Mutable   = (*DOK)(nil)
)

// key is used to specify the row and column of elements within the matrix.
type key struct {
	i, j int
}

// DOK is a Dictionary Of Keys sparse matrix implementation and implements the Matrix interface from gonum/matrix.
// This allows large sparse (mostly zero values) matrices to be stored efficiently in memory (only storing
// non-zero values).  DOK matrices are good for incrementally constructing sparse matrices but poor for arithmetic
// operations or other operations that require iterating over elements of the matrix sequentially.  As this type
// implements the gonum mat.Matrix interface, it may be used with any of the Gonum mat functions that accept
// Matrix types as parameters in place of other matrix types included in the Gonum mat package e.g. mat.Dense.
type DOK struct {
	r        int
	c        int
	elements map[key]float64
}

// NewDOK creates a new Dictionary Of Keys format sparse matrix initialised to the size of the specified r * c
// dimensions (rows * columns)
func NewDOK(r, c int) *DOK {
	if r < 0 {
		panic(mat.ErrRowAccess)
	}
	if c < 0 {
		panic(mat.ErrColAccess)
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
	if i < 0 || i >= d.r {
		panic(mat.ErrRowAccess)
	}
	if j < 0 || j >= d.c {
		panic(mat.ErrColAccess)
	}

	return d.elements[key{i, j}]
}

// T transposes the matrix.  This is an implicit transpose, wrapping the matrix in a mat.Transpose type.
func (d *DOK) T() mat.Matrix {
	return mat.Transpose{Matrix: d}
}

// Set sets the element of the matrix located at row i and column j to equal the specified value, v.  Set
// will panic if specified values for i or j fall outside the dimensions of the matrix.
func (d *DOK) Set(i, j int, v float64) {
	if i < 0 || i >= d.r {
		panic(mat.ErrRowAccess)
	}
	if j < 0 || j >= d.c {
		panic(mat.ErrColAccess)
	}

	d.elements[key{i, j}] = v
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).  The order of visiting to each non-zero element in the receiver is random.
func (d *DOK) DoNonZero(fn func(i, j int, v float64)) {
	for k, v := range d.elements {
		fn(k.i, k.j, v)
	}
}

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (d *DOK) NNZ() int {
	return len(d.elements)
}

// RawMatrix converts the matrix to a CSR matrix and returns a pointer to
// the underlying blas sparse matrix.
func (d *DOK) RawMatrix() *blas.SparseMatrix {
	return d.ToCSR().RawMatrix()
}

// ToDense returns a mat.Dense dense format version of the matrix.  The returned mat.Dense
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (d *DOK) ToDense() *mat.Dense {
	mat := mat.NewDense(d.r, d.c, nil)

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
	return d.ToCOO().ToCSRReuseMem()
}

// ToCSC returns a CSC (Compressed Sparse Column)(AKA CCS (Compressed Column Storage)) sparse format
// version of the matrix.  The returned CSC matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (d *DOK) ToCSC() *CSC {
	return d.ToCOO().ToCSCReuseMem()
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (d *DOK) ToType(matType MatrixType) mat.Matrix {
	return matType.Convert(d)
}

// MulVecTo performs matrix vector multiplication (dst+=A*x or dst+=A^T*x), where A is
// the receiver, and stores the result in dst.  MulVecTo panics if ac != len(x) or
// ar != len(dst)
func (d *DOK) MulVecTo(dst []float64, trans bool, x []float64) {
	d.ToCSR().MulVecTo(dst, trans, x)
}
