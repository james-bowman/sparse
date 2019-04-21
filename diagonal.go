package sparse

import (
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser       = (*DIA)(nil)
	_ mat.ColViewer = (*DIA)(nil)
	_ mat.RowViewer = (*DIA)(nil)
)

// DIA matrix type is a specialised matrix designed to store DIAgonal values of square symmetrical
// matrices (all zero values except along the diagonal running top left to bottom right).  The DIA matrix
// type is specifically designed to take advantage of the sparsity pattern of square symmetrical matrices.
type DIA struct {
	m, n int
	data []float64
}

// NewDIA creates a new DIAgonal format sparse matrix.
// The matrix is initialised to the size of the specified m * m dimensions (rows * columns)
// (creating a square) with the specified slice containing it's diagonal values.  The diagonal slice
// will be used as the backing slice to the matrix so changes to values of the slice will be reflected
// in the matrix.
func NewDIA(m int, n int, diagonal []float64) *DIA {
	if uint(m) < 0 || m < len(diagonal) {
		panic(mat.ErrRowAccess)
	}
	if uint(n) < 0 || n < len(diagonal) {
		panic(mat.ErrColAccess)
	}

	return &DIA{m: m, n: n, data: diagonal}
}

// Dims returns the size of the matrix as the number of rows and columns
func (d *DIA) Dims() (int, int) {
	return d.m, d.n
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.
func (d *DIA) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(d.m) {
		panic(mat.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.n) {
		panic(mat.ErrColAccess)
	}

	if i == j {
		return d.data[i]
	}
	return 0
}

// T returns the matrix transposed.  In the case of a DIA (DIAgonal) sparse matrix this method
// returns a new DIA matrix with the m and n values transposed.
func (d *DIA) T() mat.Matrix {
	return &DIA{m: d.n, n: d.m, data: d.data}
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).  The order of visiting to each non-zero element is from top left to bottom right.
func (d *DIA) DoNonZero(fn func(i, j int, v float64)) {
	for i, v := range d.data {
		fn(i, i, v)
	}
}

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (d *DIA) NNZ() int {
	return len(d.data)
}

// Diagonal returns the diagonal values of the matrix from top left to bottom right.
// The values are returned as a slice backed by the same array as backing the receiver
// so changes to values in the returned slice will be reflected in the receiver.
func (d *DIA) Diagonal() []float64 {
	return d.data
}

// RowView slices the matrix and returns a Vector containing a copy of elements
// of row i.
func (d *DIA) RowView(i int) mat.Vector {
	return mat.NewVecDense(d.n, d.ScatterRow(i, nil))
}

// ColView slices the matrix and returns a Vector containing a copy of elements
// of column j.
func (d *DIA) ColView(j int) mat.Vector {
	return mat.NewVecDense(d.m, d.ScatterCol(j, nil))
}

// ScatterRow returns a slice representing row i of the matrix in dense format.  row
// is used as the storage for the operation unless it is nil in which case, new
// storage of the correct length will be allocated.  This method will panic if i
// is out of range or row is not the same length as the number of columns in the matrix i.e.
// the correct size to receive the dense representation of the row.
func (d *DIA) ScatterRow(i int, row []float64) []float64 {
	if i >= d.m || i < 0 {
		panic(mat.ErrRowAccess)
	}
	if row != nil && len(row) != d.n {
		panic(mat.ErrRowLength)
	}
	if row == nil {
		row = make([]float64, d.n)
	}
	if i < len(d.data) {
		row[i] = d.data[i]
	}
	return row
}

// ScatterCol returns a slice representing column j of the matrix in dense format.  Col
// is used as the storage for the operation unless it is nil in which case, new
// storage of the correct length will be allocated.  This method will panic if j
// is out of range or col is not the same length as the number of rows in the matrix i.e.
// the correct size to receive the dense representation of the column.
func (d *DIA) ScatterCol(j int, col []float64) []float64 {
	if j >= d.n || j < 0 {
		panic(mat.ErrColAccess)
	}
	if col != nil && len(col) != d.m {
		panic(mat.ErrColLength)
	}
	if col == nil {
		col = make([]float64, d.m)
	}
	if j < len(d.data) {
		col[j] = d.data[j]
	}
	return col
}

// MulVecTo performs matrix vector multiplication (dst+=A*x or dst+=A^T*x), where A is
// the receiver, and stores the result in dst.  MulVecTo panics if ac != len(x) or
// ar != len(dst)
func (d *DIA) MulVecTo(dst []float64, trans bool, x []float64) {
	if !trans {
		if d.n != len(x) || d.m != len(dst) {
			panic(mat.ErrShape)
		}
	} else {
		if d.m != len(x) || d.n != len(dst) {
			panic(mat.ErrShape)
		}
	}

	for i, v := range d.data {
		dst[i] += v * x[i]
	}
}

// Trace returns the trace.
func (d *DIA) Trace() float64 {
	return floats.Sum(d.data)
}
