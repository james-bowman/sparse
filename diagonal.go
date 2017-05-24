package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// DIA matrix type is a specialised matrix designed to store DIAgonal values of square symmetrical
// matrices (all zero values except along the diagonal running top left to bottom right).  The DIA matrix
// type is specifically designed to take advantage of the sparsity pattern of square symmetrical matrices.
type DIA struct {
	m    int
	data []float64
}

// NewDIA creates a new DIAgonal format sparse matrix.
// The matrix is initialised to the size of the specified m * m dimensions (rows * columns)
// (creating a square) with the specified slice containing it's diagonal values.  The diagonal slice
// will be used as the backing slice to the matrix so changes to values of the slice will be reflected
// in the matrix.
func NewDIA(m int, diagonal []float64) *DIA {
	if uint(m) < 0 || m != len(diagonal) {
		panic(matrix.ErrRowAccess)
	}

	return &DIA{m: m, data: diagonal}
}

// Dims returns the size of the matrix as the number of rows and columns
func (d *DIA) Dims() (int, int) {
	return d.m, d.m
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.
func (d *DIA) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(d.m) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(d.m) {
		panic(matrix.ErrColAccess)
	}

	if i == j {
		return d.data[i]
	}
	return 0
}

// T returns the matrix transposed.  In the case of a DIA (DIAgonal) sparse matrix this method
// simply returns the receiver as the matrix is perfectly symmetrical and transposing has no effect.
func (d *DIA) T() mat64.Matrix {
	return d
}

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (d *DIA) NNZ() int {
	return d.m
}

// Diagonal returns the diagonal values of the matrix from top left to bottom right.
// The values are returned as a slice backed by the same array as backing the receiver
// so changes to values in the returned slice will be reflected in the receiver.
func (d *DIA) Diagonal() []float64 {
	return d.data
}
