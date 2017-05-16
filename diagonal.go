package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type DIA struct {
	m    int
	data []float64
}

func NewDIA(m int, diagonal []float64) *DIA {
	if uint(m) < 0 {
		panic(matrix.ErrRowAccess)
	}

	return &DIA{m: m, data: diagonal}
}

func (d *DIA) Dims() (int, int) {
	return d.m, d.m
}

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
// simply returns the receiver as the matrix is symmetrical and transposing has no effect.
func (d *DIA) T() mat64.Matrix {
	return d
}

func (d *DIA) NNZ() int {
	return d.m
}

func (d *DIA) Diagonal() []float64 {
	return d.data
}
