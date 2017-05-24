package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type compressedSparse struct {
	i, j   int
	indptr []int
	ind    []int
	data   []float64
}

func (c *compressedSparse) NNZ() int {
	return len(c.data)
}

func (c *compressedSparse) at(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(c.i) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.j) {
		panic(matrix.ErrColAccess)
	}

	// todo: consider a binary search if we can assume the data is ordered.
	for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
		if c.ind[k] == j {
			return c.data[k]
		}
	}

	return 0
}

type CSR struct {
	compressedSparse
}

func NewCSR(r int, c int, ia []int, ja []int, data []float64) *CSR {
	if uint(r) < 0 {
		panic(matrix.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(matrix.ErrColAccess)
	}

	return &CSR{
		compressedSparse: compressedSparse{
			i: r, j: c,
			indptr: ia,
			ind:    ja,
			data:   data,
		},
	}
}

func (c *CSR) Dims() (int, int) {
	return c.i, c.j
}

func (c *CSR) At(m, n int) float64 {
	return c.at(m, n)
}

func (c *CSR) T() mat64.Matrix {
	return NewCSC(c.j, c.i, c.indptr, c.ind, c.data)
}

func (c *CSR) ToDense() *mat64.Dense {
	mat := mat64.NewDense(c.i, c.j, nil)

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			mat.Set(i, c.ind[j], c.data[j])
		}
	}

	return mat
}

func (c *CSR) ToDOK() *DOK {
	dok := NewDOK(c.i, c.j)
	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			dok.Set(i, c.ind[j], c.data[j])
		}
	}

	return dok
}

func (c *CSR) ToCOO() *COO {
	rows := make([]int, c.NNZ())

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			rows[j] = i
		}
	}

	coo := NewCOO(c.i, c.j, rows, c.ind, c.data)

	return coo
}

func (c *CSR) ToCSR() *CSR {
	return c
}

func (c *CSR) ToCSC() *CSC {
	return c.ToCOO().ToCSC()
}

func (c *CSR) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(c)
}

func (c *CSR) Mul(a, b mat64.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if dia, ok := a.(*DIA); ok {
		if ac != br {
			panic(matrix.ErrShape)
		}
		c.mulDIA(dia, b, false)
		return
	}
	if dia, ok := b.(*DIA); ok {
		if bc != ar {
			panic(matrix.ErrShape)
		}
		c.mulDIA(dia, a, true)
		return
	}

	if ac != br {
		panic(matrix.ErrShape)
	}

	c.indptr = make([]int, ar+1)

	c.i, c.j = ar, bc
	t := 0

	lhs, isCsr := a.(*CSR)

	if isCsr {
		for i := 0; i < ar; i++ {
			c.indptr[i] = t
			for j := 0; j < bc; j++ {
				var v float64
				// TODO Consider converting all Sparsers to CSR
				for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
					v += lhs.data[k] * b.At(lhs.ind[k], j)
				}
				if v != 0 {
					t++
					c.ind = append(c.ind, j)
					c.data = append(c.data, v)
				}
			}
		}
	} else {
		row := make([]float64, ac)
		for i := 0; i < ar; i++ {
			c.indptr[i] = t
			for ci := range row {
				row[ci] = a.At(i, ci)
			}
			for j := 0; j < bc; j++ {
				var v float64
				for ci, e := range row {
					v += e * b.At(ci, j)
				}
				if v != 0 {
					t++
					c.ind = append(c.ind, j)
					c.data = append(c.data, v)
				}
			}
		}
	}

	c.indptr[c.i] = t
}

func (c *CSR) mulDIA(dia *DIA, other mat64.Matrix, trans bool) {
	var csMat compressedSparse
	isCS := false

	if csr, ok := other.(*CSR); ok {
		// TODO consider converting all sparsers to CSR (for RHS operand)
		csMat = csr.compressedSparse
		isCS = true
		c.ind = make([]int, len(csMat.ind))
		c.data = make([]float64, len(csMat.data))
	}

	c.i, c.j = other.Dims()
	c.indptr = make([]int, c.i+1)
	t := 0
	raw := dia.Diagonal()

	for i := 0; i < c.i; i++ {
		c.indptr[i] = t
		var v float64

		if isCS {
			for k := csMat.indptr[i]; k < csMat.indptr[i+1]; k++ {
				var rawval float64
				if trans {
					rawval = raw[csMat.ind[k]]
				} else {
					rawval = raw[i]
				}
				v = csMat.data[k] * rawval
				if v != 0 {
					c.ind[t] = csMat.ind[k]
					c.data[t] = v
					t++
				}
			}
		} else {
			for k := 0; k < c.j; k++ {
				var rawval float64
				if trans {
					rawval = raw[k]
				} else {
					rawval = raw[i]
				}
				v = other.At(i, k) * rawval
				if v != 0 {
					c.ind = append(c.ind, k)
					c.data = append(c.data, v)
					t++
				}
			}
		}
	}

	c.indptr[c.i] = t

	return
}

func (c *CSR) Mul2(a, b mat64.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	c.indptr = make([]int, ar+1)

	if rhs, ok := b.(*DIA); ok {
		c.i, c.j = ar, ac
		var size int
		if lhs, ok := a.(*CSR); ok {
			size = lhs.NNZ()
		} else {
			size = ar * bc
		}
		c.ind = make([]int, size)
		c.data = make([]float64, size)
		t := 0
		raw := rhs.Diagonal()

		for i := 0; i < ar; i++ {
			c.indptr[i] = t
			var v float64
			if lhs, ok := a.(*CSR); ok {
				for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
					v = lhs.data[k] * raw[i]
					if v != 0 {
						c.ind[t] = k
						c.data[t] = v
						t++
					}
				}

			} else {
				for k := 0; k < ac; k++ {
					v = a.At(i, k) * raw[i]
					if v != 0 {
						c.ind[t] = k
						c.data[t] = v
						t++
					}
				}
			}
		}

		c.indptr[c.i] = t
		c.ind = c.ind[:t]
		c.data = c.data[:t]
		return
	}

	if ac != br {
		panic(matrix.ErrShape)
	}

	c.i, c.j = ar, bc
	c.ind = make([]int, ar*bc)
	c.data = make([]float64, ar*bc)

	t := 0

	for i := 0; i < ar; i++ {
		c.indptr[i] = t
		for j := 0; j < bc; j++ {
			var v float64
			if lhs, ok := a.(*CSR); ok {
				for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
					v += lhs.data[k] * b.At(lhs.ind[k], j)
				}
				if v != 0 {
					c.ind[t] = j
					c.data[t] = v
					t++
				}
			} else {
				for k := 0; k < ac; k++ {
					v += a.At(i, k) * b.At(k, j)
				}
				if v != 0 {
					c.ind[t] = j
					c.data[t] = v
					t++
				}
				//c.set(i, j, v)
			}
		}
	}
	c.indptr[c.i] = t
	c.ind = c.ind[:t]
	c.data = c.data[:t]
}

func (c *CSR) RowNNZ(i int) int {
	if uint(i) < 0 || uint(i) >= uint(c.i) {
		panic(matrix.ErrRowAccess)
	}
	return c.indptr[i+1] - c.indptr[i]
}

type CSC struct {
	compressedSparse
}

func NewCSC(r int, c int, indptr []int, ind []int, data []float64) *CSC {
	if uint(r) < 0 {
		panic(matrix.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(matrix.ErrColAccess)
	}

	return &CSC{
		compressedSparse: compressedSparse{
			i: c, j: r,
			indptr: indptr,
			ind:    ind,
			data:   data,
		},
	}
}

func (c *CSC) Dims() (int, int) {
	return c.j, c.i
}

func (c *CSC) At(m, n int) float64 {
	return c.at(n, m)
}

func (c *CSC) T() mat64.Matrix {
	return NewCSR(c.i, c.j, c.indptr, c.ind, c.data)
}

func (c *CSC) ToDense() *mat64.Dense {
	mat := mat64.NewDense(c.j, c.i, nil)

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			mat.Set(c.ind[j], i, c.data[j])
		}
	}

	return mat
}

func (c *CSC) ToDOK() *DOK {
	dok := NewDOK(c.j, c.i)
	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			dok.Set(c.ind[j], i, c.data[j])
		}
	}

	return dok
}

func (c *CSC) ToCOO() *COO {
	cols := make([]int, c.NNZ())

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			cols[j] = i
		}
	}

	coo := NewCOO(c.j, c.i, c.ind, cols, c.data)

	return coo
}

func (c *CSC) ToCSR() *CSR {
	return c.ToCOO().ToCSR()
}

func (c *CSC) ToCSC() *CSC {
	return c
}

func (c *CSC) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(c)
}
