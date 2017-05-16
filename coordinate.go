package sparse

import (
	"sort"

	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

type COO struct {
	r             int
	c             int
	rows          []int
	cols          []int
	data          []float64
	colMajor      bool
	canonicalised bool
}

func NewCOO(r int, c int, rows []int, cols []int, data []float64) *COO {
	if uint(r) < 0 {
		panic(matrix.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(matrix.ErrColAccess)
	}

	coo := &COO{r: r, c: c}

	if rows != nil || cols != nil || data != nil {
		if rows != nil && cols != nil && data != nil {
			coo.rows = rows
			coo.cols = cols
			coo.data = data

			coo.Canonicalise()
		} else {
			panic(matrix.ErrRowAccess)
		}
	}

	return coo
}

func (c *COO) NNZ() int {
	return len(c.data)
}

func (d *COO) Dims() (r, c int) {
	return d.r, d.c
}

func (c *COO) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(c.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.c) {
		panic(matrix.ErrColAccess)
	}

	result := 0.0
	for k := 0; k < len(c.data); k++ {
		if c.rows[k] == i && c.cols[k] == j {
			if c.canonicalised {
				return c.data[k]
			}
			// if not canonicalised then sum values for duplicate elements
			result += c.data[k]
		}
	}

	return result
}

func (c *COO) T() mat64.Matrix {
	return NewCOO(c.c, c.r, c.cols, c.rows, c.data)
}

func (c *COO) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(c.r) {
		panic(matrix.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.c) {
		panic(matrix.ErrColAccess)
	}

	c.rows = append(c.rows, i)
	c.cols = append(c.cols, j)
	c.data = append(c.data, v)
	c.canonicalised = false
}

func (c *COO) Canonicalise() {
	sort.Sort(c)

	//  Remove duplicates (summing values of duplicate elements)
	k := 0
	for i := 1; i < len(c.data); i++ {
		if (c.rows[k] != c.rows[i]) || (c.cols[k] != c.cols[i]) {
			k++
			c.rows[k], c.cols[k], c.data[k] = c.rows[i], c.cols[i], c.data[i]
		} else {
			c.data[k] += c.data[i]
		}
	}

	c.canonicalised = true
}

func (c *COO) Len() int {
	return c.NNZ()
}

func (c *COO) Less(i, j int) bool {
	if c.colMajor {
		return c.isColMajorOrdered(i, j)
	}
	return c.isRowMajorOrdered(i, j)
}

func (c *COO) Swap(i, j int) {
	c.rows[i], c.rows[j] = c.rows[j], c.rows[i]
	c.cols[i], c.cols[j] = c.cols[j], c.cols[i]
	c.data[i], c.data[j] = c.data[j], c.data[i]
}

func (c *COO) isRowMajorOrdered(i, j int) bool {
	if c.rows[i] < c.rows[j] {
		return true
	}
	if c.rows[i] == c.rows[j] {
		if c.cols[i] < c.cols[j] {
			return true
		}
	}
	return false
}

func (c *COO) isColMajorOrdered(i, j int) bool {
	if c.cols[i] < c.cols[j] {
		return true
	}
	if c.cols[i] == c.cols[j] {
		if c.rows[i] < c.rows[j] {
			return true
		}
	}
	return false
}

func (c *COO) ToDense() *mat64.Dense {
	if !c.canonicalised {
		c.Canonicalise()
	}

	mat := mat64.NewDense(c.r, c.c, nil)
	for i := 0; i < len(c.data); i++ {
		mat.Set(c.rows[i], c.cols[i], c.data[i])
	}

	return mat
}

func (c *COO) ToDOK() *DOK {
	if !c.canonicalised {
		c.Canonicalise()
	}

	dok := NewDOK(c.r, c.c)
	for i := 0; i < len(c.data); i++ {
		dok.Set(c.rows[i], c.cols[i], c.data[i])
	}

	return dok
}

func (c *COO) ToCOO() *COO {
	return c
}

func (c *COO) ToCSR() *CSR {
	if !c.canonicalised || c.colMajor {
		c.colMajor = false
		c.Canonicalise()
	}

	// build row pointers
	ia := make([]int, c.r+1)
	var j int
	k := 0
	for i := 1; i < c.r+1; i++ {
		for j = k; j < len(c.rows) && c.rows[j] < i; j++ {
			// empty
		}
		k = j
		ia[i] = j
	}

	return NewCSR(c.r, c.c, ia, c.cols, c.data)
}

func (c *COO) ToCSC() *CSC {
	if !c.canonicalised || !c.colMajor {
		c.colMajor = true
		c.Canonicalise()
	}

	// build col pointers
	ja := make([]int, c.c+1)
	var i int
	k := 0
	for j := 1; j < c.c+1; j++ {
		for i = k; i < len(c.cols) && c.cols[i] < j; i++ {
			// empty
		}
		k = i
		ja[j] = i
	}

	return NewCSC(c.r, c.c, ja, c.rows, c.data)
}

func (c *COO) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(c)
}
