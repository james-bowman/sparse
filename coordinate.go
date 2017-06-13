package sparse

import (
	"sort"

	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// COO is a COOrdinate format sparse matrix implementation (sometimes called `Tiplet` format) and implements the
// Matrix interface from gonum/matrix.  This allows large sparse (mostly zero values) matrices to be stored
// efficiently in memory (only storing non-zero values).  COO matrices are good for constructing sparse matrices
// incrementally and very good at converting to CSR and CSC formats but poor for arithmetic operations.  As this
// type implements the gonum mat64.Matrix interface, it may be used with any of the Gonum mat64 functions that
// accept Matrix types as parameters in place of other matrix types included in the Gonum mat64 package e.g. mat64.Dense.
type COO struct {
	r             int
	c             int
	rows          []int
	cols          []int
	data          []float64
	colMajor      bool
	canonicalised bool
}

// NewCOO creates a new DIAgonal format sparse matrix.
// The matrix is initialised to the size of the specified r * c dimensions (rows * columns)
// with the specified slices containing either nil or containing rows and cols indexes of non-zero elements
// and the non-zero data values themselves respectively.  If not nil, the supplied slices will be used as the
// backing storage to the matrix so changes to values of the slices will be reflected in the created matrix
// and vice versa.
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

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (c *COO) NNZ() int {
	return len(c.data)
}

// Dims returns the size of the matrix as the number of rows and columns
func (c *COO) Dims() (int, int) {
	return c.r, c.c
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.  As the COO format allows duplicate elements, any
// duplicate values will be summed together.
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

// T transposes the matrix creating a new COO matrix sharing the same backing data but switching
// column and row sizes and index slices i.e. rows become columns and columns become rows.
func (c *COO) T() mat64.Matrix {
	return NewCOO(c.c, c.r, c.cols, c.rows, c.data)
}

// Set sets the element of the matrix located at row i and column j to equal the specified value, v.  Set
// will panic if specified values for i or j fall outside the dimensions of the matrix.  Duplicate values
// are allowed.
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

// Canonicalise sorts the slices (rows, cols, data) representing the NNZ values of the matrix
// (by default into row major ordering) and removes any duplicate elements (by summing them).
// The matrix is canonicalised upon initial construction and before converting to other formats
// and improves performance for operations.
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

// Len returns the length of the storage of the matrix i.e. the Number of Non Zero values.  This is required
// for sorting.
func (c *COO) Len() int {
	return c.NNZ()
}

// Less compares two items from the matrix backing storage and checks they are in the correct specified ordering
// (either row major or col major - row major is default) for sorting.
func (c *COO) Less(i, j int) bool {
	if c.colMajor {
		return c.isColMajorOrdered(i, j)
	}
	return c.isRowMajorOrdered(i, j)
}

// Swap swaps 2 row, column indexes and corresponding data values for 2 Non Zero values from the matrix for sorting.
func (c *COO) Swap(i, j int) {
	c.rows[i], c.rows[j] = c.rows[j], c.rows[i]
	c.cols[i], c.cols[j] = c.cols[j], c.cols[i]
	c.data[i], c.data[j] = c.data[j], c.data[i]
}

// isRowMajorOrdered checks the two specified elements are in row major order
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

// isColMajorOrdered checks the two specified elements are in column major order
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

// ToDense returns a mat64.Dense dense format version of the matrix.  The returned mat64.Dense
// matrix will not share underlying storage with the receiver. nor is the receiver modified by this call
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

// ToDOK returns a DOK (Dictionary Of Keys) sparse format version of the matrix.  The returned DOK
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
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

// ToCOO returns the receiver
func (c *COO) ToCOO() *COO {
	return c
}

// ToCSR returns a CSR (Compressed Sparse Row)(AKA CRS (Compressed Row Storage)) sparse format
// version of the matrix.  The returned CSR matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (c *COO) ToCSR() *CSR {
	rows := make([]int, len(c.rows))
	copy(rows, c.rows)
	cols := make([]int, len(c.cols))
	copy(cols, c.cols)
	data := make([]float64, len(c.data))
	copy(data, c.data)

	coo := &COO{c.r, c.c, rows, cols, data, c.colMajor, c.canonicalised}

	if !coo.canonicalised || coo.colMajor {
		coo.colMajor = false
		coo.Canonicalise()
	}

	// build row pointers
	ia := make([]int, c.r+1)
	var j int
	k := 0
	for i := 1; i < coo.r+1; i++ {
		for j = k; j < len(coo.rows) && coo.rows[j] < i; j++ {
			// empty
		}
		k = j
		ia[i] = j
	}

	return NewCSR(coo.r, coo.c, ia, coo.cols, coo.data)
}

// ToCSC returns a CSC (Compressed Sparse Column)(AKA CCS (Compressed Column Storage)) sparse format
// version of the matrix.  The returned CSC matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (c *COO) ToCSC() *CSC {
	rows := make([]int, len(c.rows))
	cols := make([]int, len(c.cols))
	data := make([]float64, len(c.data))

	copy(rows, c.rows)
	copy(cols, c.cols)
	copy(data, c.data)

	coo := &COO{c.r, c.c, rows, cols, data, c.colMajor, c.canonicalised}

	if !coo.canonicalised || !coo.colMajor {
		coo.colMajor = true
		coo.Canonicalise()
	}

	// build col pointers
	ja := make([]int, c.c+1)
	var i int
	k := 0
	for j := 1; j < coo.c+1; j++ {
		for i = k; i < len(coo.cols) && coo.cols[i] < j; i++ {
			// empty
		}
		k = i
		ja[j] = i
	}

	return NewCSC(coo.r, coo.c, ja, coo.rows, coo.data)
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (c *COO) ToType(matType MatrixType) mat64.Matrix {
	return matType.Convert(c)
}
