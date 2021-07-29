package sparse

import (
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser       = (*COO)(nil)
	_ TypeConverter = (*COO)(nil)
	_ mat.Mutable   = (*COO)(nil)
)

// COO is a COOrdinate format sparse matrix implementation (sometimes called `Triplet` format) and implements the
// Matrix interface from gonum/matrix.  This allows large sparse (mostly zero-valued) matrices to be stored
// efficiently in memory (only storing non-zero values).  COO matrices are good for constructing sparse matrices
// initially and very good at converting to CSR and CSC formats but poor for arithmetic operations.  As this
// type implements the gonum mat.Matrix interface, it may be used with any of the Gonum mat functions that
// accept Matrix types as parameters in place of other matrix types included in the Gonum mat package e.g. mat.Dense.
type COO struct {
	r    int
	c    int
	rows []int
	cols []int
	data []float64
}

// NewCOO creates a new COOrdinate format sparse matrix.
// The matrix is initialised to the size of the specified r * c dimensions (rows * columns)
// with the specified slices containing either nil or containing rows and cols indexes of non-zero elements
// and the non-zero data values themselves respectively.  If not nil, the supplied slices will be used as the
// backing storage to the matrix so changes to values of the slices will be reflected in the created matrix
// and vice versa.
func NewCOO(r int, c int, rows []int, cols []int, data []float64) *COO {
	if uint(r) < 0 {
		panic(mat.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(mat.ErrColAccess)
	}

	coo := &COO{r: r, c: c}

	if rows != nil || cols != nil || data != nil {
		if rows != nil && cols != nil && data != nil {
			coo.rows = rows
			coo.cols = cols
			coo.data = data
		} else {
			panic(mat.ErrRowAccess)
		}
	}

	return coo
}

// NNZ returns the number of stored data elements. This number includes explicit
// zeroes, if stored, and may be exceed the total number of matrix elements
// (rows * columns) if duplicate coordinates are stored.
func (c *COO) NNZ() int {
	return len(c.data)
}

// DoNonZero calls the function fn for each of the stored data elements in the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).  The order of visiting to each non-zero element is not guaranteed.
func (c *COO) DoNonZero(fn func(i, j int, v float64)) {
	nnz := c.NNZ()
	for i := 0; i < nnz; i++ {
		fn(c.rows[i], c.cols[i], c.data[i])
	}
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
		panic(mat.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.c) {
		panic(mat.ErrColAccess)
	}

	result := 0.0
	for k := 0; k < len(c.data); k++ {
		if c.rows[k] == i && c.cols[k] == j {
			// sum values for duplicate elements
			result += c.data[k]
		}
	}

	return result
}

// T transposes the matrix creating a new COO matrix, reusing the same underlying
// storage, but switching column and row sizes and index slices i.e. rows become
// columns and columns become rows.
func (c *COO) T() mat.Matrix {
	return NewCOO(c.c, c.r, c.cols, c.rows, c.data)
}

// RawMatrix converts the matrix into a CSR matrix and returns a pointer
// to the underlying blas sparse matrix.
func (c *COO) RawMatrix() *blas.SparseMatrix {
	return c.ToCSR().RawMatrix()
}

// Set sets the element of the matrix located at row i and column j to equal the
// specified value, v.  Set will panic if specified values for i or j fall outside
// the dimensions of the matrix.  Duplicate values are allowed and will be added.
func (c *COO) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(c.r) {
		panic(mat.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.c) {
		panic(mat.ErrColAccess)
	}

	c.rows = append(c.rows, i)
	c.cols = append(c.cols, j)
	c.data = append(c.data, v)
}

// ToDense returns a mat.Dense dense format version of the matrix.  The returned mat.Dense
// matrix will not share underlying storage with the receiver. nor is the receiver modified by this call
func (c *COO) ToDense() *mat.Dense {
	mat := mat.NewDense(c.r, c.c, nil)
	for i := 0; i < len(c.data); i++ {
		mat.Set(c.rows[i], c.cols[i], mat.At(c.rows[i], c.cols[i])+c.data[i])
	}

	return mat
}

// ToDOK returns a DOK (Dictionary Of Keys) sparse format version of the matrix.  The returned DOK
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *COO) ToDOK() *DOK {
	dok := NewDOK(c.r, c.c)
	for i := 0; i < len(c.data); i++ {
		dok.Set(c.rows[i], c.cols[i], dok.At(c.rows[i], c.cols[i])+c.data[i])
	}

	return dok
}

// ToCOO returns the receiver
func (c *COO) ToCOO() *COO {
	return c
}

func cumsum(p []int, c []int, n int) int {
	nz := 0
	for i := 0; i < n; i++ {
		p[i] = nz
		nz += c[i]
		c[i] = p[i]
	}
	p[n] = nz
	return nz
}

func compress(row []int, col []int, data []float64, n int) (ia []int, ja []int, d []float64) {
	//w := make([]int, n+1)
	w := getInts(n+1, true)
	defer putInts(w)
	ia = make([]int, n+1)
	ja = make([]int, len(col))
	d = make([]float64, len(data))

	for _, v := range row {
		w[v]++
	}
	cumsum(ia, w, n)

	for j, v := range col {
		p := w[row[j]]
		ja[p] = v
		d[p] = data[j]
		w[row[j]]++
	}
	return
}

func dedupe(ia []int, ja []int, d []float64, m int, n int) ([]int, []float64) {
	//w := make([]int, n)
	w := getInts(n, true)
	defer putInts(w)
	nz := 0

	for i := 0; i < m; i++ {
		q := nz
		for j := ia[i]; j < ia[i+1]; j++ {
			if w[ja[j]] > q {
				d[w[ja[j]]] += d[j]
			} else {
				w[ja[j]] = nz
				ja[nz] = ja[j]
				d[nz] = d[j]
				nz++
			}
		}
		ia[i] = q
	}
	ia[m] = nz

	return ja[:nz], d[:nz]
}

func compressInPlace(row []int, col []int, data []float64, n int) (ia []int, ja []int, d []float64) {
	//w := make([]int, n+1)
	w := getInts(n+1, true)
	defer putInts(w)
	for _, v := range row {
		w[v+1]++
	}
	for i := 0; i < n; i++ {
		w[i+1] += w[i]
	}

	var i, j int
	var ipos, iNext, jNext int
	var dt, dNext float64
	for init := 0; init < len(data); {
		dt = data[init]
		i = row[init]
		j = col[init]
		row[init] = -1
		for {
			ipos = w[i]
			dNext = data[ipos]
			iNext = row[ipos]
			jNext = col[ipos]

			data[ipos] = dt
			col[ipos] = j
			row[ipos] = -1
			w[i]++
			if iNext < 0 {
				break
			}
			dt = dNext
			i = iNext
			j = jNext
		}
		init++
		for init < len(data) && row[init] < 0 {
			init++
		}
	}

	ia = useInts(row, n+1, false)
	ia[0] = 0
	for i := 0; i < n; i++ {
		ia[i+1] = w[i]
	}
	ja = col
	d = data

	return
}

// ToCSR returns a CSR (Compressed Sparse Row)(AKA CRS (Compressed Row Storage)) sparse format
// version of the matrix.  The returned CSR matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (c *COO) ToCSR() *CSR {
	ia, ja, data := compress(c.rows, c.cols, c.data, c.r)
	ja, data = dedupe(ia, ja, data, c.r, c.c)
	return NewCSR(c.r, c.c, ia, ja, data)
}

// ToCSRReuseMem returns a CSR (Compressed Sparse Row)(AKA CRS (Compressed Row Storage)) sparse format
// version of the matrix.  Unlike with ToCSR(), The returned CSR matrix WILL share underlying storage with the
// receiver and the receiver will be modified by this call.
func (c *COO) ToCSRReuseMem() *CSR {
	ia, ja, data := compressInPlace(c.rows, c.cols, c.data, c.r)
	return NewCSR(c.r, c.c, ia, ja, data)
}

// ToCSC returns a CSC (Compressed Sparse Column)(AKA CCS (Compressed Column Storage)) sparse format
// version of the matrix.  The returned CSC matrix will not share underlying storage with the
// receiver nor is the receiver modified by this call.
func (c *COO) ToCSC() *CSC {
	ja, ia, data := compress(c.cols, c.rows, c.data, c.c)
	ia, data = dedupe(ja, ia, data, c.c, c.r)
	return NewCSC(c.r, c.c, ja, ia, data)
}

// ToCSCReuseMem returns a CSC (Compressed Sparse Column)(AKA CCS (Compressed Column Storage)) sparse format
// version of the matrix.  Unlike with ToCSC(), The returned CSC matrix WILL share underlying storage with the
// receiver and the receiver will be modified by this call.
func (c *COO) ToCSCReuseMem() *CSC {
	ja, ia, data := compressInPlace(c.cols, c.rows, c.data, c.c)
	return NewCSC(c.r, c.c, ja, ia, data)
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (c *COO) ToType(matType MatrixType) mat.Matrix {
	return matType.Convert(c)
}

// MulVecTo performs matrix vector multiplication (dst+=A*x or dst+=A^T*x), where A is
// the receiver, and stores the result in dst.  MulVecTo panics if ac != len(x) or
// ar != len(dst)
func (c *COO) MulVecTo(dst []float64, trans bool, x []float64) {
	if trans {
		if c.c != len(dst) || c.r != len(x) {
			panic(mat.ErrShape)
		}
		for i, v := range c.data {
			dst[c.cols[i]] += v * x[c.rows[i]]
		}
		return
	}

	if c.c != len(x) || c.r != len(dst) {
		panic(mat.ErrShape)
	}
	for i, v := range c.data {
		dst[c.rows[i]] += v * x[c.cols[i]]
	}
}
