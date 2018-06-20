package sparse

import (
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

var (
	coo *COO

	_ Sparser       = coo
	_ TypeConverter = coo

	_ mat.Mutable = coo

	_ mat.ColViewer = coo
	_ mat.RowViewer = coo
)

// COO is a COOrdinate format sparse matrix implementation (sometimes called `Tiplet` format) and implements the
// Matrix interface from gonum/matrix.  This allows large sparse (mostly zero values) matrices to be stored
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

// NewCOO creates a new DIAgonal format sparse matrix.
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

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (c *COO) NNZ() int {
	return len(c.data)
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
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
	w := make([]int, n+1)
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
	w := make([]int, n)
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
	w := make([]int, n+1)

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

	if n+1 > cap(row) {
		ia = make([]int, n+1)
	} else {
		ia = row[:n+1]
	}
	for i := 0; i < n; i++ {
		ia[i+1] = w[i]
	}
	ia[0] = 0
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

// RowView slices the matrix and returns a Vector containing a copy of elements
// of row i.
func (c *COO) RowView(i int) mat.Vector {
	return mat.NewVecDense(c.c, mat.Row(nil, i, c))
}

// ColView slices the matrix and returns a Vector containing a copy of elements
// of column i.
func (c *COO) ColView(j int) mat.Vector {
	return mat.NewVecDense(c.r, mat.Col(nil, j, c))
}
