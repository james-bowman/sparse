package sparse

import (
	"sort"

	"gonum.org/v1/gonum/mat"
)

var (
	csr *CSR

	_ Sparser       = csr
	_ TypeConverter = csr

	_ mat.Mutable = csr

	_ mat.ColViewer    = csr
	_ mat.RowViewer    = csr
	_ mat.RawColViewer = csr
	_ mat.RawRowViewer = csr

	_ mat.RowNonZeroDoer = csr

	csc *CSC

	_ Sparser       = csc
	_ TypeConverter = csc

	_ mat.Mutable = csc

	_ mat.ColViewer    = csc
	_ mat.RowViewer    = csc
	_ mat.RawColViewer = csc
	_ mat.RawRowViewer = csc

	_ mat.ColNonZeroDoer = csc
)

// compressedSparse represents the common structure for representing compressed sparse
// matrix formats e.g. CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column)
type compressedSparse struct {
	i, j   int
	indptr []int
	ind    []int
	data   []float64
}

// createWorkspace creates a temporary workspace to store the result
// of matrix operations avoiding the issue of mutating operands mid
// operation where they overlap with the receiver
// e.g.
//	result.Mul(result, a)
// createWorkspace will attempt to reuse previously allocated memory
// for the temporary workspace where ever possible to avoid allocating
// memory and placing strain on GC.
func (c *compressedSparse) createWorkspace(dim int, size int, zero bool) (indptr, ind []int, data []float64) {
	indptr = getInts(dim, zero)
	ind = getInts(size, zero)
	data = getFloats(size, zero)
	return
}

// commitWorkspace commits a temporary workspace previously created
// with createWorkspace.  This has the effect of updaing the receiver
// with the values from the temporary workspace and returning the
// memory used by the workspace to the pool for other operations to
// reuse.
func (c *compressedSparse) commitWorkspace(indptr, ind []int, data []float64) {
	c.indptr, indptr = indptr, c.indptr
	c.ind, ind = ind, c.ind
	c.data, data = data, c.data
	putInts(indptr)
	putInts(ind)
	putFloats(data)
}

// NNZ returns the Number of Non Zero elements in the sparse matrix.
func (c *compressedSparse) NNZ() int {
	return len(c.data)
}

// at returns the element of the matrix located at coordinate i, j.  Depending upon the
// context and the type of compressed sparse (CSR or CSC) i and j could represent rows
// and columns or columns and rows respectively.
func (c *compressedSparse) at(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(c.i) {
		panic(mat.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.j) {
		panic(mat.ErrColAccess)
	}

	// binary search through elements (assumes dimension is sorted)
	// TODO, small, low density or individual sparse row/columns
	// may be faster using linear search.  Consider selecting
	// algorithms dynamically.
	idx := c.indptr[i] + sort.SearchInts(c.ind[c.indptr[i]:c.indptr[i+1]], j)
	if idx < c.indptr[i+1] && c.ind[idx] == j {
		return c.data[idx]
	}

	// for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
	// 	if c.ind[k] == j {
	// 		return c.data[k]
	// 	}
	// }

	return 0
}

// set is a generic compressed sparse method to set a matrix element for both CSR and CSC
// matrices
func (c *compressedSparse) set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(c.i) {
		panic(mat.ErrRowAccess)
	}
	if uint(j) < 0 || uint(j) >= uint(c.j) {
		panic(mat.ErrColAccess)
	}

	if v == 0 {
		// don't bother storing zero values
		return
	}

	// TODO switch to binary search
	for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
		if c.ind[k] == j {
			// if element(i, j) is already a non-zero value then simply update the existing
			// value without altering the sparsity pattern
			c.data[k] = v
			return
		}

		if c.ind[k] > j {
			// element(i, j) doesn't exist in current sparsity pattern and is mid row/col
			// so add it
			c.insert(i, j, v, k)
			return
		}
	}

	// element(i, j) doesn't exist in current sparsity pattern and is beyond the last
	// non-zero element of a row/col or an empty row/col - so add it
	c.insert(i, j, v, c.indptr[i+1])
}

// insert inserts a new non-zero element into the sparse matrix, updating the
// sparsity pattern
func (c *compressedSparse) insert(i int, j int, v float64, insertionPoint int) {
	c.ind = append(c.ind, 0)
	copy(c.ind[insertionPoint+1:], c.ind[insertionPoint:])
	c.ind[insertionPoint] = j

	c.data = append(c.data, 0)
	copy(c.data[insertionPoint+1:], c.data[insertionPoint:])
	c.data[insertionPoint] = v

	for n := i + 1; n <= c.i; n++ {
		c.indptr[n]++
	}
}

// nativeSlice slices the compressed sparse matrix along its primary axis
// i.e. row slice for CSR and column slice for CSC.  This is much more
// efficient than a foreign slice which must scan for each slice element.
func (c *compressedSparse) nativeSlice(i int) []float64 {
	if i >= c.i || i < 0 {
		panic(mat.ErrRowAccess)
	}

	slice := make([]float64, c.j)

	for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
		slice[c.ind[j]] = c.data[j]
	}

	return slice
}

// foreignSlice slices the compressed sparse matrix along its secondary
// axis i.e. column slice for CSR and row slice for CSC.  This is much
// less efficient than a native slice as each element must be looked up
// individually.  Consider converting the matrix (CSR->CSR/CSC->CSR) if
// multiple slices along secondary axis are required so native slices
// can be used instead.
func (c *compressedSparse) foreignSlice(j int) []float64 {
	if j >= c.j || j < 0 {
		panic(mat.ErrColAccess)
	}

	slice := make([]float64, c.i)

	for i := range slice {
		slice[i] = c.at(i, j)
	}

	return slice
}

// CSR is a Compressed Sparse Row format sparse matrix implementation (sometimes called Compressed Row
// Storage (CRS) format) and implements the Matrix interface from gonum/matrix.  This allows large sparse
// (mostly zero values) matrices to be stored efficiently in memory (only storing non-zero values).
// CSR matrices are poor for constructing sparse matrices incrementally but very good for arithmetic operations.
// CSR, and their sibling CSC, matrices are similar to COOrdinate matrices except the row index slice is
// compressed.  Rather than storing the row indices of each non zero values (length == NNZ) each element, i,
// of the slice contains the cumulative count of non zero values in the matrix up to row i-1 of the matrix.
// In this way, it is possible to address any element, i j, in the matrix with the following:
//
// 		for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
//			if c.ind[k] == j {
//				return c.data[k]
//			}
//		}
//
// It should be clear that CSR is like CSC except the slices are row major order rather than column major and
// CSC is essentially the transpose of a CSR.
// As this type implements the gonum mat.Matrix interface, it may be used with any of the Gonum mat
// functions that accept Matrix types as parameters in place of other matrix types included in the Gonum
// mat package e.g. mat.Dense.
type CSR struct {
	compressedSparse
}

// NewCSR creates a new Compressed Sparse Row format sparse matrix.
// The matrix is initialised to the size of the specified r * c dimensions (rows * columns)
// with the specified slices containing row pointers and cols indexes of non-zero elements
// and the non-zero data values themselves respectively.  The supplied slices will be used as the
// backing storage to the matrix so changes to values of the slices will be reflected in the created matrix
// and vice versa.
func NewCSR(r int, c int, ia []int, ja []int, data []float64) *CSR {
	if uint(r) < 0 {
		panic(mat.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(mat.ErrColAccess)
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

// Dims returns the size of the matrix as the number of rows and columns
func (c *CSR) Dims() (int, int) {
	return c.i, c.j
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.
func (c *CSR) At(m, n int) float64 {
	return c.at(m, n)
}

// Set sets the element of the matrix located at row i and column j to value v.  Set will panic if
// specified values for i or j fall outside the dimensions of the matrix.
func (c *CSR) Set(m, n int, v float64) {
	c.set(m, n, v)
}

// T transposes the matrix creating a new CSC matrix sharing the same backing data storage but switching
// column and row sizes and index & index pointer slices i.e. rows become columns and columns become rows.
func (c *CSR) T() mat.Matrix {
	return NewCSC(c.j, c.i, c.indptr, c.ind, c.data)
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).  The order of visiting to each non-zero element is row major.
func (c *CSR) DoNonZero(fn func(i, j int, v float64)) {
	for i := 0; i < len(c.indptr)-1; i++ {
		c.DoRowNonZero(i, fn)
	}
}

// DoRowNonZero calls the function fn for each of the non-zero elements of row i
// in the receiver.  The function fn takes a row/column index and the element value
// of the receiver at (i, j).
func (c *CSR) DoRowNonZero(i int, fn func(i, j int, v float64)) {
	for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
		fn(i, c.ind[j], c.data[j])
	}
}

// Clone copies the specified matrix into the receiver
func (c *CSR) Clone(b mat.Matrix) {
	c.i, c.j = b.Dims()

	c.indptr = make([]int, c.i+1)

	k := 0
	for i := 0; i < c.i; i++ {
		c.indptr[i] = k
		for j := 0; j < c.j; j++ {
			if v := b.At(i, j); v != 0 {
				c.ind = append(c.ind, j)
				c.data = append(c.data, v)
				k++
			}
		}

	}
	c.indptr[c.i] = k
}

// ToDense returns a mat.Dense dense format version of the matrix.  The returned mat.Dense
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSR) ToDense() *mat.Dense {
	mat := mat.NewDense(c.i, c.j, nil)

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			mat.Set(i, c.ind[j], c.data[j])
		}
	}

	return mat
}

// ToDOK returns a DOK (Dictionary Of Keys) sparse format version of the matrix.  The returned DOK
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSR) ToDOK() *DOK {
	dok := NewDOK(c.i, c.j)
	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			dok.Set(i, c.ind[j], c.data[j])
		}
	}

	return dok
}

// ToCOO returns a COOrdinate sparse format version of the matrix.  The returned COO matrix will
// not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSR) ToCOO() *COO {
	rows := make([]int, c.NNZ())
	cols := make([]int, c.NNZ())
	data := make([]float64, c.NNZ())

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			rows[j] = i
			cols[j] = c.ind[j]
			data[j] = c.data[j]
		}
	}

	coo := NewCOO(c.i, c.j, rows, cols, data)

	return coo
}

// ToCSR returns the receiver
func (c *CSR) ToCSR() *CSR {
	return c
}

// ToCSC returns a Compressed Sparse Column sparse format version of the matrix.  The returned CSC matrix
// will not share underlying storage with the receiver nor is the receiver modified by this call.
// NB, the current implementation uses COO as an intermediate format so converts to COO before converting
// to CSC but attempts to reuse memory in the intermediate formats.
func (c *CSR) ToCSC() *CSC {
	return c.ToCOO().ToCSCReuseMem()
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (c *CSR) ToType(matType MatrixType) mat.Matrix {
	return matType.Convert(c)
}

// RowNNZ returns the Number of Non Zero values in the specified row i.  RowNNZ will panic if i is out of range.
func (c *CSR) RowNNZ(i int) int {
	if uint(i) < 0 || uint(i) >= uint(c.i) {
		panic(mat.ErrRowAccess)
	}
	return c.indptr[i+1] - c.indptr[i]
}

// RowView slices the Compressed Sparse Row matrix along its primary axis.
// Returns a VecCOO sparse Vector that shares the same storage with
// the receiver for row i.
func (c *CSR) RowView(i int) mat.Vector {
	//return mat.NewVecDense(c.j, c.nativeSlice(i))
	return NewVecCOO(c.j, c.ind[c.indptr[i]:c.indptr[i+1]], c.data[c.indptr[i]:c.indptr[i+1]])
}

// ColView slices the Compressed Sparse Row matrix along its secondary axis.
// Returns a VecDense dense Vector containing a copy of elements of column j.
func (c *CSR) ColView(j int) mat.Vector {
	return mat.NewVecDense(c.i, c.foreignSlice(j))
}

// RawRowView returns a slice representing row i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func (c *CSR) RawRowView(i int) []float64 {
	return c.nativeSlice(i)
}

// RawColView returns a slice representing col i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.  RawColView
// is much slower than RawRowView for CSR matrices so if multiple RawColView calls
// are required, consider first converting to a CSC matrix.
func (c *CSR) RawColView(j int) []float64 {
	return c.foreignSlice(j)
}

// CSC is a Compressed Sparse Column format sparse matrix implementation (sometimes called Compressed Column
// Storage (CCS) format) and implements the Matrix interface from gonum/matrix.  This allows large sparse
// (mostly zero values) matrices to be stored efficiently in memory (only storing non-zero values).
// CSC matrices are poor for constructing sparse matrices incrementally but very good for arithmetic operations.
// CSC, and their sibling CSR, matrices are similar to COOrdinate matrices except the column index slice is
// compressed.  Rather than storing the column indices of each non zero values (length == NNZ) each element, i,
// of the slice contains the cumulative count of non zero values in the matrix up to column i-1 of the matrix.
// In this way, it is possible to address any element, j i, in the matrix with the following:
//
// 		for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
//			if c.ind[k] == j {
//				return c.data[k]
//			}
//		}
//
// It should be clear that CSC is like CSR except the slices are column major order rather than row major and CSC
// is essentially the transpose of a CSR.
// As this type implements the gonum mat.Matrix interface, it may be used with any of the Gonum mat functions
// that accept Matrix types as parameters in place of other matrix types included in the Gonum mat package
// e.g. mat.Dense.
type CSC struct {
	compressedSparse
}

// NewCSC creates a new Compressed Sparse Column format sparse matrix.
// The matrix is initialised to the size of the specified r * c dimensions (rows * columns)
// with the specified slices containing column pointers and row indexes of non-zero elements
// and the non-zero data values themselves respectively.  The supplied slices will be used as the
// backing storage to the matrix so changes to values of the slices will be reflected in the created matrix
// and vice versa.
func NewCSC(r int, c int, indptr []int, ind []int, data []float64) *CSC {
	if uint(r) < 0 {
		panic(mat.ErrRowAccess)
	}
	if uint(c) < 0 {
		panic(mat.ErrColAccess)
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

// Dims returns the size of the matrix as the number of rows and columns
func (c *CSC) Dims() (int, int) {
	return c.j, c.i
}

// At returns the element of the matrix located at row i and column j.  At will panic if specified values
// for i or j fall outside the dimensions of the matrix.
func (c *CSC) At(m, n int) float64 {
	return c.at(n, m)
}

// Set sets the element of the matrix located at row i and column j to value v.  Set will panic if
// specified values for i or j fall outside the dimensions of the matrix.
func (c *CSC) Set(m, n int, v float64) {
	c.set(n, m, v)
}

// T transposes the matrix creating a new CSR matrix sharing the same backing data storage but switching
// column and row sizes and index & index pointer slices i.e. rows become columns and columns become rows.
func (c *CSC) T() mat.Matrix {
	return NewCSR(c.i, c.j, c.indptr, c.ind, c.data)
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).  The order of visiting to each non-zero element is column major.
func (c *CSC) DoNonZero(fn func(i, j int, v float64)) {
	for i := 0; i < len(c.indptr)-1; i++ {
		c.DoColNonZero(i, fn)
	}
}

// DoColNonZero calls the function fn for each of the non-zero elements of column j
// in the receiver.  The function fn takes a row/column index and the element value
// of the receiver at (i, j).
func (c *CSC) DoColNonZero(j int, fn func(i, j int, v float64)) {
	for i := c.indptr[j]; i < c.indptr[j+1]; i++ {
		fn(c.ind[i], j, c.data[i])
	}
}

// ToDense returns a mat.Dense dense format version of the matrix.  The returned mat.Dense
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSC) ToDense() *mat.Dense {
	mat := mat.NewDense(c.j, c.i, nil)

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			mat.Set(c.ind[j], i, c.data[j])
		}
	}

	return mat
}

// ToDOK returns a DOK (Dictionary Of Keys) sparse format version of the matrix.  The returned DOK
// matrix will not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSC) ToDOK() *DOK {
	dok := NewDOK(c.j, c.i)
	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			dok.Set(c.ind[j], i, c.data[j])
		}
	}

	return dok
}

// ToCOO returns a COOrdinate sparse format version of the matrix.  The returned COO matrix will
// not share underlying storage with the receiver nor is the receiver modified by this call.
func (c *CSC) ToCOO() *COO {
	rows := make([]int, c.NNZ())
	cols := make([]int, c.NNZ())
	data := make([]float64, c.NNZ())

	for i := 0; i < len(c.indptr)-1; i++ {
		for j := c.indptr[i]; j < c.indptr[i+1]; j++ {
			cols[j] = i
			rows[j] = c.ind[j]
			data[j] = c.data[j]
		}
	}

	coo := NewCOO(c.j, c.i, rows, cols, data)

	return coo
}

// ToCSR returns a Compressed Sparse Row sparse format version of the matrix.  The returned CSR matrix
// will not share underlying storage with the receiver nor is the receiver modified by this call.
// NB, the current implementation uses COO as an intermediate format so converts to COO before converting
// to CSR but attempts to reuse memory in the intermediate formats.
func (c *CSC) ToCSR() *CSR {
	return c.ToCOO().ToCSRReuseMem()
}

// ToCSC returns the receiver
func (c *CSC) ToCSC() *CSC {
	return c
}

// ToType returns an alternative format version fo the matrix in the format specified.
func (c *CSC) ToType(matType MatrixType) mat.Matrix {
	return matType.Convert(c)
}

// RowView slices the Compressed Sparse Column matrix along its secondary axis.
// Returns a VecDense dense Vector containing a copy of elements of row i.
func (c *CSC) RowView(i int) mat.Vector {
	return mat.NewVecDense(c.i, c.foreignSlice(i))
}

// ColView slices the Compressed Sparse Column matrix along its primary axis.
// Returns a VecCOO sparse Vector that shares the same underlying storage as
// column i of the receiver.
func (c *CSC) ColView(j int) mat.Vector {
	//return mat.NewVecDense(c.j, c.nativeSlice(j))
	return NewVecCOO(c.j, c.ind[c.indptr[j]:c.indptr[j+1]], c.data[c.indptr[j]:c.indptr[j+1]])
}

// RawRowView returns a slice representing row i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.  RawRowView
// is much slower than RawColView for CSC matrices so if multiple RawRowView calls
// are required, consider first converting to a CSR matrix.
func (c *CSC) RawRowView(i int) []float64 {
	return c.foreignSlice(i)
}

// RawColView returns a slice representing col i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func (c *CSC) RawColView(j int) []float64 {
	return c.nativeSlice(j)
}
