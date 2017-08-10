package sparse

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Sparser is the interface for Sparse matrices.  Sparser contains the mat.Matrix interface so automatically
// exposes all mat.Matrix methods.
type Sparser interface {
	mat.Matrix

	// NNZ returns the Number of Non Zero elements in the sparse matrix.
	NNZ() int
}

// TypeConverter interface for converting to other matrix formats
type TypeConverter interface {
	// ToDense returns a mat.Dense dense format version of the matrix.
	ToDense() *mat.Dense

	// ToDOK returns a Dictionary Of Keys (DOK) sparse format version of the matrix.
	ToDOK() *DOK

	// ToCOO returns a COOrdinate sparse format version of the matrix.
	ToCOO() *COO

	// ToCSR returns a Compressed Sparse Row (CSR) sparse format version of the matrix.
	ToCSR() *CSR

	// ToCSC returns a Compressed Sparse Row (CSR) sparse format version of the matrix.
	ToCSC() *CSC

	// ToType returns an alternative format version fo the matrix in the format specified.
	ToType(matType MatrixType) mat.Matrix
}

// MatrixType represents a type of Matrix format.  This is used to specify target format types for conversion, etc.
type MatrixType interface {
	// Convert converts to the type of matrix format represented by the receiver from the specified TypeConverter.
	Convert(from TypeConverter) mat.Matrix
}

// DenseType represents the mat.Dense matrix type format
type DenseType int

// Convert converts the specified TypeConverter to mat.Dense format
func (d DenseType) Convert(from TypeConverter) mat.Matrix {
	return from.ToDense()
}

// DOKType represents the DOK (Dictionary Of Keys) matrix type format
type DOKType int

// Convert converts the specified TypeConverter to DOK (Dictionary of Keys) format
func (s DOKType) Convert(from TypeConverter) mat.Matrix {
	return from.ToDOK()
}

// COOType represents the COOrdinate matrix type format
type COOType int

// Convert converts the specified TypeConverter to COOrdinate format
func (s COOType) Convert(from TypeConverter) mat.Matrix {
	return from.ToCOO()
}

// CSRType represents the CSR (Compressed Sparse Row) matrix type format
type CSRType int

// Convert converts the specified TypeConverter to CSR (Compressed Sparse Row) format
func (s CSRType) Convert(from TypeConverter) mat.Matrix {
	return from.ToCSR()
}

// CSCType represents the CSC (Compressed Sparse Column) matrix type format
type CSCType int

// Convert converts the specified TypeConverter to CSC (Compressed Sparse Column) format
func (s CSCType) Convert(from TypeConverter) mat.Matrix {
	return from.ToCSC()
}

const (
	// DenseFormat is an enum value representing Dense matrix format
	DenseFormat DenseType = iota

	// DOKFormat is an enum value representing DOK matrix format
	DOKFormat DOKType = iota

	// COOFormat is an enum value representing COO matrix format
	COOFormat COOType = iota

	// CSRFormat is an enum value representing CSR matrix format
	CSRFormat CSRType = iota

	// CSCFormat is an enum value representing CSC matrix format
	CSCFormat CSCType = iota
)

// Random constructs a new matrix of the specified type e.g. Dense, COO, CSR, etc.
// It is constructed with random values randomly placed through the matrix according to the
// matrix size, specified by dimensions r * c (rows * columns), and the specified density
// of non zero values.  Density is a value between 0 and 1 (0 >= density >= 1) where a density
// of 1 will construct a matrix entirely composed of non zero values and a density of 0 will
// have only zero values.
func Random(t MatrixType, r int, c int, density float32) mat.Matrix {
	d := int(density * float32(r) * float32(c))

	m := make([]int, d)
	n := make([]int, d)
	data := make([]float64, d)

	for i := 0; i < d; i++ {
		data[i] = rand.Float64()
		m[i] = rand.Intn(r)
		n[i] = rand.Intn(c)
	}

	return NewCOO(r, c, m, n, data).ToType(t)
}

// RawRowView returns a slice representing row i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func rawRowView(m mat.Matrix, i int) []float64 {
	r, c := m.Dims()

	if i >= r || i < 0 {
		panic(mat.ErrRowAccess)
	}

	slice := make([]float64, c)

	for j := range slice {
		slice[j] = m.At(i, j)
	}

	return slice
}

// RawColView returns a slice representing col i of the matrix.  This is a copy
// of the data within the matrix and does not share underlying storage.
func rawColView(m mat.Matrix, j int) []float64 {
	r, c := m.Dims()

	if j >= c || j < 0 {
		panic(mat.ErrColAccess)
	}

	slice := make([]float64, r)

	for i := range slice {
		slice[i] = m.At(i, j)
	}

	return slice
}
