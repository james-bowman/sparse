package sparse

import (
	"math/rand"

	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

// Sparser is the interface for Sparse matrices.  Sparser contains the mat.Matrix interface so automatically
// exposes all mat.Matrix methods.
type Sparser interface {
	mat.Matrix
	mat.NonZeroDoer

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

// alias reports whether x and y share the same base array.
func aliasFloats(x, y []float64) bool {
	return cap(x) > 0 && cap(y) > 0 && &x[0:cap(x)][cap(x)-1] == &y[0:cap(y)][cap(y)-1]
}

// alias reports whether x and y share the same base array.
func aliasInts(x, y []int) bool {
	return cap(x) > 0 && cap(y) > 0 && &x[0:cap(x)][cap(x)-1] == &y[0:cap(y)][cap(y)-1]
}

// useFloats attempts to reuse the specified slice of floats ensuring it has
// sufficient capacity for at least n elements.  If slice does not have sufficient
// capacity for n elements, new storage will be allocated.  If clear is true,
// all values in the slice will be zeroed.
func useFloats(slice []float64, n int, clear bool) []float64 {
	if n <= cap(slice) {
		slice = slice[:n]
		if clear {
			for i := range slice {
				slice[i] = 0
			}
		}
		return slice
	}
	return make([]float64, n)
}

// useInts attempts to reuse the specified slice of ints ensuring it has
// sufficient capacity for at least n elements.  If slice does not have sufficient
// capacity for n elements, new storage will be allocated.  If clear is true,
// all values in the slice will be zeroed.
func useInts(slice []int, n int, clear bool) []int {
	if n <= cap(slice) {
		slice = slice[:n]
		if clear {
			for i := range slice {
				slice[i] = 0
			}
		}
		return slice
	}
	return make([]int, n)
}

// Normer is an interface for calculating the Norm of a matrix.
// This allows matrices to implement format specific Norm
// implementations optimised for each format processing only non-zero
// elements for different sparsity patterns across sparse matrix formats.
type Normer interface {
	Norm(L float64) float64
}

// Norm returns the norm of the matrix as a scalar value.  This
// implementation is able to take advantage of sparse matrix types
// and only process non-zero values providing the supplied matrix
// implements the Normer interface.  If the supplied matrix does
// not implement Normer then the function will invoke mat.Norm()
// to process the matrix.
func Norm(m mat.Matrix, L float64) float64 {
	if n, isNormer := m.(Normer); isNormer {
		return n.Norm(L)
	}

	return mat.Norm(m, L)
}

// BlasCompatibleSparser is an interface which represents Sparse matrices compatible with
// sparse BLAS routines i.e. implementing the RawMatrix() method as a means of obtaining
// a BLAS sparse matrix representation of the matrix.
type BlasCompatibleSparser interface {
	Sparser
	RawMatrix() *blas.SparseMatrix
}

// MulMatVec (y = alpha * a * x + y) performs sparse matrix multiplication with a vector and
// stores the result in a mat.VecDense vector.  y is a *mat.VecDense, if c is nil, a new mat.VecDense
// of the correct dimensions (Ac x 1) will be allocated and returned as the result of the function.
// x is an implementation of mat.Vector and a is a sparse matri of type CSR, CSC or a format
// that implements the BlasCompatibleSparser interface.  Matrix A will be scaled by alpha.
// If transA is true, the matrix A will be transposed as part of the operation.  The function
// will panic Ac != len(x) or if (y != nil and (Ac != len(y)))
func MulMatVec(transA bool, alpha float64, a BlasCompatibleSparser, x mat.Vector, y *mat.VecDense) *mat.VecDense {
	// A is m x n (or n x m if transA), x is n, y is m
	ar, ac := a.Dims()
	if transA {
		ar, ac = ac, ar
	}
	if ac != x.Len() {
		panic(mat.ErrShape)
	}
	if y == nil {
		y = mat.NewVecDense(ar, nil)
	} else {
		if ar != y.Len() {
			panic(mat.ErrShape)
		}
	}

	yraw := y.RawVector()

	var araw *blas.SparseMatrix
	if as, ok := a.(*CSC); ok {
		// as CSC is the natural transpose of CSR, we will transpose here to CSR
		// then transpose back during the multiplication operation
		araw = as.T().(*CSR).RawMatrix()
		transA = !transA
	} else {
		araw = a.RawMatrix()
	}

	// xd, xIsDense := x.(*mat.VecDense)
	xd, xIsDense := x.(mat.RawVectorer)
	if !xIsDense {
		if xs, xIsSparse := x.(*Vector); xIsSparse {
			xd = xs.ToDense()
		} else {
			xd = mat.VecDenseCopyOf(x)
		}
	}
	xraw := xd.RawVector()
	blas.Dusmv(transA, alpha, araw, xraw.Data, xraw.Inc, yraw.Data, yraw.Inc)
	return y

}

// MulMatMat (c = alpha * a * b + c) performs sparse matrix multiplication with another matrix and
// stores the result in a mat.Dense matrix.  c is a *mat.Dense, if c is nil, a new mat.Dense
// of the correct dimensions (Ar x Bc) will be allocated and returned as the result from the
// function. b is an implementation of mat.Matrix and a is a sparse matrix of type CSR, CSC or
// a format that implements the BlasCompatibleSparser interface.  Matrix A
// will be scaled by alpha.  If transA is true, the matrix A will be transposed as part of the
// operation.  The function will panic if Ac != Br or if (C != nil and (ar != Cr or Bc != Cc))
func MulMatMat(transA bool, alpha float64, a BlasCompatibleSparser, b mat.Matrix, c *mat.Dense) *mat.Dense {
	// A is m x n (or n x m if transA), B is n x k, C is m x k
	ar, ac := a.Dims()
	if transA {
		ar, ac = ac, ar
	}
	br, bc := b.Dims()

	if ac != br {
		panic(mat.ErrShape)
	}
	if c == nil {
		c = mat.NewDense(ar, bc, nil)
	} else {
		cr, cc := c.Dims()
		if ar != cr || bc != cc {
			panic(mat.ErrShape)
		}
	}
	craw := c.RawMatrix()

	var araw *blas.SparseMatrix
	if as, ok := a.(*CSC); ok {
		// as CSC is the natural transpose of CSR, we will transpose here to CSR
		// then transpose back during the multiplication operation
		araw = as.T().(*CSR).RawMatrix()
		transA = !transA
	} else {
		araw = a.RawMatrix()
	}

	if bd, bIsDense := b.(mat.RawMatrixer); bIsDense {
		braw := bd.RawMatrix()
		blas.Dusmm(transA, bc, alpha, araw, braw.Data, braw.Stride, craw.Data, craw.Stride)
		return c
	}

	if bs, bIsCSC := b.(*CSC); bIsCSC {
		col := getFloats(br, true)
		for j := 0; j < bc; j++ {
			begin, end := bs.matrix.Indptr[j], bs.matrix.Indptr[j+1]
			ind := bs.matrix.Ind[begin:end]
			blas.Dussc(bs.matrix.Data[begin:end], col, 1, ind)
			blas.Dusmv(transA, alpha, araw, col, 1, craw.Data[j:], craw.Stride)
			for _, v := range ind {
				col[v] = 0
			}
		}
		putFloats(col)
		return c
	}

	if bs, bIsCSR := b.(*CSR); bIsCSR {
		if transA {
			for i := 0; i < ac; i++ {
				begin, end := bs.matrix.Indptr[i], bs.matrix.Indptr[i+1]
				for t := araw.Indptr[i]; t < araw.Indptr[i+1]; t++ {
					blas.Dusaxpy(alpha*araw.Data[t], bs.matrix.Data[begin:end], bs.matrix.Ind[begin:end], craw.Data[araw.Ind[t]*craw.Stride:(araw.Ind[t]+1)*craw.Stride], 1)
				}
			}
		} else {
			for i := 0; i < ar; i++ {
				for t := araw.Indptr[i]; t < araw.Indptr[i+1]; t++ {
					begin, end := bs.matrix.Indptr[araw.Ind[t]], bs.matrix.Indptr[araw.Ind[t]+1]
					blas.Dusaxpy(alpha*araw.Data[t], bs.matrix.Data[begin:end], bs.matrix.Ind[begin:end], craw.Data[i*craw.Stride:(i+1)*craw.Stride], 1)
				}
			}
		}
		return c
	}

	col := getFloats(br, false)
	for j := 0; j < bc; j++ {
		col = mat.Col(col, j, b)
		blas.Dusmv(transA, alpha, araw, col, 1, craw.Data[j:], craw.Stride)
	}
	putFloats(col)
	return c
}
