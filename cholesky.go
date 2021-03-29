package sparse

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Cholesky shadows the gonum mat.Cholesly type
type Cholesky struct {
	// internal representation is CSR in lower triangular form
	chol *CSR

	// some operations use a columnar version
	cholc *CSC
	cond  float64
}

// Dims of the matrix
func (ch *Cholesky) Dims() (r, c int) {
	return ch.chol.Dims()
}

// Symmetric of matrix
func (ch *Cholesky) Symmetric() int {
	r, _ := ch.Dims()
	return r
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// At from the matrix
func (ch *Cholesky) At(i, j int) float64 {
	var val float64
	ri := ch.chol.RowView(i).(*Vector)
	rj := ch.chol.RowView(j).(*Vector)
	// FIXME: check types
	val = dotSparseSparseNoSortBefore(ri, rj, nil, min(i, j)+1)
	return val
}

// T is the same as symmetric
func (ch *Cholesky) T() mat.Matrix {
	return ch
}

func newCSR(r, c int) *CSR {
	// FIXME: creating a CSR directly leads to panics
	coo := NewCOO(r, c, nil, nil, nil)
	return coo.ToCSR()
}

func newCSC(r, c int) *CSC {
	// FIXME: creating a CSC directly leads to panics
	coo := NewCOO(r, c, nil, nil, nil)
	return coo.ToCSC()
}

// Det returns the determinant of the factored matrix
func (ch *Cholesky) Det() float64 {
	return math.Exp(ch.LogDet())
}

// LogDet returns ln(determinant) of the factored matrix
func (ch *Cholesky) LogDet() float64 {
	det := 0.0
	for i := 0; i < ch.Symmetric(); i++ {
		det += 2 * math.Log(ch.chol.At(i, i))
	}
	return det
}

// Factorize a CSR
// the CSR must be symmetric positive-definite or this won't work
// FIXME: enforce sym positive definite
func (ch *Cholesky) Factorize(a *CSR) {
	r, c := a.Dims()
	if r != c {
		panic(mat.ErrShape)
	}
	ch.chol = newCSR(r, c)
	cholCSR(a, ch.chol)
}

// LTo returns the factored matrix in lower-triangular form as a CSR
func (ch *Cholesky) LTo(dst *CSR) {
	r, c := ch.chol.Dims()
	rDst, cDst := dst.Dims()
	if r != rDst || c != cDst {
		panic(mat.ErrShape)
	}
	ch.chol.DoNonZero(func(i, j int, v float64) {
		dst.Set(i, j, v)
	})
}

func (ch *Cholesky) buildCholC() {
	if ch.cholc == nil {
		r := ch.Symmetric()
		ch.cholc = newCSC(r, r)
		ch.chol.DoNonZero(func(i, j int, v float64) {
			ch.cholc.Set(i, j, v)
		})
	}
}

// SolveVecTo shadows Cholesky.SolveVecTo
// dst is Dense as this doesn't make any sense with sparse solutions
func (ch *Cholesky) SolveVecTo(dst *mat.VecDense, b mat.Vector) error {
	r := ch.Symmetric()
	dstLen := dst.Len()
	if r != dstLen {
		panic(mat.ErrShape)
	}

	// we are going to need to scan down columns too
	ch.buildCholC()

	// textbook setup and approach:
	// Ax=b
	// LLtx=b
	// L is ch.Chol

	// forward substitute
	// Ly=b
	y := mat.NewVecDense(r, nil)
	for i := 0; i < r; i++ {
		denom := ch.chol.At(i, i)
		k := b.AtVec(i)
		sum := 0.0
		ch.chol.DoRowNonZero(i, func(x, z int, v float64) {
			if z < i {
				sum += y.AtVec(z) * v
			}
		})
		y.SetVec(i, (k-sum)/denom)
	}

	// backward substitute
	// Lt x=y
	for i := r - 1; i >= 0; i-- {
		denom := ch.chol.At(i, i)
		k := y.AtVec(i)
		sum := 0.0
		ch.cholc.DoColNonZero(i, func(x, z int, v float64) {
			if x > i {
				sum += dst.AtVec(x) * v
			}
		})
		dst.SetVec(i, (k-sum)/denom)
	}

	return nil
}

// SolveTo goes column-by-column and applies SolveVecTo
func (ch *Cholesky) SolveTo(dst *mat.Dense, b mat.Matrix) error {
	rows, cols := b.Dims()
	n := ch.Symmetric()
	if dst.IsEmpty() {
		dst.ReuseAs(n, cols)
	}
	bv, bHasColView := b.(mat.ColViewer)
	for c := 0; c < cols; c++ {
		dstView := dst.ColView(c).(*mat.VecDense)
		if bHasColView {
			cv := bv.ColView(c)
			ch.SolveVecTo(dstView, cv)
		} else {
			cv := mat.NewVecDense(rows, nil)
			ch.SolveVecTo(dstView, cv)
		}
	}
	return nil
}

// basic textbook "dot product" algo, here for comparison against the
// sparse version
func cholSimple(matrix mat.Matrix, lower *mat.TriDense) {
	r, _ := matrix.Dims()

	for i := 0; i < r; i++ {
		for j := 0; j <= i; j++ {
			var sum float64
			if i == j {
				for k := 0; k < j; k++ {
					sum += math.Pow(lower.At(j, k), 2)
				}
				lower.SetTri(j, j, math.Sqrt(matrix.At(j, j)-sum))
			} else {
				for k := 0; k < j; k++ {
					sum += lower.At(i, k) * lower.At(j, k)
				}
				lower.SetTri(i, j, (matrix.At(i, j)-sum)/lower.At(j, j))
			}
		}
	}
}

// the core sparse factoring algo
// this is simply the textbook "dot product" algo using a sparse dot
func cholCSR(matrix *CSR, lower *CSR) {
	r, _ := matrix.Dims()

	for i := 0; i < r; i++ {
		if matrix.RowNNZ(i) == 0 {
			continue
		}
		for j := 0; j <= i; j++ {
			iRow := lower.RowView(i)
			iRowS, iRowIsSparse := iRow.(*Vector)
			jRow := lower.RowView(j)
			jRowS, jRowIsSparse := jRow.(*Vector)
			if !iRowIsSparse || !jRowIsSparse {
				panic(mat.ErrShape)
			}
			if i == j {
				sum := floats.Dot(jRowS.data, jRowS.data)
				if sum == 0.0 && matrix.At(i, i) == 0.0 {
					continue
				}
				lower.Set(j, j, math.Sqrt(matrix.At(i, i)-sum))
			} else {
				rowDotSum := dotSparseSparseNoSort(iRowS, jRowS, nil)
				if rowDotSum == 0.0 && matrix.At(i, j) == 0.0 {
					continue
				}
				lower.Set(i, j, (matrix.At(i, j)-rowDotSum)/lower.At(j, j))
			}
		}
	}
}
