package sparse

import (
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

// createWorkspace creates a temporary workspace to store the result
// of matrix operations avoiding the issue of mutating operands mid
// operation where they overlap with the receiver
// e.g.
//	result.Mul(result, a)
// createWorkspace will attempt to reuse previously allocated memory
// for the temporary workspace where ever possible to avoid allocating
// memory and placing strain on GC.
func (c *CSR) createWorkspace(dim int, size int, zero bool) (indptr, ind []int, data []float64) {
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
func (c *CSR) commitWorkspace(indptr, ind []int, data []float64) {
	c.matrix.Indptr, indptr = indptr, c.matrix.Indptr
	c.matrix.Ind, ind = ind, c.matrix.Ind
	c.matrix.Data, data = data, c.matrix.Data
	putInts(indptr)
	putInts(ind)
	putFloats(data)
}

// Mul takes the matrix product of the supplied matrices a and b and stores the result
// in the receiver.  Some specific optimisations are available for operands of certain
// sparse formats e.g. CSR * CSR uses Gustavson Algorithm (ACM 1978) for fast
// sparse matrix multiplication.
// If the number of columns does not equal the number of rows in b, Mul will panic.
func (c *CSR) Mul(a, b mat.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(mat.ErrShape)
	}

	if dia, ok := a.(*DIA); ok {
		// handle case where matrix A is a DIA
		c.mulDIA(dia, b, false)
		return
	}
	if dia, ok := b.(*DIA); ok {
		// handle case where matrix B is a DIA
		c.mulDIA(dia, a, true)
		return
	}
	// TODO: handle cases where both matrices are DIA

	lhs, isCsr := a.(*CSR)
	if isCsr {
		if rhs, isRCsr := b.(*CSR); isRCsr {
			// handle case where matrix A is CSR and matrix B is CSR
			c.mulCSRCSR(lhs, rhs)
			return
		}
		if rhs, isCsc := b.(*CSC); isCsc {
			// handle case where matrix A is CSR and matrix B is CSC
			c.mulCSRCSC(lhs, rhs)
			return
		}
	}

	indptr, ind, data := c.createWorkspace(ar+1, 0, false)
	t := 0

	if isCsr {
		// handle case where matrix A is CSR (matrix B can be any implementation of mat.Matrix)
		for i := 0; i < ar; i++ {
			indptr[i] = t
			for j := 0; j < bc; j++ {
				var v float64
				// TODO Consider converting all LHS Sparser args to CSR
				// TODO Consider converting all RHS Sparser args to CSC
				for k := lhs.matrix.Indptr[i]; k < lhs.matrix.Indptr[i+1]; k++ {
					v += lhs.matrix.Data[k] * b.At(lhs.matrix.Ind[k], j)
				}
				if v != 0 {
					t++
					ind = append(ind, j)
					data = append(data, v)
				}
			}
		}
	} else {
		// handle any implementation of mat.Matrix for both matrix A and B
		row := getFloats(ac, false)
		for i := 0; i < ar; i++ {
			indptr[i] = t
			// perhaps counter-intuatively, transferring the row elements of the first operand
			// into a slice as part of a separate loop (rather than accessing each element within
			// the main loop (a.At(m, n) * b.At(m, n)) then ranging over them as part of the
			// main loop is about twice as fast.  This is related to lining the data up into CPU
			// cache rather than accessing from RAM.
			// This seems to have interesting implications when using formats with more expensive
			// lookups - placing the more costly format first (and extracting its rows into a
			// slice) appears approximately twice as fast as switching the order of the formats.
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
					ind = append(ind, j)
					data = append(data, v)
				}
			}
		}
		putFloats(row)
	}

	indptr[ar] = t
	c.matrix.I, c.matrix.J = ar, bc
	c.commitWorkspace(indptr, ind, data)
}

// MulMatRawVec computes the matrix vector product between lhs and rhs and stores
// the result in out
func MulMatRawVec(lhs *CSR, rhs []float64, out []float64) {
	m, n := lhs.Dims()
	if len(rhs) != n {
		panic(mat.ErrShape)
	}
	if len(out) != m {
		panic(mat.ErrShape)
	}

	blas.Usmv(false, 1, lhs.RawMatrix(), rhs, 1, out, 1)
}

// mulCSRCSR handles CSR = CSR * CSR using Gustavson Algorithm (ACM 1978)
func (c *CSR) mulCSRCSR(lhs *CSR, rhs *CSR) {
	ar, _ := lhs.Dims()
	_, bc := rhs.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, false)

	x := make([]float64, bc)

	// rows in C
	for i := 0; i < ar; i++ {
		// each element t in row i of A
		for t := lhs.matrix.Indptr[i]; t < lhs.matrix.Indptr[i+1]; t++ {
			// each element j in row [A.Ind[t]] of B
			for j := rhs.matrix.Indptr[lhs.matrix.Ind[t]]; j < rhs.matrix.Indptr[lhs.matrix.Ind[t]+1]; j++ {
				x[rhs.matrix.Ind[j]] += lhs.matrix.Data[t] * rhs.matrix.Data[j]
			}
		}
		for j, v := range x {
			if v != 0 {
				ind = append(ind, j)
				data = append(data, v)
				x[j] = 0
			}
		}
		indptr[i+1] = len(ind)
	}
	c.matrix.I, c.matrix.J = ar, bc
	c.commitWorkspace(indptr, ind, data)
}

// mulCSRCSC handles special case of matrix multiplication where the LHS matrix
// (A) is CSR format and the RHS matrix (B) is CSC format.
func (c *CSR) mulCSRCSC(lhs *CSR, rhs *CSC) {
	ar, _ := lhs.Dims()
	_, bc := rhs.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, false)

	t := 0
	for i := 0; i < ar; i++ {
		indptr[i] = t
		for j := 0; j < bc; j++ {
			var v float64
			rhsStart := rhs.matrix.Indptr[j]
			rhsEnd := rhs.matrix.Indptr[j+1] - 1
			b := rhsStart

			for k := lhs.matrix.Indptr[i]; k < lhs.matrix.Indptr[i+1]; k++ {
				var bi int
				for bi = b; bi < rhsEnd && rhs.matrix.Ind[bi] < lhs.matrix.Ind[k]; bi++ {
					// empty
				}
				b = bi
				if lhs.matrix.Ind[k] == rhs.matrix.Ind[bi] {
					v += lhs.matrix.Data[k] * rhs.matrix.Data[bi]
				}
			}
			if v != 0 {
				t++
				ind = append(ind, j)
				data = append(data, v)
			}
		}
	}
	indptr[ar] = t
	c.matrix.I, c.matrix.J = ar, bc
	c.commitWorkspace(indptr, ind, data)
}

// mulDIA takes the matrix product of the diagonal matrix dia and an other matrix, other and stores the result
// in the receiver.  This method caters for the specialised case of multiplying by a diagonal matrix where
// significant optimisation is possible due to the sparsity pattern of the matrix.  If trans is true, the method
// will assume that other was the LHS (Left Hand Side) operand and that dia was the RHS.
func (c *CSR) mulDIA(dia *DIA, other mat.Matrix, trans bool) {
	var csMat blas.SparseMatrix
	isCS := false
	var size int

	if csr, ok := other.(*CSR); ok {
		// TODO consider implicitly converting all sparsers to CSR
		// or at least iterating only over the non-zero elements
		csMat = csr.matrix
		isCS = true
		size = len(csMat.Ind)
	}

	rows, cols := other.Dims()
	indptr, ind, data := c.createWorkspace(rows+1, size, true)

	t := 0
	raw := dia.Diagonal()

	for i := 0; i < rows; i++ {
		indptr[i] = t
		var v float64

		if isCS {
			for k := csMat.Indptr[i]; k < csMat.Indptr[i+1]; k++ {
				var rawval float64
				if trans {
					rawval = raw[csMat.Ind[k]]
				} else {
					rawval = raw[i]
				}
				v = csMat.Data[k] * rawval
				if v != 0 {
					ind[t] = csMat.Ind[k]
					data[t] = v
					t++
				}
			}
		} else {
			for k := 0; k < cols; k++ {
				var rawval float64
				if trans {
					rawval = raw[k]
				} else {
					rawval = raw[i]
				}
				v = other.At(i, k) * rawval
				if v != 0 {
					ind = append(ind, k)
					data = append(data, v)
					t++
				}
			}
		}
	}
	indptr[rows] = t

	c.matrix.I, c.matrix.J = rows, cols
	c.commitWorkspace(indptr, ind, data)
}

// Add adds matrices a and b together and stores the result in the receiver.
// If matrices a and b are not the same shape then the method will panic.
func (c *CSR) Add(a, b mat.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(mat.ErrShape)
	}

	lCsr, lIsCsr := a.(*CSR)
	rCsr, rIsCsr := b.(*CSR)

	// TODO optimisation for DIA matrices
	if lIsCsr && rIsCsr {
		c.addCSRCSR(lCsr, rCsr)
		return
	} else if lIsCsr {
		c.addCSR(lCsr, b)
		return
	} else if rIsCsr {
		c.addCSR(rCsr, a)
		return
	}
	// dumb addition with no sparcity optimisations/savings
	indptr, ind, data := c.createWorkspace(0, 0, false)
	var offset int
	indptr = append(indptr, offset)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			v := a.At(i, j) + b.At(i, j)
			if v != 0 {
				ind = append(ind, j)
				data = append(data, v)
				offset++
			}
		}
		indptr = append(indptr, offset)
	}
	c.matrix.I, c.matrix.J = ar, ac
	c.commitWorkspace(indptr, ind, data)
}

// addCSR adds a CSR matrix to any implementation of mat.Matrix and stores the
// result in the receiver.
func (c *CSR) addCSR(csr *CSR, other mat.Matrix) {
	ar, ac := csr.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, false)

	row := getFloats(ac, false)
	a := csr.RawMatrix()

	for i := 0; i < ar; i++ {
		begin := csr.matrix.Indptr[i]
		end := csr.matrix.Indptr[i+1]

		if bDense, isDense := other.(*mat.Dense); isDense {
			row = bDense.RawRowView(i)
			blas.Usaxpy(1, a.Data[begin:end], a.Ind[begin:end], row, 1)
		} else {
			row = mat.Row(row, i, other)
			blas.Usaxpy(1, a.Data[begin:end], a.Ind[begin:end], row, 1)
		}

		for j, v := range row {
			if v != 0 {
				ind = append(ind, j)
				data = append(data, v)
				row[j] = 0
			}
		}
		indptr[i+1] = len(ind)
	}
	putFloats(row)
	c.matrix.I, c.matrix.J = ar, ac
	c.commitWorkspace(indptr, ind, data)
}

// addCSRCSR adds 2 CSR matrices together storing the result in the receiver.
// This method is specially optimised to take advantage of the sparsity patterns
// of the 2 CSR matrices.
func (c *CSR) addCSRCSR(a, b *CSR) {
	ar, ac := a.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, true)

	for row, start1 := range a.matrix.Indptr[0 : len(a.matrix.Indptr)-1] {
		indptr[row+1] = indptr[row]
		start2 := b.matrix.Indptr[row]
		end1 := a.matrix.Indptr[row+1]
		end2 := b.matrix.Indptr[row+1]
		if start1 == end1 {
			if start2 == end2 {
				continue
			}
			for k := start2; k < end2; k++ {
				data = append(data, b.matrix.Data[k])
				ind = append(ind, b.matrix.Ind[k])
				indptr[row+1]++
			}
			continue
		} else if start2 == end2 {
			for k := start1; k < end1; k++ {
				data = append(data, a.matrix.Data[k])
				ind = append(ind, a.matrix.Ind[k])
				indptr[row+1]++
			}
			continue
		}
		i := start1
		j := start2
		for {
			if i == end1 && j == end2 {
				break
			} else if i == end1 {
				for k := j; k < end2; k++ {
					data = append(data, b.matrix.Data[k])
					ind = append(ind, b.matrix.Ind[k])
					indptr[row+1]++
				}
				break
			} else if j == end2 {
				for k := i; k < end1; k++ {
					data = append(data, a.matrix.Data[k])
					ind = append(ind, a.matrix.Ind[k])
					indptr[row+1]++
				}
				break
			}
			if a.matrix.Ind[i] == b.matrix.Ind[j] {
				val := a.matrix.Data[i] + b.matrix.Data[j]
				data = append(data, val)
				ind = append(ind, a.matrix.Ind[i])
				indptr[row+1]++
				i++
				j++
			} else if a.matrix.Ind[i] < b.matrix.Ind[j] {
				data = append(data, a.matrix.Data[i])
				ind = append(ind, a.matrix.Ind[i])
				indptr[row+1]++
				i++
			} else {
				data = append(data, b.matrix.Data[j])
				ind = append(ind, b.matrix.Ind[j])
				indptr[row+1]++
				j++
			}
		}
	}
	c.matrix.I, c.matrix.J = ar, ac
	c.commitWorkspace(indptr, ind, data)
}
