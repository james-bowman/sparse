package sparse

import (
	"gonum.org/v1/gonum/mat"
)

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
				for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
					v += lhs.data[k] * b.At(lhs.ind[k], j)
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
			// bizarely transferring the row elements of the first operand into a slice as part
			// of a separate loop (rather than accessing each element within the main loop
			// (a.At(m, n) * b.At(m, n)) then ranging over them as part of the main loop is
			// about twice as fast.  This is related to lining the data up into CPU
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
	c.i, c.j = ar, bc
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

	for row := 0; row < m; row++ {
		val := 0.0
		for colind := lhs.indptr[row]; colind < lhs.indptr[row+1]; colind++ {
			val += lhs.data[colind] * rhs[lhs.ind[colind]]
		}
		out[row] = val
	}
}

// mulCSRCSR handles CSR = CSR * CSR using Gustavson Algorithm (ACM 1978)
func (c *CSR) mulCSRCSR(lhs *CSR, rhs *CSR) {
	ar, _ := lhs.Dims()
	_, bc := rhs.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, false)
	indptr[0] = 0

	x := make([]float64, bc)

	// rows in C
	for i := 0; i < ar; i++ {
		// each t in row B[i]
		for t := lhs.indptr[i]; t < lhs.indptr[i+1]; t++ {
			// each j in row A[t]
			for j := rhs.indptr[lhs.ind[t]]; j < rhs.indptr[lhs.ind[t]+1]; j++ {
				x[rhs.ind[j]] += lhs.data[t] * rhs.data[j]
			}
		}
		for j, v := range x {
			if v != 0 {
				ind = append(ind, j)
				data = append(data, v)
			}
			x[j] = 0
		}
		indptr[i+1] = len(ind)
	}
	c.i, c.j = ar, bc
	c.commitWorkspace(indptr, ind, data)
}

// mulCSRCSC handles special case of matrix multiplication (dot product) where the LHS matrix
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
			rhsStart := rhs.indptr[j]
			rhsEnd := rhs.indptr[j+1] - 1
			b := rhsStart

			for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
				var bi int
				for bi = b; bi < rhsEnd && rhs.ind[bi] < lhs.ind[k]; bi++ {
					// empty
				}
				b = bi
				if lhs.ind[k] == rhs.ind[bi] {
					v += lhs.data[k] * rhs.data[bi]
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
	c.i, c.j = ar, bc
	c.commitWorkspace(indptr, ind, data)
}

// mulDIA takes the matrix product of the diagonal matrix dia and an other matrix, other and stores the result
// in the receiver.  This method caters for the specialised case of multiplying by a diagonal matrix where
// significant optimisation is possible due to the sparsity pattern of the matrix.  If trans is true, the method
// will assume that other was the LHS (Left Hand Side) operand and that dia was the RHS.
func (c *CSR) mulDIA(dia *DIA, other mat.Matrix, trans bool) {
	var csMat compressedSparse
	isCS := false
	var size int

	if csr, ok := other.(*CSR); ok {
		// TODO consider implicitly converting all sparsers to CSR
		// or at least iterating only over the non-zero elements
		csMat = csr.compressedSparse
		isCS = true
		size = len(csMat.ind)
	}

	rows, cols := other.Dims()
	indptr, ind, data := c.createWorkspace(rows+1, size, true)

	t := 0
	raw := dia.Diagonal()

	for i := 0; i < rows; i++ {
		indptr[i] = t
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
					ind[t] = csMat.ind[k]
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

	c.i, c.j = rows, cols
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
	c.i, c.j = ar, ac
	c.commitWorkspace(indptr, ind, data)
}

// addCSR adds a CSR matrix to any implementation of mat.Matrix and stores the
// result in the receiver.
func (c *CSR) addCSR(csr *CSR, other mat.Matrix) {
	ar, ac := csr.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, false)

	t := 0
	//row := make([]float64, ac)
	row := getFloats(ac, false)
	for i := 0; i < ar; i++ {
		indptr[i] = t

		for ci := range row {
			row[ci] = other.At(i, ci)
		}

		csrStart := csr.indptr[i]
		csrEnd := csr.indptr[i+1] - 1
		b := csrStart

		for ci, e := range row {
			var v float64
			var bi int
			for bi = b; bi < csrEnd && csr.ind[bi] < ci; bi++ {
				// empty
			}
			b = bi
			if ci == csr.ind[bi] {
				v = e + csr.data[bi]
			} else {
				v = e
			}
			if v != 0 {
				t++
				ind = append(ind, ci)
				data = append(data, v)
			}
		}
	}
	putFloats(row)
	indptr[ar] = t
	c.i, c.j = ar, ac
	c.commitWorkspace(indptr, ind, data)
}

// addCSRCSR adds 2 CSR matrices together storing the result in the receiver.
// This method is specially optimised to take advantage of the sparsity patterns
// of the 2 CSR matrices.
func (c *CSR) addCSRCSR(a, b *CSR) {
	ar, ac := a.Dims()
	indptr, ind, data := c.createWorkspace(ar+1, 0, true)

	for row, start1 := range a.indptr[0 : len(a.indptr)-1] {
		indptr[row+1] = indptr[row]
		start2 := b.indptr[row]
		end1 := a.indptr[row+1]
		end2 := b.indptr[row+1]
		if start1 == end1 {
			if start2 == end2 {
				continue
			}
			for k := start2; k < end2; k++ {
				data = append(data, b.data[k])
				ind = append(ind, b.ind[k])
				indptr[row+1]++
			}
			continue
		} else if start2 == end2 {
			for k := start1; k < end1; k++ {
				data = append(data, a.data[k])
				ind = append(ind, a.ind[k])
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
					data = append(data, b.data[k])
					ind = append(ind, b.ind[k])
					indptr[row+1]++
				}
				break
			} else if j == end2 {
				for k := i; k < end1; k++ {
					data = append(data, a.data[k])
					ind = append(ind, a.ind[k])
					indptr[row+1]++
				}
				break
			}
			if a.ind[i] == b.ind[j] {
				val := a.data[i] + b.data[j]
				data = append(data, val)
				ind = append(ind, a.ind[i])
				indptr[row+1]++
				i++
				j++
			} else if a.ind[i] < b.ind[j] {
				data = append(data, a.data[i])
				ind = append(ind, a.ind[i])
				indptr[row+1]++
				i++
			} else {
				data = append(data, b.data[j])
				ind = append(ind, b.ind[j])
				indptr[row+1]++
				j++
			}
		}
	}
	c.i, c.j = ar, ac
	c.commitWorkspace(indptr, ind, data)
}
