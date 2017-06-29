package sparse

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// Mul takes the matrix product (Dot product) of the supplied matrices a and b and stores the result
// in the receiver.  If the number of columns does not equal the number of rows in b, Mul will panic.
func (c *CSR) Mul(a, b mat64.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(matrix.ErrShape)
	}

	if dia, ok := a.(*DIA); ok {
		c.mulDIA(dia, b, false)
		return
	}
	if dia, ok := b.(*DIA); ok {
		c.mulDIA(dia, a, true)
		return
	}

	c.indptr = make([]int, ar+1)

	c.i, c.j = ar, bc
	t := 0

	lhs, isCsr := a.(*CSR)

	if isCsr {
		for i := 0; i < ar; i++ {
			c.indptr[i] = t
			for j := 0; j < bc; j++ {
				var v float64
				// TODO Consider converting all Sparsers to CSR
				for k := lhs.indptr[i]; k < lhs.indptr[i+1]; k++ {
					v += lhs.data[k] * b.At(lhs.ind[k], j)
				}
				if v != 0 {
					t++
					c.ind = append(c.ind, j)
					c.data = append(c.data, v)
				}
			}
		}
	} else {
		row := make([]float64, ac)
		for i := 0; i < ar; i++ {
			c.indptr[i] = t
			// bizarely transferring the row elements into a slice as part of a separate loop
			// (rather than accessing each element within the main loop (a.At(m, n) * b.At(m, n))
			// then ranging over them as the main loop is about twice as fast.  Possibly a
			// result of inlining the call as compiler optimisation?
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
					c.ind = append(c.ind, j)
					c.data = append(c.data, v)
				}
			}
		}
	}

	c.indptr[c.i] = t
}

// mulDIA takes the matrix product of the diagonal matrix dia and an other matrix, other and stores the result
// in the receiver.  This method caters for the specialised case of multiplying by a diagonal matrix where
// significant optimisation is possible due to the sparsity pattern of the matrix.  If trans is true, the method
// will assume that other was the LHS (Left Hand Side) operand and that dia was the RHS.
func (c *CSR) mulDIA(dia *DIA, other mat64.Matrix, trans bool) {
	var csMat compressedSparse
	isCS := false

	if csr, ok := other.(*CSR); ok {
		// TODO consider implicitly converting all sparsers to CSR
		// or at least iterating only over the non-zero elements
		csMat = csr.compressedSparse
		isCS = true
		c.ind = make([]int, len(csMat.ind))
		c.data = make([]float64, len(csMat.data))
	}

	c.i, c.j = other.Dims()
	c.indptr = make([]int, c.i+1)
	t := 0
	raw := dia.Diagonal()

	for i := 0; i < c.i; i++ {
		c.indptr[i] = t
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
					c.ind[t] = csMat.ind[k]
					c.data[t] = v
					t++
				}
			}
		} else {
			for k := 0; k < c.j; k++ {
				var rawval float64
				if trans {
					rawval = raw[k]
				} else {
					rawval = raw[i]
				}
				v = other.At(i, k) * rawval
				if v != 0 {
					c.ind = append(c.ind, k)
					c.data = append(c.data, v)
					t++
				}
			}
		}
	}

	c.indptr[c.i] = t

	return
}

// Add adds matrices a and b together and stores the result in the receiver.
// If matrices a and b are not the same shape then the method will panic.
func (c *CSR) Add(a, b mat64.Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ar != br || ac != bc {
		panic(matrix.ErrShape)
	}

	// take a copy of the largest (higher NNZ if sparse or copy if dense) matrix
	// then iterate over NZ values of smaller matrix (lower NNZ) and add elements
	// in-place to corresponding element in copied matrix.
	lCsr, lIsCsr := a.(*CSR)
	rCsr, rIsCsr := b.(*CSR)
	var other *CSR

	if lIsCsr && rIsCsr {
		if lCsr.NNZ() >= rCsr.NNZ() {
			*c = *(lCsr.Copy().(*CSR))
			other = rCsr
		} else {
			*c = *(rCsr.Copy().(*CSR))
			other = lCsr
		}
	} else if lIsCsr {
		c.From(b)
		other = lCsr
	} else if rIsCsr {
		c.From(a)
		other = rCsr
	} else {
		// dumb addition with no sparcity optimisations/savings
		c.i, c.j = ar, ac
		c.indptr = make([]int, c.i+1)
		for i := 0; i < ar; i++ {
			for j := 0; j < ac; j++ {
				c.Set(i, j, a.At(i, j)+b.At(i, j))
			}
		}
		return
	}

	for i := 0; i < len(other.indptr)-1; i++ {
		for j := other.indptr[i]; j < other.indptr[i+1]; j++ {
			c.Set(i, other.ind[j], other.data[j]+c.At(i, other.ind[j]))
		}
	}
}
