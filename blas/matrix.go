package blas

import (
	"sort"
)

// SparseMatrix represents the common structure for representing compressed sparse
// matrix formats e.g. CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column)
type SparseMatrix struct {
	I, J   int
	Indptr []int
	Ind    []int
	Data   []float64
}

// At returns the element of the matrix located at coordinate i, j.
func (m *SparseMatrix) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(m.I) {
		panic("sparse/blas: index out of range")
	}
	if uint(j) < 0 || uint(j) >= uint(m.J) {
		panic("sparse/blas: index out of range")
	}

	// binary search through elements (assumes dimension is sorted)
	// TODO, small, low density or individual sparse row/columns
	// may be faster using linear search.  Consider selecting
	// algorithms dynamically.
	idx := m.Indptr[i] + sort.SearchInts(m.Ind[m.Indptr[i]:m.Indptr[i+1]], j)
	if idx < m.Indptr[i+1] && m.Ind[idx] == j {
		return m.Data[idx]
	}

	// for k := c.indptr[i]; k < c.indptr[i+1]; k++ {
	// 	if c.ind[k] == j {
	// 		return c.data[k]
	// 	}
	// }

	return 0
}

// Set is a generic method to set a matrix element.
func (m *SparseMatrix) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(m.I) {
		panic("sparse/blas: index out of range")
	}
	if uint(j) < 0 || uint(j) >= uint(m.J) {
		panic("sparse/blas: index out of range")
	}

	if v == 0 {
		// don't bother storing zero values
		return
	}

	// TODO switch to binary search
	for k := m.Indptr[i]; k < m.Indptr[i+1]; k++ {
		if m.Ind[k] == j {
			// if element(i, j) is already a non-zero value then simply update the existing
			// value without altering the sparsity pattern
			m.Data[k] = v
			return
		}

		if m.Ind[k] > j {
			// element(i, j) doesn't exist in current sparsity pattern and is mid row/col
			// so add it
			m.insert(i, j, v, k)
			return
		}
	}

	// element(i, j) doesn't exist in current sparsity pattern and is beyond the last
	// non-zero element of a row/col or an empty row/col - so add it
	m.insert(i, j, v, m.Indptr[i+1])
}

// insert inserts a new non-zero element into the sparse matrix, updating the
// sparsity pattern
func (m *SparseMatrix) insert(i int, j int, v float64, insertionPoint int) {
	m.Ind = append(m.Ind, 0)
	copy(m.Ind[insertionPoint+1:], m.Ind[insertionPoint:])
	m.Ind[insertionPoint] = j

	m.Data = append(m.Data, 0)
	copy(m.Data[insertionPoint+1:], m.Data[insertionPoint:])
	m.Data[insertionPoint] = v

	for n := i + 1; n <= m.I; n++ {
		m.Indptr[n]++
	}
}
