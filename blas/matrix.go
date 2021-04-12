package blas

import (
	"gonum.org/v1/gonum/floats/scalar"
	"gonum.org/v1/gonum/mat"
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

	for k := m.Indptr[i]; k < m.Indptr[i+1]; k++ {
		if m.Ind[k] == j {
			return m.Data[k]
		}
	}

	return 0
}

// Set is a generic method to set a matrix element.  Note: setting a non-zero element to zero
// does not remove the element from the sparcity pattern but will actually store a zero value.
func (m *SparseMatrix) Set(i, j int, v float64) {
	if uint(i) < 0 || uint(i) >= uint(m.I) {
		panic("sparse/blas: index out of range")
	}
	if uint(j) < 0 || uint(j) >= uint(m.J) {
		panic("sparse/blas: index out of range")
	}

	for k := m.Indptr[i]; k < m.Indptr[i+1]; k++ {
		if m.Ind[k] == j {
			// if element(i, j) is already a non-zero value then simply update the existing
			// value without altering the sparsity pattern
			m.Data[k] = v
			return
		}
	}

	if v == 0 {
		// don't bother storing new zero values
		return
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

func (m *SparseMatrix) nnzWithin(epsilon float64) int {
	count := 0
	for _, v := range m.Data {
		if !scalar.EqualWithinAbs(v, 0, epsilon) {
			count++
		}
	}
	return count
}

// Cull returns a new SparseMatrix with all entries within epsilon of 0 removed.
func (m *SparseMatrix) Cull(epsilon float64) *SparseMatrix {
	nMajor := len(m.Indptr)
	targetNNZ := m.nnzWithin(epsilon)
	newIndPtr := make([]int, nMajor)
	newInd := make([]int, targetNNZ)
	newData := make([]float64, targetNNZ)
	curPos := 0
	for major := 0; major < nMajor-1; major++ {
		startIdx := m.Indptr[major]
		endIdx := m.Indptr[major+1]
		newIndPtr[major] = curPos
		for minor := startIdx; minor < endIdx; minor++ {
			col := m.Ind[minor]
			v := m.Data[minor]
			if !scalar.EqualWithinAbs(v, 0, epsilon) {
				newInd[curPos] = col
				newData[curPos] = v
				curPos++
			}
		}
	}
	if curPos != targetNNZ {
		panic(mat.ErrShape)
	}
	newIndPtr[nMajor-1] = curPos
	return &SparseMatrix{
		I: m.I, J: m.J,
		Indptr: newIndPtr,
		Ind:    newInd,
		Data:   newData,
	}
}
