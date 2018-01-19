package sparse

import (
	"math"
	"sort"

	"github.com/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser     = (*VecCOO)(nil)
	_ mat.Matrix  = (*VecCOO)(nil)
	_ mat.Reseter = (*VecCOO)(nil)
)

// VecCOO is a sparse vector format.  It implements the mat.Vector
// interface but is optimised for sparsely populated vectors where
// most of the elements contain zero values by only storing and
// processing the non-zero values.  The format is similar to the
// triplet format used by COO matrices (and CSR/CSC) but only uses
// 2 arrays because the vector is 1 dimensional rather than 2.
type VecCOO struct {
	len  int
	ind  []int
	data []float64
}

// NewVecCOO returns a new sparse vector of length len with
// elements specified by ind[] containing the values conatined
// within data.  VecCOO will reuse the same storage as the slices
// passed in and so any changes to the vector will be reflected
// in the slices and vice versa.
func NewVecCOO(len int, ind []int, data []float64) *VecCOO {
	return &VecCOO{
		len:  len,
		ind:  ind,
		data: data,
	}
}

// Dims returns the dimensions of the vector.  This will be
// equivalent to Len(), 1
func (v *VecCOO) Dims() (r, c int) {
	return v.len, 1
}

// At returns the element at r, c.  At will panic if c != 0.
func (v *VecCOO) At(r, c int) float64 {
	if c != 0 {
		panic(mat.ErrColAccess)
	}
	return v.AtVec(r)
}

// T returns the transpose of the receiver.
func (v *VecCOO) T() mat.Matrix {
	return mat.TransposeVec{Vector: v}
}

// NNZ returns the number of non-zero elements in the vector.
func (v *VecCOO) NNZ() int {
	return len(v.data)
}

// AtVec returns the i'th element of the Vector.
func (v *VecCOO) AtVec(i int) float64 {
	if i < 0 || i >= v.len {
		panic(mat.ErrRowAccess)
	}

	idx := sort.SearchInts(v.ind, i)
	if idx < len(v.ind) && v.ind[idx] == i {
		return v.data[idx]
	}

	// for j := 0; j < len(v.ind); j++ {
	// 	if v.ind[j] == i {
	// 		return v.data[j]
	// 	}
	// }
	return 0.0
}

// Len returns the length of the vector
func (v *VecCOO) Len() int {
	return v.len
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).
func (v *VecCOO) DoNonZero(fn func(i int, j int, v float64)) {
	for i := 0; i < len(v.ind); i++ {
		fn(v.ind[i], 0, v.data[i])
	}
}

// Reset zeros the length of the vector so that it can be reused.
//
// See the Reseter interface for more information.
func (v *VecCOO) Reset() {
	v.len = 0
	v.ind = v.ind[:0]
	v.data = v.data[:0]
}

// IsZero returns whether the receiver is zero-sized. Zero-sized vectors can be the
// receiver for size-restricted operations. VecCOO can be zeroed using Reset.
func (v *VecCOO) IsZero() bool {
	return v.len == 0
}

// AddVec adds the vectors a and b, placing the result in the receiver.
// AddVec will panic if a and b are not the same length.  If a and b
// are both sparse VecCOO vectors then AddVec will only process the
// non-zero elements.
func (v *VecCOO) AddVec(a, b mat.Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(mat.ErrShape)
	}

	v.len = ar

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*VecCOO)
	sb, bIsSparse := b.(*VecCOO)
	if aIsSparse && bIsSparse {
		v.addVecSparse(sa, 1, sb)
		return
	}

	// fallback to basic vector addition with no optimisations
	// for sparseness
	ind, data := v.createWorkspace(0, false)

	for i := 0; i < v.len; i++ {
		p := a.AtVec(i) + b.AtVec(i)
		if p != 0 {
			ind = append(ind, i)
			data = append(data, p)
		}
	}

	v.commitWorkspace(ind, data)
}

// addVecSparse adds the vectors a and alpha*b.  This method is
// optimised for processing sparse VecCOO vectors and only processes
// non-zero elements.
func (v *VecCOO) addVecSparse(a *VecCOO, alpha float64, b *VecCOO) {
	ind, data := v.createWorkspace(0, false)

	var i, j int
	for i < len(a.ind) && j < len(b.ind) {
		if a.ind[i] == b.ind[j] {
			s := b.data[j]
			if alpha != 1 {
				s = alpha * b.data[j]
			}
			ind = append(ind, a.ind[i])
			data = append(data, a.data[i]+s)
			i++
			j++
			continue
		}
		if a.ind[i] < b.ind[j] {
			var k int
			for k = i; k < len(a.ind) && a.ind[k] < b.ind[j]; k++ {
				ind = append(ind, a.ind[k])
				data = append(data, a.data[k])
			}
			i = k
			continue
		}
		if a.ind[i] > b.ind[j] {
			var k int
			for k = j; k < len(b.ind) && b.ind[k] < a.ind[i]; k++ {
				s := b.data[j]
				if alpha != 1 {
					s = alpha * b.data[j]
				}
				ind = append(ind, b.ind[k])
				data = append(data, s)
			}
			j = k
			continue
		}
	}
	for k := i; k < len(a.ind); k++ {
		ind = append(ind, a.ind[k])
		data = append(data, a.data[k])
	}
	for k := j; k < len(b.ind); k++ {
		s := b.data[j]
		if alpha != 1 {
			s = alpha * b.data[j]
		}
		ind = append(ind, b.ind[k])
		data = append(data, s)
	}

	v.commitWorkspace(ind, data)
}

// ScaleVec scales the vector a by alpha, placing the result in the
// receiver.
func (v *VecCOO) ScaleVec(alpha float64, a mat.Vector) {
	v.len = a.Len()
	ind, data := v.createWorkspace(0, false)

	for i := 0; i < v.len; i++ {
		val := a.AtVec(i)
		if val != 0 {
			ind = append(ind, i)
			data = append(data, alpha*val)
		}
	}
	v.commitWorkspace(ind, data)
}

// AddScaledVec adds the vectors a and alpha*b, placing the result
// in the receiver.  AddScaledVec will panic if a and b are not the
// same length.
func (v *VecCOO) AddScaledVec(a mat.Vector, alpha float64, b mat.Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(mat.ErrShape)
	}

	v.len = ar

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*VecCOO)
	sb, bIsSparse := b.(*VecCOO)
	if aIsSparse && bIsSparse {
		v.addVecSparse(sa, alpha, sb)
		return
	}

	ind, data := v.createWorkspace(0, false)

	for i := 0; i < v.len; i++ {
		val := a.AtVec(i) + alpha*b.AtVec(i)
		if val != 0 {
			ind = append(ind, i)
			data = append(data, val)
		}
	}
	v.commitWorkspace(ind, data)
}

// Norm calculates the Norm of the vector only processing the
// non-zero elements.
// See Normer interface for more details.
func (v *VecCOO) Norm(L float64) float64 {
	if L == 2 {
		twoNorm := math.Pow(math.Abs(v.data[0]), 2)
		for i := 1; i < len(v.data); i++ {
			twoNorm += math.Pow(v.data[i], 2)
		}
		return math.Sqrt(twoNorm)
	}
	return floats.Norm(v.data, L)
}

// Dot returns the sum of the element-wise product of a and b.
// Dot panics if the matrix sizes are unequal.  If both vectors
// are sparse VecCOO then Dot will only process non-zero elements
// otherwise this method simply delegates to mat.Dot()
func Dot(a, b mat.Vector) float64 {
	if av, ok := a.(*VecCOO); ok {
		if bv, ok := b.(*VecCOO); ok {
			la := a.Len()
			lb := b.Len()
			if la != lb {
				panic(mat.ErrShape)
			}
			return dotSparse(av, bv)
		}
	}
	return mat.Dot(a, b)
}

// dotSparse returns the sum of the element-wise product of a and
// b where a and b are both sparse VecCOO vectors.  dotSparse
// will only process non-zero elements in the vectors.
func dotSparse(a, b *VecCOO) float64 {
	var result float64
	var lhs, rhs *VecCOO

	if a.NNZ() < b.NNZ() {
		lhs, rhs = a, b
	} else {
		lhs, rhs = b, a
	}

	var j int
	for k := 0; k < len(lhs.ind); k++ {
		var bi int
		for bi = j; bi < len(rhs.ind) && rhs.ind[bi] < lhs.ind[k]; bi++ {
			// empty
		}
		j = bi
		if j >= len(rhs.ind) {
			break
		}
		if lhs.ind[k] == rhs.ind[bi] {
			result += lhs.data[k] * rhs.data[bi]
		}
	}

	return result
}

// createWorkspace creates a temporary workspace to store the result
// of vector operations avoiding the issue of mutating operands mid
// operation where they overlap with the receiver
// e.g.
//	result.AddVec(result, a)
// createWorkspace will attempt to reuse previously allocated memory
// for the temporary workspace where ever possible to avoid allocating
// memory and placing strain on GC.
func (v *VecCOO) createWorkspace(size int, zero bool) ([]int, []float64) {
	ind := getInts(size, zero)
	data := getFloats(size, zero)

	return ind, data
}

// commitWorkspace commits a temporary workspace previously created
// with createWorkspace.  This has the effect of updaing the receiver
// with the values from the temporary workspace and returning the
// memory used by the workspace to the pool for other operations to
// reuse.
func (v *VecCOO) commitWorkspace(indexes []int, data []float64) {
	v.ind, indexes = indexes, v.ind
	v.data, data = data, v.data
	putInts(indexes)
	putFloats(data)
}
