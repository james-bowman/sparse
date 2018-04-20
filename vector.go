package sparse

import (
	"math"
	"sort"

	"github.com/gonum/floats"
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser    = (*Vector)(nil)
	_ mat.Matrix = (*Vector)(nil)
	_ mat.Vector = (*Vector)(nil)
)

// Vector is a sparse vector format.  It implements the mat.Vector
// interface but is optimised for sparsely populated vectors where
// most of the elements contain zero values by only storing and
// processing the non-zero values.  The format is similar to the
// triplet format used by COO matrices (and CSR/CSC) but only uses
// 2 arrays because the vector is 1 dimensional rather than 2.
type Vector struct {
	len  int
	ind  []int
	data []float64
}

// NewVector returns a new sparse vector of length len with
// elements specified by ind[] containing the values conatined
// within data.  Vector will reuse the same storage as the slices
// passed in and so any changes to the vector will be reflected
// in the slices and vice versa.
func NewVector(len int, ind []int, data []float64) *Vector {
	return &Vector{
		len:  len,
		ind:  ind,
		data: data,
	}
}

// Dims returns the dimensions of the vector.  This will be
// equivalent to Len(), 1
func (v *Vector) Dims() (r, c int) {
	return v.len, 1
}

// At returns the element at r, c.  At will panic if c != 0.
func (v *Vector) At(r, c int) float64 {
	if c != 0 {
		panic(mat.ErrColAccess)
	}
	return v.AtVec(r)
}

// T returns the transpose of the receiver.
func (v *Vector) T() mat.Matrix {
	return mat.TransposeVec{Vector: v}
}

// NNZ returns the number of non-zero elements in the vector.
func (v *Vector) NNZ() int {
	return len(v.data)
}

// AtVec returns the i'th element of the Vector.
func (v *Vector) AtVec(i int) float64 {
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
func (v *Vector) Len() int {
	return v.len
}

// DoNonZero calls the function fn for each of the non-zero elements of the receiver.
// The function fn takes a row/column index and the element value of the receiver at
// (i, j).
func (v *Vector) DoNonZero(fn func(i int, j int, v float64)) {
	for i := 0; i < len(v.ind); i++ {
		fn(v.ind[i], 0, v.data[i])
	}
}

// Gather gathers the entries from the supplied mat.VecDense structure
// that have corresponding non-zero entries in the receiver into the
// receiver.  The method will panic if denseVector is not the same
// length as the receiver.
func (v *Vector) Gather(denseVector *mat.VecDense) {
	if v.len != denseVector.Len() {
		panic(mat.ErrShape)
	}
	vec := denseVector.RawVector()
	blas.Usga(vec.Data, vec.Inc, v.data, v.ind)
}

// GatherAndZero gathers the entries from the supplied mat.VecDense
// structure that have corresponding non-zero entries in the receiver
// into the receiver and then zeros those entries in denseVector.
// The method will panic if denseVector is not the same length
// as the receiver.
func (v *Vector) GatherAndZero(denseVector *mat.VecDense) {
	if v.len != denseVector.Len() {
		panic(mat.ErrShape)
	}

	vec := denseVector.RawVector()
	blas.Usgz(vec.Data, vec.Inc, v.data, v.ind)
}

// Scatter scatters elements from the receiver into the supplied mat.VecDense
// structure, denseVector and returns a pointer to it.  The method will panic
// if denseVector is not the same length as the receiver (unless it is nil)
func (v *Vector) Scatter(denseVector *mat.VecDense) *mat.VecDense {
	if v.len != denseVector.Len() {
		panic(mat.ErrShape)
	}
	vec := denseVector.RawVector()
	blas.Ussc(v.data, vec.Data, vec.Inc, v.ind)
	return denseVector
}

// CloneVec clones the supplied mat.Vector, a into the receiver, overwriting
// the previous values of the receiver.  If the receiver is of a different
// length from a, it will be resized to accomodate the values from a.
func (v *Vector) CloneVec(a mat.Vector) {
	if v == a {
		return
	}
	v.len = a.Len()

	if sv, isSparse := a.(*Vector); isSparse {
		size := len(sv.ind)
		if size > cap(v.ind) {
			v.ind = make([]int, size)
			v.data = make([]float64, size)
		} else {
			v.ind = v.ind[:size]
			v.data = v.data[:size]
		}
		for i, val := range sv.ind {
			v.ind[i] = val
			v.data[i] = sv.data[i]
		}
		return
	}

	v.ind = v.ind[:0]
	v.data = v.data[:0]

	for i := 0; i < v.len; i++ {
		val := a.AtVec(i)
		if val != 0 {
			v.ind = append(v.ind, i)
			v.data = append(v.data, val)
		}
	}
}

// ToDense converts the sparse vector to a dense vector
// The returned dense matrix is a new copy of the receiver.
func (v *Vector) ToDense() *mat.VecDense {
	return v.Scatter(mat.NewVecDense(v.len, nil))
}

// AddVec adds the vectors a and b, placing the result in the receiver.
// AddVec will panic if a and b are not the same length.  If a and b
// are both sparse Vector vectors then AddVec will only process the
// non-zero elements.
func (v *Vector) AddVec(a, b mat.Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(mat.ErrShape)
	}

	v.len = ar

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*Vector)
	sb, bIsSparse := b.(*Vector)
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
// optimised for processing sparse Vector vectors and only processes
// non-zero elements.
func (v *Vector) addVecSparse(a *Vector, alpha float64, b *Vector) {
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
func (v *Vector) ScaleVec(alpha float64, a mat.Vector) {
	v.len = a.Len()

	if alpha == 0 {
		v.ind = v.ind[:0]
		v.data = v.data[:0]
		return
	}

	if s, isSparse := a.(*Vector); isSparse {
		ind, data := v.createWorkspace(s.NNZ(), false)
		copy(ind, s.ind)
		for i, val := range s.data {
			data[i] = alpha * val
		}
		v.commitWorkspace(ind, data)
		return
	}

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
func (v *Vector) AddScaledVec(a mat.Vector, alpha float64, b mat.Vector) {
	ar := a.Len()
	br := b.Len()

	if ar != br {
		panic(mat.ErrShape)
	}

	v.len = ar

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*Vector)
	sb, bIsSparse := b.(*Vector)
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
func (v *Vector) Norm(L float64) float64 {
	if L == 2 {
		twoNorm := math.Pow(math.Abs(v.data[0]), 2)
		for i := 1; i < len(v.data); i++ {
			twoNorm += math.Pow(v.data[i], 2)
		}
		return math.Sqrt(twoNorm)
	}
	return floats.Norm(v.data, L)
}

// Dot returns the sum of the element-wise product (dot product) of a and b.
// Dot panics if the matrix sizes are unequal.  For sparse vectors, Dot will
// only process non-zero elements otherwise this method simply delegates to
// mat.Dot()
func Dot(a, b mat.Vector) float64 {
	if a.Len() != b.Len() {
		panic(mat.ErrShape)
	}

	as, aIsSparse := a.(*Vector)
	bs, bIsSparse := b.(*Vector)

	if aIsSparse {
		if bIsSparse {
			// bdense := make([]float64, bs.Len())
			// blas.Ussc(bs.data, bdense, 1, bs.ind)
			// return blas.Usdot(as.data, as.ind, bdense, 1)
			dotSparseSparse(as, bs)
		}
		if bdense, bIsDense := b.(mat.RawVectorer); bIsDense {
			raw := bdense.RawVector()
			return blas.Usdot(as.data, as.ind, raw.Data, raw.Inc)
		}
		return dotSparse(as, b)
	}
	if bIsSparse {
		if adense, aIsDense := a.(mat.RawVectorer); aIsDense {
			raw := adense.RawVector()
			return blas.Usdot(bs.data, bs.ind, raw.Data, raw.Inc)
		}
		return dotSparse(bs, a)
	}
	return mat.Dot(a, b)
}

// dotSparseSparse returns the sum of the element-wise product of
// a and b where a and b are both sparse Vector vectors.  dotSparse
// will only process non-zero elements in the vectors.
func dotSparseSparse(a, b *Vector) float64 {
	var result float64
	var lhs, rhs *Vector

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

// dotSparse returns the sum of the element-wise multiplication
// of a and b where a is sparse and b is any implementation of
// mat.Vector.
func dotSparse(a *Vector, b mat.Vector) float64 {
	var result float64
	for i, ind := range a.ind {
		result += a.data[i] * b.AtVec(ind)
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
func (v *Vector) createWorkspace(size int, zero bool) ([]int, []float64) {
	ind := getInts(size, zero)
	data := getFloats(size, zero)

	return ind, data
}

// commitWorkspace commits a temporary workspace previously created
// with createWorkspace.  This has the effect of updaing the receiver
// with the values from the temporary workspace and returning the
// memory used by the workspace to the pool for other operations to
// reuse.
func (v *Vector) commitWorkspace(indexes []int, data []float64) {
	v.ind, indexes = indexes, v.ind
	v.data, data = data, v.data
	putInts(indexes)
	putFloats(data)
}
