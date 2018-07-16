package sparse

import (
	"math"

	"github.com/gonum/floats"
	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser     = (*Vector)(nil)
	_ mat.Matrix  = (*Vector)(nil)
	_ mat.Vector  = (*Vector)(nil)
	_ mat.Reseter = (*Vector)(nil)
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
	for j := 0; j < len(v.ind); j++ {
		if v.ind[j] == i {
			return v.data[j]
		}
	}
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
	blas.Dusga(vec.Data, vec.Inc, v.data, v.ind)
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
	blas.Dusgz(vec.Data, vec.Inc, v.data, v.ind)
}

// Scatter scatters elements from the receiver into the supplied mat.VecDense
// structure, denseVector and returns a pointer to it.  The method will panic
// if denseVector is not the same length as the receiver (unless it is nil)
func (v *Vector) Scatter(denseVector *mat.VecDense) *mat.VecDense {
	if v.len != denseVector.Len() {
		panic(mat.ErrShape)
	}
	vec := denseVector.RawVector()
	blas.Dussc(v.data, vec.Data, vec.Inc, v.ind)
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

	if s, isSparse := a.(*Vector); isSparse {
		nnz := s.NNZ()
		v.ind = useInts(v.ind, nnz, false)
		v.data = useFloats(v.data, nnz, false)
		copy(v.ind, s.ind)
		copy(v.data, s.data)
		return
	}

	if v.IsZero() {
		v.len = a.Len()
		nnz := v.len / 10
		v.ind = useInts(v.ind, nnz, false)
		v.data = useFloats(v.data, nnz, false)
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

	if temp, restore := v.spalloc(a, b, ar); temp {
		defer restore()
	}

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*Vector)
	sb, bIsSparse := b.(*Vector)
	if aIsSparse && bIsSparse {
		v.addVecSparse(1, sa, 1, sb)
		return
	}

	for i := 0; i < v.len; i++ {
		p := a.AtVec(i) + b.AtVec(i)
		if p != 0 {
			v.ind = append(v.ind, i)
			v.data = append(v.data, p)
		}
	}
}

// addVecSparse2 adds the vectors a and alpha*b.  This method is
// optimised for processing sparse Vector vectors and only processes
// non-zero elements.
func (v *Vector) addVecSparse(alpha float64, a *Vector, beta float64, b *Vector) {
	spa := NewSPA(a.len)
	spa.Scatter(a.data, a.ind, alpha, &v.ind)
	spa.Scatter(b.data, b.ind, beta, &v.ind)
	spa.Gather(&v.data, &v.ind)
}

// ScaleVec scales the vector a by alpha, placing the result in the
// receiver.
func (v *Vector) ScaleVec(alpha float64, a mat.Vector) {
	alen := a.Len()
	if !v.IsZero() && alen != v.len {
		panic(mat.ErrShape)
	}

	if alpha == 0 {
		v.len = alen
		v.ind = v.ind[:0]
		v.data = v.data[:0]
		return
	}

	if s, isSparse := a.(*Vector); isSparse {
		nnz := s.NNZ()
		v.len = alen
		v.ind = useInts(v.ind, nnz, false)
		v.data = useFloats(v.data, nnz, false)
		copy(v.ind, s.ind)
		for i, val := range s.data {
			v.data[i] = alpha * val
		}
		return
	}

	if v.IsZero() {
		v.len = a.Len()
		nnz := v.len / 10
		v.ind = useInts(v.ind, nnz, false)
		v.data = useFloats(v.data, nnz, false)
	}
	v.ind = v.ind[:0]
	v.data = v.data[:0]

	for i := 0; i < v.len; i++ {
		val := a.AtVec(i)
		if val != 0 {
			v.ind = append(v.ind, i)
			v.data = append(v.data, alpha*val)
		}
	}
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

	if temp, restore := v.spalloc(a, b, ar); temp {
		defer restore()
	}

	// Sparse specific optimised implementation
	sa, aIsSparse := a.(*Vector)
	sb, bIsSparse := b.(*Vector)
	if aIsSparse && bIsSparse {
		v.addVecSparse(1, sa, alpha, sb)
		return
	}

	for i := 0; i < v.len; i++ {
		val := a.AtVec(i) + alpha*b.AtVec(i)
		if val != 0 {
			v.ind = append(v.ind, i)
			v.data = append(v.data, val)
		}
	}
}

// Norm calculates the Norm of the vector only processing the
// non-zero elements.
// See Normer interface for more details.
func (v *Vector) Norm(L float64) float64 {
	if L == 2 {
		return math.Sqrt(Dot(v, v))
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
			buf := getFloats(bs.len, true)
			defer putFloats(buf)
			blas.Dussc(bs.data, buf, 1, bs.ind)
			val := blas.Dusdot(as.data, as.ind, buf, 1)
			return val
		}
		if bdense, bIsDense := b.(mat.RawVectorer); bIsDense {
			raw := bdense.RawVector()
			return blas.Dusdot(as.data, as.ind, raw.Data, raw.Inc)
		}
		return dotSparse(as, b)
	}
	if bIsSparse {
		if adense, aIsDense := a.(mat.RawVectorer); aIsDense {
			raw := adense.RawVector()
			return blas.Dusdot(bs.data, bs.ind, raw.Data, raw.Inc)
		}
		return dotSparse(bs, a)
	}
	return mat.Dot(a, b)
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

// Reset zeros the dimensions of the vector so that it can be reused as the
// receiver of a dimensionally restricted operation.
//
// See the Gonum mat.Reseter interface for more information.
func (v *Vector) Reset() {
	v.len = 0
	v.ind = v.ind[:0]
	v.data = v.data[:0]
}

// IsZero returns whether the receiver is zero-sized. Zero-sized vectors can be the
// receiver for size-restricted operations. Vectors can be zeroed using the Reset
// method.
func (v *Vector) IsZero() bool {
	return v.len == 0
}

// reuseAs resizes a zero-sized vector to be len long or checks a non-zero-sized vector
// is already the correct size (len).  If the vector is resized, the method will
// ensure there is sufficient initial capacity allocated in the underlying storage
// to store up to nnz non-zero elements although this will be extended
// automatically later as needed (using Go's built-in append function).
func (v *Vector) reuseAs(len, nnz int) {
	if v.IsZero() {
		v.len = len
		v.ind = useInts(v.ind, nnz, false)
		v.data = useFloats(v.data, nnz, false)

		return
	}

	if len != v.len {
		panic(mat.ErrShape)
	}
}

// checkOverlap checks whether the receiver overlaps or is an alias for the
// matrix a.  The method returns true (indicating overlap) if c == a or if
// any of the receiver's internal data structures share underlying storage with a.
func (v *Vector) checkOverlap(a mat.Matrix) bool {
	if v == a {
		return true
	}

	switch a := a.(type) {
	case *Vector:
		return aliasInts(v.ind, a.ind) ||
			aliasFloats(v.data, a.data)
	case *COO:
		return aliasInts(v.ind, a.cols) ||
			aliasInts(v.ind, a.rows) ||
			aliasFloats(v.data, a.data)
	case *CSR, *CSC:
		m := a.(BlasCompatibleSparser).RawMatrix()
		return aliasInts(v.ind, m.Ind) ||
			aliasFloats(v.data, m.Data)
	default:
		return false
	}
}

// temporaryWorkspace returns a new Vector w of length len with
// initial capacity allocated for nnz non-zero elements and
// returns a callback to defer which performs cleanup at the return of the call.
// This should be used when a method receiver is the same pointer as an input argument.
func (v *Vector) temporaryWorkspace(len, nnz int) (w *Vector, restore func()) {
	w = getVecWorkspace(len, nnz)
	return w, func() {
		v.CloneVec(w)
		putVecWorkspace(w)
	}
}

// spalloc ensures appropriate storage is allocated for the receiver sparse vector
// ensuring it is of length len and checking for any overlap or aliasing
// between operands a or b with c in which case a temporary isolated workspace is
// allocated and the returned value isTemp is true with restore representing a
// function to clean up and restore the workspace once finished.
func (v *Vector) spalloc(a mat.Matrix, b mat.Matrix, len int) (isTemp bool, restore func()) {
	var nnz int
	lSp, lIsSp := a.(Sparser)
	rSp, rIsSp := b.(Sparser)
	if lIsSp && rIsSp {
		nnz = lSp.NNZ() + rSp.NNZ()
	} else {
		// assume 10% of elements will be non-zero
		nnz = len / 10
	}
	v.reuseAs(len, nnz)
	if v.checkOverlap(a) || v.checkOverlap(b) {
		var tmp *Vector
		tmp, restore = v.temporaryWorkspace(len, nnz)
		*v = *tmp
		isTemp = true
	}
	v.ind = v.ind[:0]
	v.data = v.data[:0]
	return
}
