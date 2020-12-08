package sparse

import (
	"math"
	"sort"

	"github.com/james-bowman/sparse/blas"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	_ Sparser     = (*Vector)(nil)
	_ mat.Matrix  = (*Vector)(nil)
	_ mat.Vector  = (*Vector)(nil)
	_ mat.Reseter = (*Vector)(nil)
	_ mat.Mutable = (*Vector)(nil)
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

// Set sets the element at row r, column c to the value val. Set will panic if c != 0.
func (v *Vector) Set(r, c int, val float64) {
	if c != 0 {
		panic(mat.ErrColAccess)
	}
	v.SetVec(r, val)
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

// SetVec sets the i'th element to the value val. It panics if i is out of bounds.
func (v *Vector) SetVec(i int, val float64) {

	// Panic if the sought index is out of bounds
	if i < 0 || i >= v.len {
		panic(mat.ErrRowAccess)
	}

	// Identify where in the slice this index would exist
	j := sort.SearchInts(v.ind, i)

	// The value is zero so we are really removing it
	if val == 0.0 {
		if j < len(v.ind) && v.ind[j] == i {
			v.ind = append(v.ind[:j], v.ind[j+1:]...)
			v.data = append(v.data[:j], v.data[j+1:]...)
		}
		return
	}

	// Set the value
	if j == len(v.ind) {
		v.ind = append(v.ind, i)
		v.data = append(v.data, val)
	} else if j < len(v.ind) {
		if v.ind[j] == i {
			v.data[j] = val
		} else {
			v.ind = append(v.ind[:j], append([]int{i}, v.ind[j:]...)...)
			v.data = append(v.data[:j], append([]float64{val}, v.data[j:]...)...)
		}
	}
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

// RawVector returns the underlying sparse vector data and indices
// respectively for raw manipulation or use in sparse BLAS routines.
func (v *Vector) RawVector() ([]float64, []int) {
	return v.data, v.ind
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
// length from a, it will be resized to accommodate the values from a.
func (v *Vector) CloneVec(a mat.Vector) {
	if v == a {
		return
	}

	v.len = a.Len()

	if s, isSparse := a.(*Vector); isSparse {
		v.reuseAs(s.Len(), s.NNZ(), false)
		copy(v.ind, s.ind)
		copy(v.data, s.data)
		return
	}

	v.reuseAs(v.len, v.len/10, false)

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

	if t, temp, restore := v.spalloc(a, b); temp {
		defer restore()
		v = t
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
		v.reuseAs(alen, nnz, false)
		copy(v.ind, s.ind)
		for i, val := range s.data {
			v.data[i] = alpha * val
		}
		return
	}

	v.reuseAs(alen, alen/10, true)

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

	if t, temp, restore := v.spalloc(a, b); temp {
		defer restore()
		v = t
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
		return math.Sqrt(floats.Dot(v.data, v.data))
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
			return dotSparseSparse(as, bs, nil)
		}
		if bdense, bIsDense := b.(mat.RawVectorer); bIsDense {
			raw := bdense.RawVector()
			return blas.Dusdot(as.data, as.ind, raw.Data, raw.Inc)
		}
		return dotSparse(as, b, nil)
	}
	if bIsSparse {
		if adense, aIsDense := a.(mat.RawVectorer); aIsDense {
			raw := adense.RawVector()
			return blas.Dusdot(bs.data, bs.ind, raw.Data, raw.Inc)
		}
		return dotSparse(bs, a, nil)
	}
	return mat.Dot(a, b)
}

// dotSparse returns the sum of the element-wise multiplication
// of a and b where a is sparse and b is any implementation of
// mat.Vector.
func dotSparse(a *Vector, b mat.Vector, c *Vector) float64 {
	var result float64
	for i, ind := range a.ind {
		val := a.data[i] * b.AtVec(ind)
		result += val
		if c != nil {
			c.SetVec(ind, val)
		}
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
func (v *Vector) reuseAs(len, nnz int, zero bool) {
	if v.IsZero() {
		v.len = len
	} else if len != v.len {
		panic(mat.ErrShape)
	}

	v.ind = useInts(v.ind, nnz, false)
	v.data = useFloats(v.data, nnz, false)

	if zero {
		v.ind = v.ind[:0]
		v.data = v.data[:0]
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
func (v *Vector) temporaryWorkspace(len, nnz int, zero bool) (w *Vector, restore func()) {
	w = getVecWorkspace(len, nnz, zero)

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
func (v *Vector) spalloc(a mat.Vector, b mat.Vector) (t *Vector, isTemp bool, restore func()) {
	var nnz int
	t = v
	lSp, lIsSp := a.(Sparser)
	rSp, rIsSp := b.(Sparser)
	if lIsSp && rIsSp {
		nnz = lSp.NNZ() + rSp.NNZ()
	} else {
		// assume 10% of elements will be non-zero
		nnz = a.Len() / 10
	}

	if v.checkOverlap(a) || v.checkOverlap(b) {
		if !v.IsZero() && a.Len() != v.len {
			panic(mat.ErrShape)
		}
		t, restore = v.temporaryWorkspace(a.Len(), nnz, true)
		isTemp = true
	} else {
		v.reuseAs(a.Len(), nnz, true)
	}

	return
}

// MulMatSparseVec (c = alpha * a * v + c) multiplies a dense matrix by a sparse
// vector and stores the result in mat.VecDense.  c is a *mat.VecDense, if c is nil,
// a new mat.VecDense of the correct size will be allocated and returned as the
// result from the function.  a*v will be scaled by alpha.  The function will
// panic if ac != |v| or if (C != nil and |c| != ar).
// Note this is not a Sparse BLAS routine -- that library does not cover this
// case.  This is a lookalike function in the Sparse BLAS style.  As a and c are
// dense there is limited benefit to including alpha and c; this is done for
// consistency rather than performance.
func MulMatSparseVec(alpha float64, a mat.Matrix, v *Vector, c *mat.VecDense) *mat.VecDense {
	rows, cols := a.Dims()
	if cols != v.Len() {
		panic(mat.ErrShape)
	}
	if c == nil {
		c = mat.NewVecDense(rows, nil)
	} else {
		if c.Len() != rows {
			panic(mat.ErrShape)
		}
	}
	res := mat.NewVecDense(rows, nil)
	// if a has RowView() we use that and sparse.Dot
	if rv, aIsRowViewer := a.(mat.RowViewer); aIsRowViewer {
		for row := 0; row < rows; row++ {
			thisRow := rv.RowView(row)
			res.SetVec(row, Dot(thisRow, v))
		}
	} else {
		// otherwise can only rely on At()
		for row := 0; row < rows; row++ {
			thisVal := 0.0
			for i, col := range v.ind {
				thisVal += a.At(row, col) * v.data[i]
			}
			res.SetVec(row, thisVal)
		}
	}
	c.AddScaledVec(c, alpha, res)
	return c
}

type indexPair struct {
	index int
	value float64
}

// Sort the entries in a vector.
func (v *Vector) Sort() {
	if v.IsSorted() {
		return
	}
	pairs := make([]indexPair, len(v.ind))
	for i, idx := range v.ind {
		pairs[i].index = idx
		pairs[i].value = v.data[i]
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].index < pairs[j].index
	})
	for i, p := range pairs {
		v.ind[i] = p.index
		v.data[i] = p.value
	}
}

// IsSorted checks if the vector is stored in sorted order
func (v *Vector) IsSorted() bool {
	return sort.IntsAreSorted(v.ind)
}

// dotSparseSparse computes the dot product of two sparse vectors.
// This will be called by Dot if both entered vectors are Sparse.
func dotSparseSparse(a, b, c *Vector) float64 {
	a.Sort()
	b.Sort()
	return dotSparseSparseNoSort(a, b, c)
}

func dotSparseSparseNoSort(a, b, c *Vector) float64 {
	n := a.Len()
	return dotSparseSparseNoSortBefore(a, b, c, n)
}

func dotSparseSparseNoSortBefore(a, b, c *Vector, n int) float64 {
	v, _, _ := dotSparseSparseNoSortBeforeWithStart(a, b, c, n, 0, 0)
	return v
}

func dotSparseSparseNoSortBeforeWithStart(a, b, c *Vector, n, aStart, bStart int) (float64, int, int) {
	tot := 0.0
	aPos := aStart
	bPos := bStart
	aIndex := -1
	bIndex := -1
	for aPos < len(a.ind) && bPos < len(b.ind) && aIndex < n && bIndex < n {
		aIndex = a.ind[aPos]
		bIndex = b.ind[bPos]
		if aIndex == bIndex {
			val := a.data[aPos] * b.data[bPos]
			tot += val
			if c != nil {
				c.SetVec(aIndex, val)
			}
			aPos++
			bPos++
		} else {
			if aIndex < bIndex {
				aPos++
			} else {
				bPos++
			}
		}
	}
	return tot, aPos, bPos
}

// MulElemVec does element-by-element multiplication of a and b
// and puts the result in the receiver.
func (v *Vector) MulElemVec(a, b mat.Vector) {
	ar := a.Len()
	br := b.Len()
	if ar != br {
		panic(mat.ErrShape)
	}

	as, aIsSparse := a.(*Vector)
	bs, bIsSparse := b.(*Vector)

	if aIsSparse {
		aNNZ := as.NNZ()
		if bIsSparse {
			bNNZ := bs.NNZ()
			minNNZ := aNNZ
			if bNNZ < minNNZ {
				minNNZ = bNNZ
			}
			if v != nil {
				v.reuseAs(ar, minNNZ, true)
			}
			dotSparseSparse(as, bs, v)
		} else {
			if v != nil {
				v.reuseAs(ar, aNNZ, true)
			}
			dotSparse(as, b, v)
		}
	} else if bIsSparse {
		bNNZ := bs.NNZ()
		if v != nil {
			v.reuseAs(ar, bNNZ, true)
		}
		dotSparse(bs, a, v)
	} else {
		v.reuseAs(ar, ar, true)
		for i := 0; i < ar; i++ {
			v.SetVec(i, a.AtVec(i)*b.AtVec(i))
		}
	}
}
