package sparse

import (
	"fmt"
	"bytes"
	"gonum.org/v1/gonum/mat"
)

const (
	// maxLen is the biggest slice/array len one can create on a 32/64b platform.
	//bitLen = ^uint(0) >> 1
	maxDataLen = ^uint(0)
	
	// the wordSize of a binary vector.  Bits are stored in slices of words.
	wordSize = uint(64)

	// log2WordSize is the binary logarithm (log base 2) of (wordSize)
	log2WordSize = uint(6)
)

// BinaryVec is a Binary Vector or Bit Vector type.  This is useful
// for representing vectors of features with a binary state (1 or 0).
// Although part of the sparse package, this type is not sparse itself
// and stores all bits even 0s.  However, as it makes use of 64 bit
// integers to store each set of 64 bits and then bitwise operators to 
// manipulate the elements it will be more efficient in terms of both 
// storage requirements and performance than a slice of bool values 
// (8 bits per element) or even a typical Dense matrix of float64
// elements.  A compressed bitmap scheme could be used to take advantage 
// of sparseness but may have an associated overhead.
type BinaryVec struct {
	length int
	data []uint64
}

// New creates a new BitSet with a hint that length bits will be required
func NewBinaryVec(length int) *BinaryVec {
	maxSize := (maxDataLen - wordSize + uint(1))
	if uint(length) > maxSize {
		panic(fmt.Errorf("sparse: Requested bit length of Binary vector (%d) too large.  %d is the maximum allowed", length, maxSize))
	}
	elements := int((uint(length) + (wordSize - 1)) >> log2WordSize)

	vec := &BinaryVec{
		length: length,
		data: make([]uint64, elements),
	}

	return vec
}

// DistanceFrom is the number of bits that are different between the
// receiver and rhs i.e.
// 	recevier 	= 1001001
//	rhs 		= 1010101 
// 	Distance	= 3
// because there are three bits that are different between the 2
// binary vectors.  This is sometimes referred to as the `Hamming
// distance` or `Matching distance`.  In this case, the distance
// is not normalised and is simply the raw count of differences.  To
// normalise the value simply divide this value by the vector's length.
func (b *BinaryVec) DistanceFrom(rhs *BinaryVec) int {
	differences := uint64(0)
	for i, word := range b.data {
		differences += popcount(word ^ rhs.data[i])
	}
	return int(differences)
}

// Dims returns the dimensions of the matrix as the number of rows, columns.  
// As this is a vector, the second value representing the number of columns 
// will be 1. This method is part of the Gonum mat.Matrix interface
func (b *BinaryVec) Dims() (int, int) {
	return b.Len(), 1
}

// At returns the value of the element at row i and column j.
// As this is a vector (only one column), j must be 0 otherwise the 
// method panics.  This method is part of the Gonum mat.Matrix interface.
func (b *BinaryVec) At(i, j int) float64 {
	if uint(i) < 0 || uint(i) >= uint(b.length) {
		panic(mat.ErrRowAccess)
	}
	if uint(j) != 0 {
		panic(mat.ErrColAccess)
	}

	if b.data[i>>log2WordSize] & (1 << (uint(i) & (wordSize - 1))) != 0 {
		return 1.0
	}
	return 0.0
}

// T performs an implicit transpose by returning the receiver inside a Transpose.
// This method is part of the Gonum mat.Matrix interface
func (b *BinaryVec) T() mat.Matrix {
	return mat.Transpose{Matrix: b}
}

// NNZ returns the Number of Non-Zero elements (bits).  This is the number of set
// bits (represented by 1s rather than 0s) in the vector.  This is also known as the
// `Hamming weight` or `population count` (popcount).
func (b *BinaryVec) NNZ() int {
	nnz := uint64(0)
	for _, word := range b.data {
		nnz += popcount(word)
	}
	return int(nnz)
}

// Len returns the length of the vector or the total number of elements.  This method
// is part of the Gonum mat.Vector interface
func (b *BinaryVec) Len() int {
	return b.length
}

// SetBit sets the bit at the specified index (i) to 1.  If the bit is already set
// there are no adverse effects.  The method will panic if index is larger
// than Len()
func (b *BinaryVec) SetBit(i int) {
	if uint(i) < 0 || uint(i) >= uint(b.length) {
		panic(mat.ErrRowAccess)
	}
	b.data[i>>log2WordSize] |= 1 << (uint(i) & (wordSize - 1))
}

// UnsetBit unsets the bit at the specified index (i) (sets it to 0).  If the bit 
// is already unset or has simply never been set (default bit values are 0).
// there are no adverse effects.  The method will panic if index is larger
// than Len()
func (b *BinaryVec) UnsetBit(i int) {
	if uint(i) < 0 || uint(i) >= uint(b.length) {
		panic(mat.ErrRowAccess)
	}
	b.data[i>>log2WordSize] &^= 1 << (uint(i) & (wordSize - 1))
}

// Set sets the element of the matrix located at row i and column j to 1 if v != 0
// or 0 otherwise. Set will panic if specified values for i or j fall outside the 
// dimensions of the matrix.
func (b *BinaryVec) Set(i int, j int, v float64) {
	if uint(j) != 0 {
		panic(mat.ErrColAccess)
	}

	if v != 0 {
		b.SetBit(i)
		return
	}
	b.UnsetBit(i)
}

// String will output the vector as a string representation of its bits
// This method implements the fmt.Stringer interface.
func (b BinaryVec) String() string {
	var buf bytes.Buffer

	for i := len(b.data)-1; i >= 0; i-- {
		fmt.Fprintf(&buf, "%064b", b.data[i])
	}
	
	return string(buf.Bytes())
}

// Format outputs the vector to f and allows the output format
// to be specified.  Supported values of c are `x`, `X`, `b`` and `s`
// to format the bits of the vector as a hex digit or binary digit string.
// `s` (the default format) will output as binary digits.
// Please refer to the fmt package documentation for more information.
// This method implements the fmt.Formatter interface.
func (b BinaryVec) Format(f fmt.State, c rune) {
	var buf bytes.Buffer
	var format string
	switch c {
		case 'x':
			format = "%016x"		
		case 'X':
			format = "%016X"
		case 'b':
			format = "%064b"
		case 's':
			format = "%064b"
		default:
			panic(fmt.Errorf("sparse: unsupported format verb '%c' for Binary vector", c))
	}
	for i := len(b.data)-1; i >= 0; i-- {
		fmt.Fprintf(&buf, format, b.data[i])
	}
	f.Write(buf.Bytes())
}

// popcount calculates the population count of the vector (also known
// as `Hamming weight`).  This uses fewer arithmetic operations than
// any other known implementation on machines with fast multiplication.
// Thanks to Wikipedia and Hacker's Delight.
func popcount(x uint64) (n uint64) {
	x -= (x >> 1) & 0x5555555555555555
	x = (x>>2)&0x3333333333333333 + x&0x3333333333333333
	x += x >> 4
	x &= 0x0f0f0f0f0f0f0f0f
	x *= 0x0101010101010101
	return x >> 56
}

// Binary is a Binary Matrix or Bit Matrix type.
// Although part of the sparse package, this type is not sparse itself
// and stores all bits even 0s.  However, as it makes use of 64 bit
// integers to store each set of 64 bits and then bitwise operators to 
// manipulate the elements it will be more efficient in terms of both 
// storage requirements and performance than a slice of bool values 
// (8 bits per element) or even a typical Dense matrix of float64
// elements.  A compressed bitmap scheme could be used to take advantage 
// of sparseness but may have an associated overhead.
type Binary struct {
	r, c int
	cols []BinaryVec
}

// NewBinary constructs a new Binary matrix or r rows and c columns.
// If vecs is not nil, it will be used as the underlying binary column vectors.
// If vecs is nil, new storage will be allocated.
func NewBinary(r, c int, vecs []BinaryVec) *Binary {
	if vecs == nil {
		vecs = make([]BinaryVec, c)
		for i := 0; i < c; i++ {
			vecs[i] = *NewBinaryVec(r)
		}
	}

	return &Binary{r: r, c: c, cols: vecs}
}

// Dims returns the dimensions of the matrix as the number of rows, columns.  
// This method is part of the Gonum mat.Matrix interface
func (b *Binary) Dims() (int, int) {
	return b.r, b.c
}

// At returns the value of the element at row i and column k.
// i (row) and j (col) must be within the dimensions of the matrix otherwise the 
// method panics.  This method is part of the Gonum mat.Matrix interface.
func (b *Binary) At(i int, j int) float64 {
	if uint(j) < 0 || uint(j) >= uint(b.c) {
		panic(mat.ErrColAccess)
	}
	return b.cols[j].At(i, j)
}

// T performs an implicit transpose by returning the receiver inside a Transpose.
// This method is part of the Gonum mat.Matrix interface
func (b *Binary) T() mat.Matrix {
	return mat.Transpose{Matrix: b}
}

// ColView returns the mat.Vector representing the column j.  This vector will
// be a BinaryVec and will share the same storage as the matrix so any changes
// to the vector will be reflected in the matrix and vice versa.
// if j is outside the dimensions of the matrix the method will panic.
func (b *Binary) ColView(j int) mat.Vector {
	if uint(j) < 0 || uint(j) >= uint(b.c) {
		panic(mat.ErrColAccess)
	}
	return &b.cols[j]
}