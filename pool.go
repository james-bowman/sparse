package sparse

import (
	"sync"
)

const (
	pooledFloatSize = 200
	pooledIntSize   = 200
)

var (
	floatPool = sync.Pool{
		New: func() interface{} {
			return make([]float64, pooledFloatSize)
		},
	}
	intPool = sync.Pool{
		New: func() interface{} {
			return make([]int, pooledIntSize)
		},
	}
)

// getFloats returns a []float64 of length l. If clear is true,
// the slice visible is zeroed.
func getFloats(l int, clear bool) []float64 {
	w := floatPool.Get().([]float64)
	if l <= cap(w) {
		w = w[:l]
		if clear {
			for i := range w {
				w[i] = 0
			}
		}
		return w
	}
	// []float from pool is too small so return and create a
	// bigger one
	//putFloats(w)
	w = make([]float64, l)
	return w
}

// putFloats replaces a used []float64 into the appropriate size
// workspace pool. putFloats must not be called with a slice
// where references to the underlying data have been kept.
func putFloats(w []float64) {
	if cap(w) > pooledFloatSize {
		floatPool.Put(w)
	}
}

// getInts returns a []ints of length l. If clear is true,
// the slice visible is zeroed.
func getInts(l int, clear bool) []int {
	w := intPool.Get().([]int)
	if l <= cap(w) {
		w = w[:l]
		if clear {
			for i := range w {
				w[i] = 0
			}
		}
		return w
	}
	// []float from pool is too small so return and create a
	// bigger one
	//putInts(w)
	w = make([]int, l)
	return w
}

// putInts replaces a used []int into the pool.
func putInts(w []int) {
	if cap(w) > pooledIntSize {
		intPool.Put(w)
	}
}
