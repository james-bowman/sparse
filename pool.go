package sparse

import (
	"sync"

	"github.com/james-bowman/sparse/blas"
)

const (
	pooledFloatSize = 200
	pooledIntSize   = 200
)

var (
	pool = sync.Pool{
		New: func() interface{} {
			return &CSR{
				matrix: blas.SparseMatrix{},
			}
		},
	}
	vecPool = sync.Pool{
		New: func() interface{} {
			return &Vector{}
		},
	}
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

func getWorkspace(r, c, nnz int, clear bool) *CSR {
	w := pool.Get().(*CSR)
	w.matrix.Indptr = useInts(w.matrix.Indptr, r+1, false)
	w.matrix.Ind = useInts(w.matrix.Ind, nnz, false)
	w.matrix.Data = useFloats(w.matrix.Data, nnz, false)
	if clear {
		for i := range w.matrix.Indptr {
			w.matrix.Indptr[i] = 0
		}
		w.matrix.Ind = w.matrix.Ind[:0]
		w.matrix.Data = w.matrix.Data[:0]
	}
	w.matrix.I = r
	w.matrix.J = c
	return w
}

func putWorkspace(w *CSR) {
	pool.Put(w)
}

func getVecWorkspace(len, nnz int, clear bool) *Vector {
	w := vecPool.Get().(*Vector)
	w.ind = useInts(w.ind, nnz, false)
	w.data = useFloats(w.data, nnz, false)
	if clear {
		w.ind = w.ind[:0]
		w.data = w.data[:0]
	}
	w.len = len
	return w
}

func putVecWorkspace(w *Vector) {
	vecPool.Put(w)
}

// getFloats returns a []float64 of length l. If clear is true,
// the slice visible is zeroed.
func getFloats(l int, clear bool) []float64 {
	w := floatPool.Get().([]float64)
	return useFloats(w, l, clear)
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
	return useInts(w, l, clear)
}

// putInts replaces a used []int into the pool.
func putInts(w []int) {
	if cap(w) > pooledIntSize {
		intPool.Put(w)
	}
}
