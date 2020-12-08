package sparse

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVectorAdd(t *testing.T) {
	tests := []struct {
		a mat.Vector
		b mat.Vector
		r mat.Vector
	}{
		{
			a: mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			b: mat.NewVecDense(6, []float64{1, 1, 0, 1, 0, 0}),
			r: mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
		{
			a: NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: NewVector(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r: mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
	}

	for ti, test := range tests {
		var result Vector

		result.AddVec(test.a, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for add - expected:\n%v\nbut received:\n%v\n", ti+1, mat.Formatted(test.r), mat.Formatted(&result))
		}
	}
}

func TestVectorAddScaled(t *testing.T) {
	tests := []struct {
		a     mat.Vector
		alpha float64
		b     mat.Vector
		r     mat.Vector
	}{
		{
			a:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			alpha: 0,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 0, 0}),
			r:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
		},
		{
			a:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			alpha: 1,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 0, 0}),
			r:     mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
		{
			a:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			alpha: 2,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 0, 0}),
			r:     mat.NewVecDense(6, []float64{2, 3, 0, 4, 1, 0}),
		},
		{
			a:     NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 0,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
		},
		{
			a:     NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 1,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
		{
			a:     NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 2,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{2, 3, 0, 4, 1, 0}),
		},
	}

	for ti, test := range tests {
		var result Vector

		result.AddScaledVec(test.a, test.alpha, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for addScaled - expected:\n%v\nbut received:\n%v\n", ti+1, mat.Formatted(test.r), mat.Formatted(&result))
		}
	}
}

func TestVectorScale(t *testing.T) {
	tests := []struct {
		alpha float64
		b     mat.Vector
		r     mat.Vector
	}{
		{
			alpha: 0,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 0, 0}),
			r:     mat.NewVecDense(6, []float64{0, 0, 0, 0, 0, 0}),
		},
		{
			alpha: 1,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 3, 0}),
			r:     mat.NewVecDense(6, []float64{1, 1, 0, 1, 3, 0}),
		},
		{
			alpha: 2,
			b:     mat.NewVecDense(6, []float64{1, 1, 0, 2, 1, 0}),
			r:     mat.NewVecDense(6, []float64{2, 2, 0, 4, 2, 0}),
		},
		{
			alpha: 0,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{0, 0, 0, 0, 0, 0}),
		},
		{
			alpha: 1,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 2, 4}),
			r:     mat.NewVecDense(6, []float64{1, 2, 0, 4, 0, 0}),
		},
		{
			alpha: 2,
			b:     NewVector(6, []int{0, 1, 3}, []float64{1, 2, 1}),
			r:     mat.NewVecDense(6, []float64{2, 4, 0, 2, 0, 0}),
		},
	}

	for ti, test := range tests {
		var result Vector

		result.ScaleVec(test.alpha, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for Scale - expected:\n%v\nbut received:\n%v\n", ti+1, mat.Formatted(test.r), mat.Formatted(&result))
		}
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		a mat.Vector
		b mat.Vector
		r float64
	}{
		{
			a: mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			b: mat.NewVecDense(6, []float64{1, 1, 0, 2, 0, 0}),
			r: 5,
		},
		{
			a: NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: NewVector(6, []int{0, 1, 3}, []float64{1, 1, 2}),
			r: 5,
		},
		{
			a: NewVector(6, []int{4, 3, 1}, []float64{1, 2, 1}),
			b: NewVector(6, []int{1, 0, 3}, []float64{1, 1, 2}),
			r: 5,
		},
		{
			a: mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
			b: NewVector(6, []int{0, 1, 3}, []float64{1, 1, 2}),
			r: 5,
		},
		{
			a: NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: mat.NewVecDense(6, []float64{1, 1, 0, 2, 0, 0}),
			r: 5,
		},
	}

	for ti, test := range tests {

		result := Dot(test.a, test.b)
		if result != test.r {
			t.Errorf("Test %d: Incorrect result for Dot - expected:\n%v\nbut received:\n%v\n", ti+1, test.r, result)
		}
	}
}

func TestMulElemVec(t *testing.T) {
	v := NewVector(6, nil, nil)
	tests := []struct {
		a *Vector
		b *Vector
		r *Vector
	}{
		{
			a: NewVector(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: NewVector(6, []int{0, 1, 3}, []float64{1, 1, 2}),
			r: NewVector(6, []int{1, 3}, []float64{1, 4}),
		},
		{
			a: NewVector(6, []int{4, 3, 1}, []float64{1, 2, 1}),
			b: NewVector(6, []int{1, 0, 3}, []float64{1, 1, 2}),
			r: NewVector(6, []int{1, 3}, []float64{1, 4}),
		},
	}

	for ti, test := range tests {

		v.MulElemVec(test.a, test.b)
		if !mat.Equal(v, test.r) {
			t.Errorf("Test %d: Incorrect result for MulElemVec - expected:\n%v\nbut received:\n%v\n", ti+1, test.r, v)
		}
	}
}

func TestVectorNorm(t *testing.T) {
	tests := []struct {
		a      mat.Vector
		result float64
	}{
		{
			a:      mat.NewVecDense(6, []float64{0, 1, 3, 0, 2, 0}),
			result: 3.741657386773941,
		},
		{
			a:      NewVector(6, []int{1, 2, 4}, []float64{1, 3, 2}),
			result: 3.7416573867739413,
		},
	}

	for ti, test := range tests {

		result := Norm(test.a, 2)

		if test.result != result {
			t.Errorf("Test %d: Incorrect result for Norm - expected:\n%v\nbut received:\n%v\n", ti+1, test.result, result)
		}
	}
}

func TestVectorDoNonZero(t *testing.T) {
	var tests = []struct {
		nnz  int
		data *Vector
	}{
		{
			nnz:  3,
			data: NewVector(6, []int{1, 2, 4}, []float64{1, 3, 2}),
		},
		{
			nnz:  0,
			data: NewVector(6, []int{}, []float64{}),
		},
	}

	for ti, test := range tests {
		var nnz int
		test.data.DoNonZero(func(i, j int, v float64) {
			if testv := test.data.At(i, j); testv == 0 || testv != v {
				t.Logf("test %d: Expected %f at (%d, %d) but received %f\n", ti+1, v, i, j, testv)
				t.Fail()
			}
			nnz++
		})

		if nnz != test.data.NNZ() {
			t.Logf("Test %d: Expected %d Non Zero elements but found %d", ti+1, nnz, test.data.NNZ())
			t.Fail()
		}
	}
}

func TestVecGather(t *testing.T) {
	tests := []struct {
		length   int
		ind      []int
		data     []float64
		src      []float64
		expected []float64
	}{
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{0, 0, 0},
			src:      []float64{1, 0, 2, 3, 4},
			expected: []float64{0, 0, 2, 0, 4},
		},
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{1, 1, 1},
			src:      []float64{1, 0, 2, 3, 4},
			expected: []float64{0, 0, 2, 0, 4},
		},
	}

	for ti, test := range tests {
		src := mat.NewVecDense(test.length, test.src)
		dst := NewVector(test.length, test.ind, test.data)
		dst.Gather(src)

		for i := 0; i < test.length; i++ {
			if test.expected[i] != dst.AtVec(i) {
				t.Errorf("Test %d: Mismatch at index %d, Expected %v but received %v", ti+1, i, test.expected[i], dst.AtVec(i))
			}
		}
	}
}

func TestVecGatherAndZero(t *testing.T) {
	tests := []struct {
		length   int
		ind      []int
		data     []float64
		src      []float64
		expected []float64
	}{
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{0, 0, 0},
			src:      []float64{1, 0, 2, 3, 4},
			expected: []float64{0, 0, 2, 0, 4},
		},
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{1, 1, 1},
			src:      []float64{1, 0, 2, 3, 4},
			expected: []float64{0, 0, 2, 0, 4},
		},
	}

	for ti, test := range tests {
		src := mat.NewVecDense(test.length, test.src)
		dst := NewVector(test.length, test.ind, test.data)
		dst.GatherAndZero(src)

		for i := 0; i < test.length; i++ {
			if test.expected[i] != dst.AtVec(i) {
				t.Errorf("Test %d: Mismatch at index %d, Expected %v but received %v", ti+1, i, test.expected[i], dst.AtVec(i))
			}
		}
		for _, v := range test.ind {
			if src.AtVec(v) != 0 {
				t.Errorf("Test %d: Expected 0 at index %d but found %v", ti+1, v, src.AtVec(v))
			}
		}
	}
}

func TestVecScatter(t *testing.T) {
	tests := []struct {
		length   int
		ind      []int
		data     []float64
		dst      *mat.VecDense
		expected []float64
	}{
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{1, 2, 3},
			dst:      mat.NewVecDense(5, []float64{0, 0, 0, 0, 0}),
			expected: []float64{0, 1, 2, 0, 3},
		},
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{1, 1, 1},
			dst:      mat.NewVecDense(5, []float64{0, 0, 2, 3, 4}),
			expected: []float64{0, 1, 1, 3, 1},
		},
	}

	for ti, test := range tests {
		src := NewVector(test.length, test.ind, test.data)
		result := src.Scatter(test.dst)

		for i := 0; i < test.length; i++ {
			if test.dst != nil && test.expected[i] != test.dst.AtVec(i) {
				t.Errorf("Test %d: Mismatch at index %d, Expected %v but received %v", ti+1, i, test.expected[i], test.dst.AtVec(i))
			}
			if test.expected[i] != result.AtVec(i) {
				t.Errorf("Test %d: Mismatch result at index %d, Expected %v but received %v", ti+1, i, test.expected[i], result.AtVec(i))
			}
		}
	}
}

func TestCloneVec(t *testing.T) {
	basicSparse := NewVector(5, []int{0, 2, 3}, []float64{1, 2, 3})

	tests := []struct {
		src      mat.Vector
		rcv      *Vector
		expected *Vector
	}{
		{ // test src == rcv
			src:      basicSparse,
			rcv:      basicSparse,
			expected: basicSparse,
		},
		{ // test rcv empty
			src:      basicSparse,
			rcv:      &Vector{},
			expected: basicSparse,
		},
		{ // test src capacity > rcv
			src:      basicSparse,
			rcv:      NewVector(4, []int{1}, []float64{1}),
			expected: basicSparse,
		},
		{ // test src capacity < rcv
			src:      basicSparse,
			rcv:      NewVector(7, []int{0, 1, 3, 4}, []float64{1, 2, 3, 4}),
			expected: basicSparse,
		},
		{ // test src as dense vector
			src:      mat.NewVecDense(5, []float64{1, 0, 2, 3, 0}),
			rcv:      basicSparse,
			expected: basicSparse,
		},
	}

	for ti, test := range tests {
		test.rcv.CloneVec(test.src)

		if test.expected.Len() != test.rcv.Len() {
			t.Errorf("Test %d: Expected length of %d but received %d", ti, test.expected.Len(), test.rcv.Len())
		}

		for i := 0; i < test.expected.Len(); i++ {
			if test.expected.AtVec(i) != test.rcv.AtVec(i) {
				t.Errorf("Test %d: Expected %f at index %d but received %f", ti, test.expected.AtVec(i), i, test.rcv.AtVec(i))
			}
		}
	}
}

func TestVecToVecDense(t *testing.T) {
	tests := []struct {
		length   int
		ind      []int
		data     []float64
		expected []float64
	}{
		{
			length:   5,
			ind:      []int{1, 2, 4},
			data:     []float64{1, 2, 3},
			expected: []float64{0, 1, 2, 0, 3},
		},
	}

	for ti, test := range tests {
		src := NewVector(test.length, test.ind, test.data)
		result := src.ToDense()

		for i := 0; i < test.length; i++ {
			if test.expected[i] != result.AtVec(i) {
				t.Errorf("Test %d: Mismatch result at index %d, Expected %v but received %v", ti+1, i, test.expected[i], result.AtVec(i))
			}
		}
	}
}

func TestVectorSetVec(t *testing.T) {
	tests := []struct {
		idx      int
		val      float64
		source   *Vector
		expected *Vector
	}{
		{
			idx:      0,
			val:      0.1,
			source:   NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
			expected: NewVector(5, []int{0, 1, 2, 4}, []float64{0.1, 1.1, 2.2, 4.4}),
		},
		{
			idx:      3,
			val:      3.3,
			source:   NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
			expected: NewVector(5, []int{1, 2, 3, 4}, []float64{1.1, 2.2, 3.3, 4.4}),
		},
		{
			idx:      4,
			val:      4.4,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 2, 3, 4}, []float64{1.1, 2.2, 3.3, 4.4}),
		},
		{
			idx:      2,
			val:      8.8,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 2, 3}, []float64{1.1, 8.8, 3.3}),
		},
		{
			idx:      1,
			val:      0.0,
			source:   NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
			expected: NewVector(5, []int{2, 4}, []float64{2.2, 4.4}),
		},
		{
			idx:      2,
			val:      0.0,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 3}, []float64{1.1, 3.3}),
		},
		{
			idx:      3,
			val:      0.0,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 2}, []float64{1.1, 2.2}),
		},
	}

	for ti, test := range tests {
		act := new(Vector)
		act.CloneVec(test.source)
		act.SetVec(test.idx, test.val)

		if len(test.expected.ind) != len(act.ind) {
			t.Errorf("Test %d: Incorrect count for indices, Expected %v but received %v", ti+1, len(test.expected.ind), len(act.ind))
		} else if len(test.expected.data) != len(act.data) {
			t.Errorf("Test %d: Incorrect count for data, Expected %v but received %v", ti+1, len(test.expected.data), len(act.data))
		} else {
			for i := 0; i < len(test.expected.ind); i++ {
				if test.expected.ind[i] != act.ind[i] {
					t.Errorf("Test %d: Mismatch indice at index %d, Expected %v but received %v", ti+1, i, test.expected.ind[i], act.ind[i])
				}
			}
			for i := 0; i < len(test.expected.data); i++ {
				if test.expected.data[i] != act.data[i] {
					t.Errorf("Test %d: Mismatch data at index %d, Expected %v but received %v", ti+1, i, test.expected.data[i], act.data[i])
				}
			}
		}
	}
}

func TestVectorSet(t *testing.T) {
	tests := []struct {
		idx      int
		val      float64
		source   *Vector
		expected *Vector
	}{
		{
			idx:      0,
			val:      0.1,
			source:   NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
			expected: NewVector(5, []int{0, 1, 2, 4}, []float64{0.1, 1.1, 2.2, 4.4}),
		},
		{
			idx:      3,
			val:      3.3,
			source:   NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
			expected: NewVector(5, []int{1, 2, 3, 4}, []float64{1.1, 2.2, 3.3, 4.4}),
		},
		{
			idx:      4,
			val:      4.4,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 2, 3, 4}, []float64{1.1, 2.2, 3.3, 4.4}),
		},
		{
			idx:      2,
			val:      8.8,
			source:   NewVector(5, []int{1, 2, 3}, []float64{1.1, 2.2, 3.3}),
			expected: NewVector(5, []int{1, 2, 3}, []float64{1.1, 8.8, 3.3}),
		},
	}

	for ti, test := range tests {
		act := new(Vector)
		act.CloneVec(test.source)
		act.Set(test.idx, 0, test.val)

		if len(test.expected.ind) != len(act.ind) {
			t.Errorf("Test %d: Incorrect count for indices, Expected %v but received %v", ti+1, len(test.expected.ind), len(act.ind))
		} else if len(test.expected.data) != len(act.data) {
			t.Errorf("Test %d: Incorrect count for data, Expected %v but received %v", ti+1, len(test.expected.data), len(act.data))
		} else {
			for i := 0; i < len(test.expected.ind); i++ {
				if test.expected.ind[i] != act.ind[i] {
					t.Errorf("Test %d: Mismatch indice at index %d, Expected %v but received %v", ti+1, i, test.expected.ind[i], act.ind[i])
				}
			}
			for i := 0; i < len(test.expected.data); i++ {
				if test.expected.data[i] != act.data[i] {
					t.Errorf("Test %d: Mismatch data at index %d, Expected %v but received %v", ti+1, i, test.expected.data[i], act.data[i])
				}
			}
		}
	}
}

func TestVecSetVecPanic(t *testing.T) {
	tests := []struct {
		idx    int
		source *Vector
	}{
		{
			idx:    -1,
			source: NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
		},
		{
			idx:    5,
			source: NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
		},
		{
			idx:    6,
			source: NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
		},
	}

	for ti, test := range tests {
		func(ti int, idx int, src *Vector) {
			defer func() {
				msg := recover()
				if msg == nil {
					t.Errorf("Test %d: method did not panic as epected", ti+1)
				} else {
					if msg != mat.ErrRowAccess {
						t.Errorf("Test %d: recovered error was not mat.ErrRowAccess", ti+1)
					}
				}
			}()

			act := new(Vector)
			act.CloneVec(test.source)
			act.SetVec(idx, 1.1)

		}(ti, test.idx, test.source)
	}
}

func TestVecSetPanic(t *testing.T) {
	tests := []struct {
		col    int
		source *Vector
	}{
		{
			col:    -1,
			source: NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
		},
		{
			col:    1,
			source: NewVector(5, []int{1, 2, 4}, []float64{1.1, 2.2, 4.4}),
		},
	}

	for ti, test := range tests {
		func(ti int, col int, src *Vector) {
			defer func() {
				msg := recover()
				if msg == nil {
					t.Errorf("Test %d: method did not panic as epected", ti+1)
				} else {
					if msg != mat.ErrColAccess {
						t.Errorf("Test %d: recovered error was not mat.ErrColAccess", ti+1)
					}
				}
			}()

			act := new(Vector)
			act.CloneVec(test.source)
			act.Set(2, col, 1.1)

		}(ti, test.col, test.source)
	}
}

func TestMulMatSparseVec(t *testing.T) {
	permsB := []struct {
		name   string
		vector Vector
	}{
		{
			name:   "Sparse",
			vector: *NewVector(4, []int{0, 1, 3}, []float64{1, 2, 3}),
		},
	}

	tests := []struct {
		alpha  float64
		y      *mat.VecDense
		er, ec int
		eData  []float64
	}{
		{ // y = 0 * A * x
			alpha: 0,
			y:     mat.NewVecDense(3, []float64{0, 0, 0}),
			er:    3, ec: 1,
			eData: []float64{0, 0, 0},
		},
		{ // y = 1 * A * x
			alpha: 1,
			y:     mat.NewVecDense(3, []float64{0, 0, 0}),
			er:    3, ec: 1,
			eData: []float64{1, 0, 21},
		},
		{ // y = 2 * A * x
			alpha: 2,
			y:     mat.NewVecDense(3, []float64{0, 0, 0}),
			er:    3, ec: 1,
			eData: []float64{2, 0, 42},
		},
		{ // y = 1 * A * x + y
			alpha: 1,
			y:     mat.NewVecDense(3, []float64{1, 2, 0}),
			er:    3, ec: 1,
			eData: []float64{2, 2, 21},
		},
		{ // y = 1 * A * x + y
			alpha: 1,
			y:     nil,
			er:    3, ec: 1,
			eData: []float64{1, 0, 21},
		},
	}

	for ti, test := range tests {
		for _, b := range permsB {
			for _, rawa := range matrixPermutationsForA {
				var matPair = []struct {
					name   string
					matrix mat.Matrix
				}{
					{
						name:   rawa.name,
						matrix: rawa.matrix,
					},
					{
						name:   "ToDense",
						matrix: rawa.matrix.(TypeConverter).ToDense(),
					},
				}
				for _, a := range matPair {
					amat := a.matrix
					amatCopy := mat.DenseCopyOf(amat)
					var ycopy *mat.VecDense
					if test.y != nil {
						ycopy = mat.VecDenseCopyOf(test.y)
					}
					y := MulMatSparseVec(test.alpha, amat, &b.vector, ycopy)

					yr, yc := y.Dims()
					if yr != test.er || yc != test.ec {
						t.Errorf("Test %d (%s x %s): Incorrect dimensions: expected %d x %d but received %d x %d", ti+1, a.name, b.name, test.er, test.ec, yr, yc)
					}

					yraw := y.RawVector()

					for i, v := range test.eData {
						if v != yraw.Data[i] {
							e := mat.NewDense(test.er, test.ec, test.eData)
							t.Errorf("Test %d (%s x %s): Failed, expected\n%v\n but received \n%v", ti+1, a.name, b.name, mat.Formatted(e), mat.Formatted(y))
							break
						}
					}
					if !mat.Equal(amat, amatCopy) {
						t.Error("somehow matrix changed")
					}
				}
			}
		}
	}
}

func TestReset(t *testing.T) {
	tests := []struct {
		a *Vector
	}{
		{
			a: NewVector(6, []int{1, 2, 4}, []float64{1, 3, 2}),
		},
	}

	for ti, test := range tests {
		test.a.Reset()
		b := NewVector(test.a.Len(), []int{}, []float64{})

		if !mat.Equal(test.a, b) {
			t.Errorf("Test %d: Incorrect result for Reset - expected:\n%v\nbut received:\n%v\n", ti+1, b, test.a)
		}
	}
}
