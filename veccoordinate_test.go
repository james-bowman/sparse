package sparse

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestVecCOOAdd(t *testing.T) {
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
			a: NewVecCOO(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r: mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
	}

	for ti, test := range tests {
		var result VecCOO

		result.AddVec(test.a, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for add - expected:\n%v\nbut received:\n%v\n", ti, mat.Formatted(test.r), mat.Formatted(&result))
		}
	}
}

func TestVecCOOAddScaled(t *testing.T) {
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
			a:     NewVecCOO(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 0,
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{0, 1, 0, 2, 1, 0}),
		},
		{
			a:     NewVecCOO(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 1,
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{1, 2, 0, 3, 1, 0}),
		},
		{
			a:     NewVecCOO(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			alpha: 2,
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{2, 3, 0, 4, 1, 0}),
		},
	}

	for ti, test := range tests {
		var result VecCOO

		result.AddScaledVec(test.a, test.alpha, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for addScaled - expected:\n%v\nbut received:\n%v\n", ti, mat.Formatted(test.r), mat.Formatted(&result))
		}
	}
}

func TestVecCOOScale(t *testing.T) {
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
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 1}),
			r:     mat.NewVecDense(6, []float64{0, 0, 0, 0, 0, 0}),
		},
		{
			alpha: 1,
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 2, 4}),
			r:     mat.NewVecDense(6, []float64{1, 2, 0, 4, 0, 0}),
		},
		{
			alpha: 2,
			b:     NewVecCOO(6, []int{0, 1, 3}, []float64{1, 2, 1}),
			r:     mat.NewVecDense(6, []float64{2, 4, 0, 2, 0, 0}),
		},
	}

	for ti, test := range tests {
		var result VecCOO

		result.ScaleVec(test.alpha, test.b)

		if !mat.Equal(test.r, &result) {
			t.Errorf("Test %d: Incorrect result for Scale - expected:\n%v\nbut received:\n%v\n", ti, mat.Formatted(test.r), mat.Formatted(&result))
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
			a: NewVecCOO(6, []int{1, 3, 4}, []float64{1, 2, 1}),
			b: NewVecCOO(6, []int{0, 1, 3}, []float64{1, 1, 2}),
			r: 5,
		},
	}

	for ti, test := range tests {

		result := Dot(test.a, test.b)

		if result != test.r {
			t.Errorf("Test %d: Incorrect result for Dot - expected:\n%v\nbut received:\n%v\n", ti, test.r, result)
		}
	}
}

func TestVecCOONorm(t *testing.T) {
	tests := []struct {
		a      mat.Vector
		result float64
	}{
		{
			a:      mat.NewVecDense(6, []float64{0, 1, 3, 0, 2, 0}),
			result: 3.741657386773941,
		},
		{
			a:      NewVecCOO(6, []int{1, 2, 4}, []float64{1, 3, 2}),
			result: 3.7416573867739413,
		},
	}

	for ti, test := range tests {

		result := Norm(test.a, 2)

		if test.result != result {
			t.Errorf("Test %d: Incorrect result for Norm - expected:\n%v\nbut received:\n%v\n", ti, test.result, result)
		}
	}
}

func TestVecCOODoNonZero(t *testing.T) {
	var tests = []struct {
		nnz  int
		data *VecCOO
	}{
		{
			nnz:  3,
			data: NewVecCOO(6, []int{1, 2, 4}, []float64{1, 3, 2}),
		},
		{
			nnz:  0,
			data: NewVecCOO(6, []int{}, []float64{}),
		},
	}

	for ti, test := range tests {
		var nnz int
		test.data.DoNonZero(func(i, j int, v float64) {
			if testv := test.data.At(i, j); testv == 0 || testv != v {
				t.Logf("test %d: Expected %f at (%d, %d) but received %f\n", ti, v, i, j, testv)
				t.Fail()
			}
			nnz++
		})

		if nnz != test.data.NNZ() {
			t.Logf("Test %d: Expected %d Non Zero elements but found %d", ti, nnz, test.data.NNZ())
			t.Fail()
		}
	}
}
