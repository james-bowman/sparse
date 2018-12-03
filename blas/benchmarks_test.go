package blas

import (
	"fmt"
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/stat/sampleuv"
)

func BenchmarkAxpy(b *testing.B) {
	nnz := 1000
	dim := 10000
	x := make([]float64, nnz)
	indx := make([]int, nnz)
	y := make([]float64, (dim)*(dim))

	rnd := rand.New(rand.NewSource(0))

	sampleuv.WithoutReplacement(indx, dim, rnd)

	for i := range x {
		x[i] = rnd.Float64()
	}

	for i := range y {
		y[i] = rnd.Float64()
	}

	inputs := []struct {
		name  string
		alpha float64
		x     []float64
		indx  []int
		y     []float64
		incy  int
	}{
		{
			name:  "inc",
			alpha: 1,
			x:     x,
			indx:  indx,
			y:     y,
			incy:  dim,
		},
		{
			name:  "unitary",
			alpha: 1,
			x:     x,
			indx:  indx,
			y:     y[:dim],
			incy:  1,
		},
	}

	benchmarks := []struct {
		name string
		f    func(alpha float64, x []float64, indx []int, y []float64, incy int)
	}{
		{
			name: "Naive Go",
			f: func(alpha float64, x []float64, indx []int, y []float64, incy int) {
				for i, index := range indx {
					y[index*incy] += alpha * x[i]
				}
			},
		},
		{
			name: "Asm",
			f:    Dusaxpy,
		},
	}

	for _, input := range inputs {
		for _, bench := range benchmarks {
			b.Run(fmt.Sprintf("%s %s", input.name, bench.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					bench.f(input.alpha, input.x, input.indx, input.y, input.incy)
				}
			})
		}
	}
}

func BenchmarkDot(b *testing.B) {
	nnz := 1000
	dim := 10000
	x := make([]float64, nnz)
	indx := make([]int, nnz)
	y := make([]float64, (dim)*(dim))

	rnd := rand.New(rand.NewSource(0))

	sampleuv.WithoutReplacement(indx, dim, rnd)

	for i := range x {
		x[i] = rnd.Float64()
	}

	for i := range y {
		y[i] = rnd.Float64()
	}

	inputs := []struct {
		name string
		x    []float64
		indx []int
		y    []float64
		incy int
	}{
		{
			name: "inc",
			x:    x,
			indx: indx,
			y:    y,
			incy: dim,
		},
		{
			name: "unitary",
			x:    x,
			indx: indx,
			y:    y[:dim],
			incy: 1,
		},
	}

	benchmarks := []struct {
		name string
		f    func(x []float64, indx []int, y []float64, incy int) float64
	}{
		{
			name: "Naive Go",
			f: func(x []float64, indx []int, y []float64, incy int) (dot float64) {
				for i, index := range indx {
					dot += x[i] * y[index*incy]
				}
				return
			},
		},
		{
			name: "Asm",
			f:    Dusdot,
		},
	}

	for _, input := range inputs {
		for _, bench := range benchmarks {
			b.Run(fmt.Sprintf("%s %s", input.name, bench.name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					bench.f(input.x, input.indx, input.y, input.incy)
				}
			})
		}
	}
}
