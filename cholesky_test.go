package sparse

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCholesky(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		a *mat.SymDense

		cond   float64
		want   *mat.TriDense
		posdef bool
	}{
		{
			a: mat.NewSymDense(3, []float64{
				4, 1, 1,
				0, 2, 3,
				0, 0, 6,
			}),
			cond: 37,
			want: mat.NewTriDense(3, true, []float64{
				2, 0.5, 0.5,
				0, 1.3228756555322954, 2.0788046015507495,
				0, 0, 1.195228609334394,
			}),
			posdef: true,
		},
	} {
		if !cholMatches(test.a) {
			t.Error("chol mismatch")
		}
	}
	iters := 16
	src := rand.NewSource(1)
	for i := 0; i < iters; i++ {
		n := 128
		frac := 0.05
		m := randomSymDensePosDefinite(n, frac, src)
		if !cholMatches(m) {
			t.Error("mismatch on random matrix")
		}
	}
}

func TestCholeskySolveVecTo(t *testing.T) {
	t.Parallel()
	for idx, test := range []struct {
		a   *mat.SymDense
		b   *mat.VecDense
		ans *mat.VecDense
	}{
		{
			a: mat.NewSymDense(2, []float64{
				1, 0,
				0, 1,
			}),
			b:   mat.NewVecDense(2, []float64{5, 6}),
			ans: mat.NewVecDense(2, []float64{5, 6}),
		},
		{
			a: mat.NewSymDense(3, []float64{
				53, 59, 37,
				0, 83, 71,
				0, 0, 101,
			}),
			b:   mat.NewVecDense(3, []float64{5, 6, 7}),
			ans: mat.NewVecDense(3, []float64{0.20745069393718094, -0.17421475529583694, 0.11577794010226464}),
		},
	} {
		var chol mat.Cholesky
		ok := chol.Factorize(test.a)
		if !ok {
			t.Fatal("unexpected Cholesky factorization failure: not positive definite")
		}

		var x mat.VecDense
		err := chol.SolveVecTo(&x, test.b)
		if err != nil {
			t.Errorf("unexpected error from Cholesky solve: %v", err)
		}
		if !mat.EqualApprox(&x, test.ans, 1e-12) {
			t.Error("incorrect Cholesky solve solution")
		}

		var ans mat.VecDense
		ans.MulVec(test.a, &x)
		if !mat.EqualApprox(&ans, test.b, 1e-12) {
			t.Error("incorrect Cholesky solve solution product")
		}

		//		if !cholMatches(test.a) {
		//			t.Error("chol mismatch in solvevecto test")
		//		}

		n := test.a.Symmetric()
		aCOO := matToCOO(test.a, 1e-10)
		aCSR := aCOO.ToCSR()
		var sc Cholesky
		sc.Factorize(aCSR)
		xs := mat.NewVecDense(n, nil)
		sc.SolveVecTo(xs, test.b)
		if !mat.EqualApprox(xs, test.ans, 1e-12) {
			t.Error("incorrect sparse Cholesky solution", idx)
		}
	}
}

func TestCholeskyAt(t *testing.T) {
	t.Parallel()
	for _, test := range []*mat.SymDense{
		mat.NewSymDense(3, []float64{
			53, 59, 37,
			59, 83, 71,
			37, 71, 101,
		}),
	} {
		var chol Cholesky
		csr := matToCSR(test, 1e-8)
		chol.Factorize(csr)
		n := test.Symmetric()
		cn := chol.Symmetric()
		if cn != n {
			t.Errorf("Cholesky size does not match. Got %d, want %d", cn, n)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				got := chol.At(i, j)
				want := test.At(i, j)
				if math.Abs(got-want) > 1e-12 {
					t.Errorf("Cholesky at does not match at %d, %d. Got %v, want %v", i, j, got, want)
				}
			}
		}
	}
}

func TestCholeskySolveTo(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		a   *mat.SymDense
		b   *mat.Dense
		ans *mat.Dense
	}{
		{
			a: mat.NewSymDense(2, []float64{
				1, 0,
				0, 1,
			}),
			b:   mat.NewDense(2, 1, []float64{5, 6}),
			ans: mat.NewDense(2, 1, []float64{5, 6}),
		},
		{
			a: mat.NewSymDense(3, []float64{
				53, 59, 37,
				0, 83, 71,
				37, 71, 101,
			}),
			b:   mat.NewDense(3, 1, []float64{5, 6, 7}),
			ans: mat.NewDense(3, 1, []float64{0.20745069393718094, -0.17421475529583694, 0.11577794010226464}),
		},
	} {
		var chol Cholesky
		csr := matToCSR(test.a, 1e-8)
		chol.Factorize(csr)

		var x mat.Dense
		err := chol.SolveTo(&x, test.b)
		if err != nil {
			t.Errorf("unexpected error from Cholesky solve: %v", err)
		}
		if !mat.EqualApprox(&x, test.ans, 1e-12) {
			t.Error("incorrect Cholesky solve solution")
		}

		var ans mat.Dense
		ans.Mul(test.a, &x)
		if !mat.EqualApprox(&ans, test.b, 1e-12) {
			t.Error("incorrect Cholesky solve solution product")
		}
	}
}

func cholMatches(a *mat.SymDense) bool {
	_, n := a.Dims()
	var chol mat.Cholesky

	ok := chol.Factorize(a)
	if !ok {
		fmt.Println("cannot factorize")
		return false
	}
	var L mat.TriDense
	chol.LTo(&L)
	simpleRes := mat.NewTriDense(n, false, nil)
	cholSimple(a, simpleRes)
	if !mat.EqualApprox(&L, simpleRes, 1e-10) {
		return false
	}
	coo := NewCOO(n, n, nil, nil, nil)
	csrRes := coo.ToCSR()
	aCOO := matToCOO(a, 1e-10)
	aCSR := aCOO.ToCSR()
	cholCSR(aCSR, csrRes)
	if !mat.EqualApprox(&L, csrRes, 1e-10) {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				fmt.Println(L.At(i, j), csrRes.At(i, j))
			}
		}
		return false
	}

	return true
}

// computes a permutation matrix where each non-zero is rand.Float64() instead of 1
// and then return aat of that
func randomScaledPermutationMatrixAAT(n int, src rand.Source) *mat.SymDense {
	idxList := make([]int, n)
	for i := range idxList {
		idxList[i] = i
	}
	rnd := rand.New(src)
	rnd.Shuffle(n, func(i, j int) { idxList[i], idxList[j] = idxList[j], idxList[i] })
	m := mat.NewDense(n, n, nil)

	for i, j := range idxList {
		v := rnd.Float64()
		m.Set(i, j, v)
	}
	mt := m.T()
	mmt := mat.NewDense(n, n, nil)
	mmt.Mul(m, mt)
	mmtSym := mat.NewSymDense(n, mmt.RawMatrix().Data)
	return mmtSym
}

func randomSymDensePosDefinite(n int, fracNZ float64, src rand.Source) *mat.SymDense {
	ok := false
	for !ok {
		m := randomSymDensePosDefiniteInternal(n, fracNZ, src)
		var chol mat.Cholesky
		ok = chol.Factorize(m)
		if ok {
			return m
		}
	}
	return nil
}

func randomSymDensePosDefiniteInternal(n int, fracNZ float64, src rand.Source) *mat.SymDense {
	rnd := rand.New(src)
	m := mat.NewDense(n, n, nil)
	nnz := int(float64(n) * float64(n) * fracNZ)
	rList := make([]int, nnz)
	cList := make([]int, nnz)
	for i := range rList {
		rList[i] = rnd.Intn(n)
		cList[i] = rnd.Intn(n)
	}
	for i := range rList {
		r := rList[i]
		c := cList[i]
		m.Set(r, c, rnd.Float64())
		m.Set(c, r, rnd.Float64())
	}
	mt := m.T()
	mmt := mat.NewDense(n, n, nil)
	mmt.Mul(m, mt)
	mmtSym := mat.NewSymDense(n, mmt.RawMatrix().Data)
	return mmtSym
}

func matToCOO(m mat.Matrix, tol float64) *COO {
	r, c := m.Dims()
	newMat := NewCOO(r, c, nil, nil, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			if v > tol || v < -tol {
				newMat.Set(i, j, v)
			}
		}
	}
	//	nSize := float64(r * c)
	//	nnz := float64(newMat.NNZ())
	//	ratio := nnz / nSize
	//	fmt.Println("frac=", ratio)
	return newMat
}

func matToCSR(m mat.Matrix, tol float64) *CSR {
	coo := matToCOO(m, tol)
	return coo.ToCSR()
}

func BenchmarkCholSimple800(b *testing.B) { cholSimpleBench(b, 400, 0) }

func BenchmarkCholGoNum400S3(b *testing.B) { cholGoNumBench(b, 400, 3.0/400) }
func BenchmarkCholGoNum400S5(b *testing.B) { cholGoNumBench(b, 400, 5.0/400) }
func BenchmarkCholGoNum400S7(b *testing.B) { cholGoNumBench(b, 400, 7.0/400) }
func BenchmarkCholGoNum400S9(b *testing.B) { cholGoNumBench(b, 400, 9.0/400) }
func BenchmarkCholGoNum400(b *testing.B)   { cholGoNumBench(b, 400, 0.0) }
func BenchmarkCholGoNum800(b *testing.B)   { cholGoNumBench(b, 800, 0.0) }
func BenchmarkCholGoNum1600(b *testing.B)  { cholGoNumBench(b, 1600, 0.0) }
func BenchmarkCholGoNum3200(b *testing.B)  { cholGoNumBench(b, 3200, 0.0) }
func BenchmarkCholGoNum6400(b *testing.B)  { cholGoNumBench(b, 6400, 0.0) }

//func BenchmarkCholGoNum12800(b *testing.B) { cholGoNumBench(b, 12800, 0.0) }

func BenchmarkCholSparse400S3(b *testing.B) { sparseCholBench(b, 400, 3.0/400) }
func BenchmarkCholSparse400S5(b *testing.B) { sparseCholBench(b, 400, 5.0/400) }
func BenchmarkCholSparse400S7(b *testing.B) { sparseCholBench(b, 400, 7.0/400) }
func BenchmarkCholSparse400S9(b *testing.B) { sparseCholBench(b, 400, 9.0/400) }
func BenchmarkCholSparse400(b *testing.B)   { sparseCholBench(b, 400, 0.0) }
func BenchmarkCholSparse800(b *testing.B)   { sparseCholBench(b, 800, 0.0) }
func BenchmarkCholSparse1600(b *testing.B)  { sparseCholBench(b, 1600, 0.0) }
func BenchmarkCholSparse3200(b *testing.B)  { sparseCholBench(b, 3200, 0.0) }
func BenchmarkCholSparse6400(b *testing.B)  { sparseCholBench(b, 6400, 0.0) }

//func BenchmarkCholSparse12800(b *testing.B) { sparseCholBench(b, 12800, 0.0) }

func cholGoNumBench(b *testing.B, size int, frac float64) {
	src := rand.NewSource(1)
	m := randomScaledPermutationMatrixAAT(size, src)
	if frac != 0.0 {
		m = randomSymDensePosDefinite(size, frac, src)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var chol mat.Cholesky
		chol.Factorize(m)
	}
}

func cholSimpleBench(b *testing.B, size int, frac float64) {
	src := rand.NewSource(1)
	m := randomScaledPermutationMatrixAAT(size, src)
	if frac != 0.0 {
		m = randomSymDensePosDefinite(size, frac, src)
	}
	simpleRes := mat.NewTriDense(size, false, nil)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cholSimple(m, simpleRes)
	}
}

func sparseCholBench(b *testing.B, size int, frac float64) {
	src := rand.NewSource(1)
	mDense := randomScaledPermutationMatrixAAT(size, src)
	if frac != 0.0 {
		mDense = randomSymDensePosDefinite(size, frac, src)
	}
	coo := NewCOO(size, size, nil, nil, nil)
	csrRes := coo.ToCSR()
	aCSR := matToCSR(mDense, 1e-8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cholCSR(aCSR, csrRes)
	}
}
