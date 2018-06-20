package sparse

import (
	"bytes"
	"testing"

	"gonum.org/v1/gonum/mat"
)

var diagonals = []struct {
	want *DIA
	raw  []byte
}{
	{
		want: NewDIA(2, 2, []float64{1, 5}),
		raw:  []byte("\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xF0\x3F\x00\x00\x00\x00\x00\x00\x14\x40"),
	},
	{
		want: NewDIA(2, 3, []float64{1, 5}),
		raw:  []byte("\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xF0\x3F\x00\x00\x00\x00\x00\x00\x14\x40"),
	},
	{
		want: NewDIA(3, 2, []float64{1, 5}),
		raw:  []byte("\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xF0\x3F\x00\x00\x00\x00\x00\x00\x14\x40"),
	},
}

func TestDIAMarshallBinary(t *testing.T) {
	for ti, test := range diagonals {
		t.Logf("**** TestDIAMarshallBinary - Test Run %d.\n", ti+1)

		buf, err := test.want.MarshalBinary()
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := 3*sizeInt64 + test.want.NNZ()*sizeFloat64
		if len(buf) != size {
			t.Errorf("encoded size test: want=%d got=%d\n", size, len(buf))
		}

		if !bytes.Equal(buf, test.raw) {
			t.Errorf("error encoding test: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf),
				string(test.raw),
			)
		}
	}
}

func TestDIAMarshallTo(t *testing.T) {
	for ti, test := range diagonals {
		t.Logf("**** TestDIAMarshallTo - Test Run %d.\n", ti+1)
		buf := new(bytes.Buffer)
		n, err := test.want.MarshalBinaryTo(buf)
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		nnz := test.want.NNZ()
		size := nnz*sizeFloat64 + 3*sizeInt64
		if n != size {
			t.Errorf("encoded size: want=%d got=%d\n", size, n)
		}

		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("error encoding: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf.Bytes()),
				string(test.raw),
			)
		}
	}
}

func TestDIAUnmarshalBinary(t *testing.T) {
	for ti, test := range diagonals {
		t.Logf("**** TestDenseUnmarshal - Test Run %d.\n", ti+1)
		var v DIA
		err := v.UnmarshalBinary(test.raw)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestDIAUnmarshalFrom(t *testing.T) {
	for ti, test := range diagonals {
		t.Logf("**** TestDenseUnmarshalFrom - Test Run %d.\n", ti+1)
		var v DIA
		buf := bytes.NewReader(test.raw)
		n, err := v.UnmarshalBinaryFrom(buf)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if n != len(test.raw) {
			t.Errorf("error decoding: lengths differ.\n got=%d\nwant=%d\n",
				n, len(test.raw),
			)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

var (
	compressedCSR = []struct {
		want *CSR
		raw  []byte
	}{
		{
			want: NewCSR(2, 2, []int{0, 1, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCSR(2, 3, []int{0, 1, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCSR(3, 2, []int{0, 1, 2, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x04\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
	}
	compressedCSC = []struct {
		want *CSC
		raw  []byte
	}{
		{
			want: NewCSC(2, 2, []int{0, 1, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCSC(2, 3, []int{0, 1, 2, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x04\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCSC(3, 2, []int{0, 1, 2}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(indptr)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(ind)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 3 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
	}
)

func TestCSRMarshalBinary(t *testing.T) {
	for ti, test := range compressedCSR {
		t.Logf("**** TestCSRMarshallBinary - Test Run %d.\n", ti+1)

		buf, err := test.want.MarshalBinary()
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
			int64(len(test.want.matrix.Indptr))*int64(sizeInt64) + // indptr slice
			int64(len(test.want.matrix.Ind))*int64(sizeInt64) + // ind slice
			int64(len(test.want.matrix.Data))*int64(sizeFloat64) // data slice
		if len(buf) != int(size) {
			t.Errorf("encoded size test: want=%d got=%d\n", size, len(buf))
		}

		if !bytes.Equal(buf, test.raw) {
			t.Errorf("error encoding test: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf),
				string(test.raw),
			)
		}
	}
}

func TestCSRMarshallTo(t *testing.T) {
	for ti, test := range compressedCSR {
		t.Logf("**** TestCSRMarshallTo - Test Run %d.\n", ti+1)
		buf := new(bytes.Buffer)
		n, err := test.want.MarshalBinaryTo(buf)
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := len(test.want.matrix.Data)*sizeFloat64 + len(test.want.matrix.Indptr)*sizeInt64 +
			len(test.want.matrix.Ind)*sizeInt64 + 5*sizeInt64
		if n != size {
			t.Errorf("encoded size: want=%d got=%d\n", size, n)
		}

		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("error encoding: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf.Bytes()),
				string(test.raw),
			)
		}
	}
}

func TestCSRUnmarshalBinary(t *testing.T) {
	for ti, test := range compressedCSR {
		t.Logf("**** TestCSRUnmarshal - Test Run %d.\n", ti+1)
		var v CSR
		err := v.UnmarshalBinary(test.raw)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestCSRUnmarshalFrom(t *testing.T) {
	for ti, test := range compressedCSR {
		t.Logf("**** TestCSRUnmarshalFrom - Test Run %d.\n", ti+1)
		var v CSR
		buf := bytes.NewReader(test.raw)
		n, err := v.UnmarshalBinaryFrom(buf)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if n != len(test.raw) {
			t.Errorf("error decoding: lengths differ.\n got=%d\nwant=%d\n",
				n, len(test.raw),
			)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestCSCMarshalBinary(t *testing.T) {
	for ti, test := range compressedCSC {
		t.Logf("**** TestCSCMarshallBinary - Test Run %d.\n", ti+1)

		buf, err := test.want.MarshalBinary()
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
			int64(len(test.want.matrix.Indptr))*int64(sizeInt64) + // indptr slice
			int64(len(test.want.matrix.Ind))*int64(sizeInt64) + // ind slice
			int64(len(test.want.matrix.Data))*int64(sizeFloat64) // data slice
		if len(buf) != int(size) {
			t.Errorf("encoded size test: want=%d got=%d\n", size, len(buf))
		}

		if !bytes.Equal(buf, test.raw) {
			t.Errorf("error encoding test: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf),
				string(test.raw),
			)
		}
	}
}

func TestCSCMarshallTo(t *testing.T) {
	for ti, test := range compressedCSC {
		t.Logf("**** TestCSCMarshallTo - Test Run %d.\n", ti+1)
		buf := new(bytes.Buffer)
		n, err := test.want.MarshalBinaryTo(buf)
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := len(test.want.matrix.Data)*sizeFloat64 + len(test.want.matrix.Indptr)*sizeInt64 +
			len(test.want.matrix.Ind)*sizeInt64 + 5*sizeInt64
		if n != size {
			t.Errorf("encoded size: want=%d got=%d\n", size, n)
		}

		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("error encoding: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf.Bytes()),
				string(test.raw),
			)
		}
	}
}

func TestCSCUnmarshalBinary(t *testing.T) {
	for ti, test := range compressedCSC {
		t.Logf("**** TestCSCUnmarshal - Test Run %d.\n", ti+1)
		var v CSC
		err := v.UnmarshalBinary(test.raw)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestCSCUnmarshalFrom(t *testing.T) {
	for ti, test := range compressedCSC {
		t.Logf("**** TestCSCUnmarshalFrom - Test Run %d.\n", ti+1)
		var v CSC
		buf := bytes.NewReader(test.raw)
		n, err := v.UnmarshalBinaryFrom(buf)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if n != len(test.raw) {
			t.Errorf("error decoding: lengths differ.\n got=%d\nwant=%d\n",
				n, len(test.raw),
			)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

var (
	coordinates = []struct {
		want *COO
		raw  []byte
	}{
		{
			want: NewCOO(2, 2, []int{0, 1}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(rows)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(cols)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCOO(2, 3, []int{0, 1}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(rows)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(cols)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewCOO(3, 2, []int{0, 1}, []int{1, 0}, []float64{0.5, -0.5}),
			raw: []byte(
				"\x03\x00\x00\x00\x00\x00\x00\x00" + // 3
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(rows)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(cols)
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2 = len(data)
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\x3F" + // 0.5
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
	}
)

func TestCOOMarshalBinary(t *testing.T) {
	for ti, test := range coordinates {
		t.Logf("**** TestCOOMarshallBinary - Test Run %d.\n", ti+1)

		buf, err := test.want.MarshalBinary()
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
			int64(len(test.want.rows))*int64(sizeInt64) + // indptr slice
			int64(len(test.want.cols))*int64(sizeInt64) + // ind slice
			int64(len(test.want.data))*int64(sizeFloat64) // data slice
		if len(buf) != int(size) {
			t.Errorf("encoded size test: want=%d got=%d\n", size, len(buf))
		}

		if !bytes.Equal(buf, test.raw) {
			t.Errorf("error encoding test: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf),
				string(test.raw),
			)
		}
	}
}

func TestCOOMarshallTo(t *testing.T) {
	for ti, test := range coordinates {
		t.Logf("**** TestCOOMarshallTo - Test Run %d.\n", ti+1)
		buf := new(bytes.Buffer)
		n, err := test.want.MarshalBinaryTo(buf)
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		size := len(test.want.data)*sizeFloat64 + len(test.want.rows)*sizeInt64 +
			len(test.want.cols)*sizeInt64 + 5*sizeInt64
		if n != size {
			t.Errorf("encoded size: want=%d got=%d\n", size, n)
		}

		if !bytes.Equal(buf.Bytes(), test.raw) {
			t.Errorf("error encoding: bytes mismatch.\n got=%q\nwant=%q\n",
				string(buf.Bytes()),
				string(test.raw),
			)
		}
	}
}

func TestCOOUnmarshalBinary(t *testing.T) {
	for ti, test := range coordinates {
		t.Logf("**** TestCOOUnmarshal - Test Run %d.\n", ti+1)
		var v COO
		err := v.UnmarshalBinary(test.raw)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestCOOUnmarshalFrom(t *testing.T) {
	for ti, test := range coordinates {
		t.Logf("**** TestCOOUnmarshalFrom - Test Run %d.\n", ti+1)
		var v COO
		buf := bytes.NewReader(test.raw)
		n, err := v.UnmarshalBinaryFrom(buf)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if n != len(test.raw) {
			t.Errorf("error decoding: lengths differ.\n got=%d\nwant=%d\n",
				n, len(test.raw),
			)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

var (
	elements = []struct {
		want  *DOK
		items map[key]float64
		raw   []byte
		raw0  []byte
		raw1  []byte
	}{
		{
			want: NewDOK(2, 2),
			items: map[key]float64{
				key{i: 0, j: 1}: 0.5,
				key{i: 1, j: 0}: -0.5,
			},
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00", // 2 = len(elements)
			),
			raw0: []byte(
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\xE0\x3F", // 0.5
			),
			raw1: []byte(
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewDOK(2, 3),
			items: map[key]float64{
				key{i: 0, j: 1}: 0.5,
				key{i: 1, j: 0}: -0.5,
			},
			raw: []byte(
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x03\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00", // 2 = len(elements)
			),
			raw0: []byte(
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\xE0\x3F", // 0.5
			),
			raw1: []byte(
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
		{
			want: NewDOK(3, 2),
			items: map[key]float64{
				key{i: 0, j: 1}: 0.5,
				key{i: 1, j: 0}: -0.5,
			},
			raw: []byte(
				"\x03\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00" + // 2
					"\x02\x00\x00\x00\x00\x00\x00\x00", // 2 = len(elements)
			),
			raw0: []byte(
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\xE0\x3F", // 0.5
			),
			raw1: []byte(
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // 1
					"\x00\x00\x00\x00\x00\x00\x00\x00" + // 0
					"\x00\x00\x00\x00\x00\x00\xE0\xBF", // -0.5
			),
		},
	}
)

func TestDOKMarshalBinary(t *testing.T) {
	for ti, test := range elements {
		t.Logf("**** TestDOKMarshallBinary - Test Run %d.\n", ti+1)

		for k, v := range test.items {
			test.want.Set(k.i, k.j, v)
		}
		buf, err := test.want.MarshalBinary()
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}

		// Because iterating over Go maps does not have a predictable order we cannot compare the whole byte slice at once
		size := 3 * (sizeInt64) // row and column count plus lengths of the slices
		if !bytes.Equal(buf[:size], test.raw) {
			t.Errorf("error encoding test: bytes mismatch in header.\n got=%q\nwant=%q\n",
				string(buf[:size]),
				string(test.raw),
			)
		}

		// There are only 2 elements in the test case so it's one ordering or the the other
		var a, b []byte
		offset := size
		size = len(test.want.elements) * (sizeInt64 + sizeInt64 + sizeFloat64)
		a = append(test.raw0, test.raw1...)
		b = append(test.raw1, test.raw0...)
		if !(bytes.Equal(a, buf[offset:]) || bytes.Equal(b, buf[offset:])) {
			t.Errorf("error encoding test: bytes mismatch in elements.\n got=%q\nwant(a)=%q\nwant(b)=%q\n",
				string(buf[offset:]),
				string(a),
				string(b),
			)
		}
	}
}

func TestDOKMarshallTo(t *testing.T) {
	for ti, test := range elements {
		t.Logf("**** TestDOKMarshallTo - Test Run %d.\n", ti+1)

		for k, v := range test.items {
			test.want.Set(k.i, k.j, v)
		}
		bb := new(bytes.Buffer)
		n, err := test.want.MarshalBinaryTo(bb)
		if err != nil {
			t.Errorf("error encoding: %v\n", err)
			continue
		}
		buf := bb.Bytes()

		size := 3*(sizeInt64) + len(test.want.elements)*(sizeInt64+sizeInt64+sizeFloat64)
		if n != size {
			t.Errorf("encoded size: want=%d got=%d\n", size, n)
		}

		// Because iterating over Go maps does not have a predictable order we cannot compare the whole byte slice at once
		size = 3 * (sizeInt64) // row and column count plus lengths of the slices
		if !bytes.Equal(buf[:size], test.raw) {
			t.Errorf("error encoding test: bytes mismatch in header.\n got=%q\nwant=%q\n",
				string(buf[:size]),
				string(test.raw),
			)
		}

		// There are only 2 elements in the test case so it's one ordering or the the other
		var a, b []byte
		offset := size
		size = len(test.want.elements) * (sizeInt64 + sizeInt64 + sizeFloat64)
		a = append(test.raw0, test.raw1...)
		b = append(test.raw1, test.raw0...)
		if !(bytes.Equal(a, buf[offset:]) || bytes.Equal(b, buf[offset:])) {
			t.Errorf("error encoding test: bytes mismatch in elements.\n got=%q\nwant(a)=%q\nwant(b)=%q\n",
				string(buf[offset:]),
				string(a),
				string(b),
			)
		}
	}
}

func TestDOKUnmarshalBinary(t *testing.T) {
	for ti, test := range elements {
		t.Logf("**** TestDOKUnmarshal - Test Run %d.\n", ti+1)
		var v DOK
		raw := test.raw
		raw = append(raw, test.raw0...)
		raw = append(raw, test.raw1...)
		err := v.UnmarshalBinary(raw)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}

		for k, v := range test.items {
			test.want.Set(k.i, k.j, v)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}

func TestDOKUnmarshalFrom(t *testing.T) {
	for ti, test := range elements {
		t.Logf("**** TestDOKUnmarshalFrom - Test Run %d.\n", ti+1)
		var v DOK
		raw := test.raw
		raw = append(raw, test.raw0...)
		raw = append(raw, test.raw1...)
		buf := bytes.NewReader(raw)
		n, err := v.UnmarshalBinaryFrom(buf)
		if err != nil {
			t.Errorf("error decoding: %v\n", err)
			continue
		}
		if n != len(raw) {
			t.Errorf("error decoding: lengths differ.\n got=%d\nwant=%d\n",
				n, len(raw),
			)
		}

		for k, v := range test.items {
			test.want.Set(k.i, k.j, v)
		}
		if !mat.Equal(&v, test.want) {
			t.Errorf("error decoding: values differ.\n got=%v\nwant=%v\n",
				&v,
				test.want,
			)
		}
	}
}
