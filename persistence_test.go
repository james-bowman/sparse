package sparse

import (
	"bytes"
	"gonum.org/v1/gonum/mat"
	"testing"
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
