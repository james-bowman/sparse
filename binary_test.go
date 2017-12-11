package sparse

import (
	"strconv"
	"testing"
	"fmt"
)

func createBinVec(bitString string) (*BinaryVec, error) {
	bits := []byte(bitString)
	vec := NewBinaryVec(len(bits))
		
	for i, bit := range bits {
		f, err := strconv.ParseFloat(string(bit), 64)
		if err != nil {
			return vec, err
		}
		vec.Set((len(bits)-1) - i, 0, f)
	}

	return vec, nil
}

func TestBinaryVectorManipulation(t *testing.T) {
	tests := []struct{
		bits string
	}{
		{
			bits: "000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "100000000000000000000000000000000000000000000000000000000000000",
		},
		{
			bits: "000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000001",
		},
	}

	for ti, test := range tests {
		bits := []byte(test.bits)
		vec, err := createBinVec(test.bits)
		if err != nil {
			t.Errorf("Test %d: Failed to parse bits because %v", ti, err)
		}

		if len(bits) != vec.Len() {
			t.Errorf("Test %d: Vector is the wrong length, expected %d but received %d", ti, len(bits), vec.Len())
		}

		for i, bit := range bits {
			b := vec.At((len(bits)-1) - i, 0)
			f, err := strconv.ParseFloat(string(bit), 64)
			if err != nil {
				t.Errorf("Test %d: Failed to parse bit (%s) as float because %v", ti, string(bit), err)
			}
			if b != f {
				t.Errorf("Test %d: Failed to get bit expected %s (%f) but received %f", ti, string(bit), b, f)
			}
		}

	}
}

func TestBinaryFormat(t *testing.T) {
	tests := []struct{
		bits string
		format rune
		expected string
	}{
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			format: 's',
			expected: "0000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			format: 'b',
			expected: "0000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "00000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000001",
			format: 'b',
			expected: "00000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "10000000000000000000000000000000000000000000000000000000000000001",
			format: 'b',
			expected: "00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			format: 'x',
			expected: "0000000000000001",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000001101",
			format: 'x',
			expected: "000000000000000d",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000001101",
			format: 'X',
			expected: "000000000000000D",
		},
	}

	for ti, test := range tests {
		vec, err := createBinVec(test.bits)
		if err != nil {
			t.Errorf("Test %d: Failed to parse bits because %v", ti, err)
		}

		out := fmt.Sprintf(fmt.Sprintf("%%%c", test.format), vec)

		if test.expected != out {
			t.Errorf("Test %d: Binary Format failed, expected '%s' but received '%s'\n", ti, test.expected, out)
		}
	}
}

func TestBinaryNNZ(t *testing.T) {
	tests := []struct{
		bits string
		nnz int
	}{
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			nnz: 1,
		},
		{
			bits: "0000100000000000000000000000110000000001000000010000000000000001",
			nnz: 6,
		},
		{
			bits: "10000000000000000000000010000000000000000010000000000110000000001000000000000000000000000000000000000000000000000000000000000011",
			nnz: 8,
		},
	}

	for ti, test := range tests {
		vec, err := createBinVec(test.bits)
		if err != nil {
			t.Errorf("Test %d: Failed to parse bits because %v", ti, err)
		}

		if test.nnz != vec.NNZ() {
			t.Errorf("Test %d: NNZ incorrect, expected %d but received %d", ti, test.nnz, vec.NNZ())
		}
	}
}

func TestBinaryDistance(t *testing.T) {
	tests := []struct{
		bitsa string
		bitsb string
		distance int
	}{
		{
			bitsa: "0000000000000000000000000000000000000000000000000000000000000001",
			bitsb: "0000000000000000000000000000000000000000000000000000000000000000",
			distance: 1,
		},
		{
			bitsa: "0000000000000000000000000000000000000000000000000000000000000000",
			bitsb: "0000000000000000000000000000000000000000000000000000000000000001",
			distance: 1,
		},
		{
			bitsa: "0001000000000000000000000000010110000000000000000000000000000001",
			bitsb: "0000000000000000000000000000001010000000000000000000000000000000",
			distance: 5,
		},
	}

	for ti, test := range tests {
		veca, err := createBinVec(test.bitsa)
		if err != nil {
			t.Errorf("Test %d: Failed to parse bitsa because %v", ti, err)
		}
		vecb, err := createBinVec(test.bitsb)
		if err != nil {
			t.Errorf("Test %d: Failed to parse bitsb because %v", ti, err)
		}

		distance := veca.DistanceFrom(vecb)

		if test.distance != distance {
			t.Errorf("Test %d: Distance incorrect, expected %d but received %d", ti, test.distance, distance)
		}
	}
}