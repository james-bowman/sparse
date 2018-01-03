package sparse

import (
	"strconv"
	"testing"
	"fmt"
)

func createBinVec(bitString string) (*BinaryVec, error) {
	bits := []rune(bitString)
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
		bits := []rune(test.bits)

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
		{ // 64 bits (1 whole word)
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			format: 's',
			expected: "0000000000000000000000000000000000000000000000000000000000000001",
		},
		{ // 128 bits (2 whole words)
			bits: "00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001",
			format: 's',
			expected: "00000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001",
		},
		{ // 66 bits (2 words)
			bits: "010000000000000000000000000000000000000000000000000000000000000001",
			format: 's',
			expected: "010000000000000000000000000000000000000000000000000000000000000001",
		},
		{ // 37 bits (1 word)
			bits: "0000000000000000000000000000000001001",
			format: 'b',
			expected: "0000000000000000000000000000000001001",
		},
		{ // 64 bits (1 whole word)
			bits: "1000000000000000000000000000000000000000000000000000000000001001",
			format: 'b',
			expected: "1000000000000000000000000000000000000000000000000000000000001001",
		},
		{ // 89 bits (2 words)
			bits: "00000000000000000000000100000000000000000000000000000000000000000000000000000000000000001",
			format: 'b',
			expected: "00000000000000000000000100000000000000000000000000000000000000000000000000000000000000001",
		},
		{ // 65 bits (2 words)
			bits: "10000000000000000000000000000000000000000000000000000000000000001",
			format: 'b',
			expected: "10000000000000000000000000000000000000000000000000000000000000001",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000001",
			format: 'x',
			expected: "1",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000001101",
			format: 'x',
			expected: "d",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000001101",
			format: 'X',
			expected: "D",
		},
		{
			bits: "10000000000000000000000000000000000000000000000000000000000001101",
			format: 'X',
			expected: "1.D",
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

func TestBinarySliceToUint64(t *testing.T) {
	tests := []struct{
		bits string
		from int
		to int
		expectedSlice string
	}{
		{
			bits: "0000000000000000000000000000000000111001101",
			from: 0, 
			to: 8,
			expectedSlice: "11001101",
		},
		{
			bits: "110101010000",
			from: 3, 
			to: 11,
			expectedSlice: "10101010",
		},
		{
			bits: "0101010100000000000000010001001101",
			from: 10, 
			to: 11,
			expectedSlice: "1",
		},
		{
			bits: "0000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000100",
			from: 2,
			to: 66,
			expectedSlice: "1000000000000000000000000000000000000000000000000000000000000001",
		},
	}

	for ti, test := range tests {
		vec, err := createBinVec(test.bits)

		if err != nil {
			t.Errorf("Test %d: failed because %v\n", ti, err)
		}

		val := vec.SliceToUint64(test.from, test.to)

		result := fmt.Sprintf("%b", val)

		if test.expectedSlice != result {
			t.Errorf("Test %d: Expected %s but received %s\n", ti, test.expectedSlice, result)
		}
	}
}