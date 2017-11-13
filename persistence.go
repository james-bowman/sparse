package sparse

import (
	"encoding/binary"
	"encoding"
	"errors"
	"io"
	"math"
)

const (
	// maxLen is the biggest slice/array len one can create on a 32/64b platform.
	maxLen = int64(int(^uint(0) >> 1))
)

var (
	sizeInt64   = binary.Size(int64(0))
	sizeFloat64 = binary.Size(float64(0))

	_ encoding.BinaryMarshaler   = (*DIA)(nil)
	_ encoding.BinaryUnmarshaler = (*DIA)(nil)
)

// MarshalBinary binary serialises the receiver into a []byte and returns the result.
//
// DIA is little-endian encoded as follows:
//   0 -  7  number of rows    (int64)
//   8 - 15  number of columns (int64)
// 	16 - 23  number of non zero elements (along the diagonal) (int64)
//  24 - ..  diagonal matrix data elements (float64)
func (m DIA) MarshalBinary() ([]byte, error) {
	bufLen := 3*int64(sizeInt64) + int64(len(m.data))*int64(sizeFloat64)
	if bufLen <= 0 {
		return nil, errors.New("sparse: buffer for data is too big")
	}

	p := 0
	buf := make([]byte, bufLen)
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(m.m))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(m.n))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(m.NNZ()))
	p += sizeInt64

	for i := 0; i < m.NNZ(); i++ {
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(m.data[i]))
		p += sizeFloat64
	}

	return buf, nil
}

// MarshalBinaryTo binary serialises the receiver and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the serialised layout.
func (m DIA) MarshalBinaryTo(w io.Writer) (int, error) {
	var n int
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(m.m))
	nn, err := w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(m.n))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(m.NNZ()))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	for i := 0; i < m.NNZ(); i++ {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(m.data[i]))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	return n, nil
}

// UnmarshalBinary binary deserialises the []byte into the receiver.
// It panics if the receiver is a non-zero DIA matrix.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - an error is returned if the resulting DIA matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (m *DIA) UnmarshalBinary(data []byte) error {
	if len(data) < 3*sizeInt64 {
		return errors.New("sparse: data is missing required attributes")
	}

	p := 0
	r := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	c := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	nnz := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64

	if int(nnz) < 0 || nnz > maxLen {
		return errors.New("sparse: data is too big")
	}
	if r < 0 || c < 0 || r < nnz || c < nnz {
		return errors.New("sparse: dimensions/data size mismatch")
	}
	if len(data) != int(nnz)*sizeFloat64 + 3*sizeInt64 {
		return errors.New("sparse: data/buffer size mismatch")
	}

	m.m = int(r)
	m.n = int(c)
	m.data = make([]float64, nnz)

	for i := range m.data {
		m.data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
		p += sizeFloat64
	}

	return nil
}

// UnmarshalBinaryFrom binary deserialises the []byte into the receiver and returns
// the number of bytes read and an error if any.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - an error is returned if the resulting DIA matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (m *DIA) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	var n   int
	var buf [8]byte

	nn, err := readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	row := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	col := int64(binary.LittleEndian.Uint64(buf[:]))
	
	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	nnz := int64(binary.LittleEndian.Uint64(buf[:]))

	if int(nnz) < 0 || nnz > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if row < 0 || col < 0 || row < nnz || col < nnz {
		return n, errors.New("sparse: dimensions/data size mismatch")
	}

	m.m = int(row)
	m.n = int(col)
	m.data = make([]float64, nnz)
	
	for i := range m.data {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		m.data[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
	}

	return n, nil
}

// readUntilFull reads from r into buf until it has read len(buf).
// It returns the number of bytes copied and an error if fewer bytes were read.
// If an EOF happens after reading fewer than len(buf) bytes, io.ErrUnexpectedEOF is returned.
func readUntilFull(r io.Reader, buf []byte) (int, error) {
	var n int
	var err error
	for n < len(buf) && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if n == len(buf) {
		return n, nil
	}
	if err == io.EOF {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
}