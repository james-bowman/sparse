package sparse

import (
	"encoding"
	"encoding/binary"
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
	_ encoding.BinaryMarshaler   = (*COO)(nil)
	_ encoding.BinaryUnmarshaler = (*COO)(nil)
	_ encoding.BinaryMarshaler   = (*DOK)(nil)
	_ encoding.BinaryUnmarshaler = (*DOK)(nil)
	_ encoding.BinaryMarshaler   = (*CSC)(nil)
	_ encoding.BinaryUnmarshaler = (*CSC)(nil)
	_ encoding.BinaryMarshaler   = (*CSR)(nil)
	_ encoding.BinaryUnmarshaler = (*CSR)(nil)
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
	if len(data) != int(nnz)*sizeFloat64+3*sizeInt64 {
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
	var n int
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

// MarshalBinary binary serialises the receiver into a []byte and returns the result.
//
// SparseMatrix is little-endian encoded as follows:
//   0 -  7  number of rows    (int64)
//   8 - 15  number of columns (int64)
//  16 - 23  number of indptr  (int64)
//  24 - 31  number of ind     (int64)
//  32 - 39  number of non zero elements (int64)
//  40 - ..  data elements for indptr, ind, and data (float64)
func (c *CSR) MarshalBinary() ([]byte, error) {
	bufLen := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
		int64(len(c.matrix.Indptr))*int64(sizeInt64) + // indptr slice
		int64(len(c.matrix.Ind))*int64(sizeInt64) + // ind slice
		int64(len(c.matrix.Data))*int64(sizeFloat64) // data slice
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errors.New("sparse: buffer for data is too big")
	}

	p := 0
	buf := make([]byte, bufLen)
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.matrix.I))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.matrix.J))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Indptr)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Ind)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Data)))
	p += sizeInt64

	for _, x := range c.matrix.Indptr {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.matrix.Ind {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.matrix.Data {
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(x))
		p += sizeFloat64
	}

	return buf, nil
}

// MarshalBinaryTo binary serialises the receiver and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the serialised layout.
func (c *CSR) MarshalBinaryTo(w io.Writer) (int, error) {
	var n int
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(c.matrix.I))
	nn, err := w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(c.matrix.J))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Indptr)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Ind)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Data)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	for _, x := range c.matrix.Indptr {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.matrix.Ind {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.matrix.Data {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(x))
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
//  - an error is returned if the resulting compressed sprase matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *CSR) UnmarshalBinary(data []byte) error {
	if len(data) < 5*sizeInt64 {
		return errors.New("sparse: data is missing required attributes")
	}

	p := 0
	c.matrix.I = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	c.matrix.J = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pn := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pi := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pd := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64

	// if int(nnz) < 0 || nnz > maxLen {
	// 	return errors.New("sparse: data is too big")
	// }
	// if r < 0 || c < 0 || r < nnz || c < nnz {
	// 	return errors.New("sparse: dimensions/data size mismatch")
	// }
	// if len(data) != int(nnz)*sizeFloat64+3*sizeInt64 {
	// 	return errors.New("sparse: data/buffer size mismatch")
	// }

	c.matrix.Indptr = make([]int, pn)
	for i := 0; i < len(c.matrix.Indptr); i++ {
		c.matrix.Indptr[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.matrix.Ind = make([]int, pi)
	for i := 0; i < len(c.matrix.Ind); i++ {
		c.matrix.Ind[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.matrix.Data = make([]float64, pd)
	for i := 0; i < len(c.matrix.Data); i++ {
		c.matrix.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
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
//  - an error is returned if the resulting compressed sparse matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *CSR) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	var n int
	var buf [8]byte

	nn, err := readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	i := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	j := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	indptrn := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	indn := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	datan := int64(binary.LittleEndian.Uint64(buf[:]))

	if int(indptrn) < 0 || indptrn > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(indn) < 0 || indn > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(datan) < 0 || datan > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if i < 0 || j < 0 {
		return n, errors.New("sparse: dimensions/data size mismatch")
	}

	c.matrix.I = int(i)
	c.matrix.J = int(j)
	c.matrix.Indptr = make([]int, indptrn)
	c.matrix.Ind = make([]int, indn)
	c.matrix.Data = make([]float64, datan)

	for i := range c.matrix.Indptr {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Indptr[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.matrix.Ind {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Ind[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.matrix.Data {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
	}

	return n, nil
}

// MarshalBinary binary serialises the receiver into a []byte and returns the result.
//
// SparseMatrix is little-endian encoded as follows:
//   0 -  7  number of rows    (int64)
//   8 - 15  number of columns (int64)
//  16 - 23  number of indptr  (int64)
//  24 - 31  number of ind     (int64)
//  32 - 39  number of non zero elements (int64)
//  40 - ..  data elements for indptr, ind, and data (float64)
func (c *CSC) MarshalBinary() ([]byte, error) {
	bufLen := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
		int64(len(c.matrix.Indptr))*int64(sizeInt64) + // indptr slice
		int64(len(c.matrix.Ind))*int64(sizeInt64) + // ind slice
		int64(len(c.matrix.Data))*int64(sizeFloat64) // data slice
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errors.New("sparse: buffer for data is too big")
	}

	p := 0
	buf := make([]byte, bufLen)
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.matrix.I))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.matrix.J))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Indptr)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Ind)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.matrix.Data)))
	p += sizeInt64

	for _, x := range c.matrix.Indptr {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.matrix.Ind {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.matrix.Data {
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(x))
		p += sizeFloat64
	}

	return buf, nil
}

// MarshalBinaryTo binary serialises the receiver and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the serialised layout.
func (c *CSC) MarshalBinaryTo(w io.Writer) (int, error) {
	var n int
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(c.matrix.I))
	nn, err := w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(c.matrix.J))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Indptr)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Ind)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.matrix.Data)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	for _, x := range c.matrix.Indptr {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.matrix.Ind {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.matrix.Data {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(x))
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
//  - an error is returned if the resulting compressed sprase matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *CSC) UnmarshalBinary(data []byte) error {
	if len(data) < 5*sizeInt64 {
		return errors.New("sparse: data is missing required attributes")
	}

	p := 0
	c.matrix.I = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	c.matrix.J = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pn := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pi := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pd := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64

	// if int(nnz) < 0 || nnz > maxLen {
	// 	return errors.New("sparse: data is too big")
	// }
	// if r < 0 || c < 0 || r < nnz || c < nnz {
	// 	return errors.New("sparse: dimensions/data size mismatch")
	// }
	// if len(data) != int(nnz)*sizeFloat64+3*sizeInt64 {
	// 	return errors.New("sparse: data/buffer size mismatch")
	// }

	c.matrix.Indptr = make([]int, pn)
	for i := 0; i < len(c.matrix.Indptr); i++ {
		c.matrix.Indptr[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.matrix.Ind = make([]int, pi)
	for i := 0; i < len(c.matrix.Ind); i++ {
		c.matrix.Ind[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.matrix.Data = make([]float64, pd)
	for i := 0; i < len(c.matrix.Data); i++ {
		c.matrix.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
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
//  - an error is returned if the resulting compressed sparse matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *CSC) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	var n int
	var buf [8]byte

	nn, err := readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	i := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	j := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	indptrn := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	indn := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	datan := int64(binary.LittleEndian.Uint64(buf[:]))

	if int(indptrn) < 0 || indptrn > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(indn) < 0 || indn > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(datan) < 0 || datan > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if i < 0 || j < 0 {
		return n, errors.New("sparse: dimensions/data size mismatch")
	}

	c.matrix.I = int(i)
	c.matrix.J = int(j)
	c.matrix.Indptr = make([]int, indptrn)
	c.matrix.Ind = make([]int, indn)
	c.matrix.Data = make([]float64, datan)

	for i := range c.matrix.Indptr {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Indptr[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.matrix.Ind {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Ind[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.matrix.Data {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.matrix.Data[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
	}

	return n, nil
}

// MarshalBinary binary serialises the receiver into a []byte and returns the result.
//
// compressedSparse is little-endian encoded as follows:
//   0 -  7  number of rows    (int64)
//   8 - 15  number of columns (int64)
//  16 - 23  number of indptr  (int64)
//  24 - 31  number of ind     (int64)
//  32 - 39  number of non zero elements (int64)
//  40 - ..  data elements for indptr, ind, and data (float64)
func (c *COO) MarshalBinary() ([]byte, error) {
	bufLen := 5*int64(sizeInt64) + // row and column count plus lengths of the slices
		//2 + // colMajor and canonicalised booleans
		int64(len(c.rows))*int64(sizeInt64) + // rows slice
		int64(len(c.cols))*int64(sizeInt64) + // cols slice
		int64(len(c.data))*int64(sizeFloat64) // data slice
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errors.New("sparse: buffer for data is too big")
	}
	p := 0
	buf := make([]byte, bufLen)
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.r))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.c))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.rows)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.cols)))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.data)))
	p += sizeInt64

	for _, x := range c.rows {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.cols {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(x))
		p += sizeInt64
	}

	for _, x := range c.data {
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(x))
		p += sizeFloat64
	}

	return buf, nil
}

// MarshalBinaryTo binary serialises the receiver and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the serialised layout.
func (c *COO) MarshalBinaryTo(w io.Writer) (int, error) {
	var n int
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(c.r))
	nn, err := w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(c.c))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.rows)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.cols)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.data)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	for _, x := range c.rows {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.cols {
		binary.LittleEndian.PutUint64(buf[:], uint64(x))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
	}

	for _, x := range c.data {
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(x))
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
//  - an error is returned if the resulting compressed sprase matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *COO) UnmarshalBinary(data []byte) error {
	if len(data) < 5*sizeInt64+2 {
		return errors.New("sparse: data is missing required attributes")
	}

	p := 0
	c.r = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	c.c = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pr := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pc := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	pd := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64

	// if int(nnz) < 0 || nnz > maxLen {
	// 	return errors.New("sparse: data is too big")
	// }
	// if r < 0 || c < 0 || r < nnz || c < nnz {
	// 	return errors.New("sparse: dimensions/data size mismatch")
	// }
	// if len(data) != int(nnz)*sizeFloat64+3*sizeInt64 {
	// 	return errors.New("sparse: data/buffer size mismatch")
	// }

	c.rows = make([]int, pr)
	for i := 0; i < len(c.rows); i++ {
		c.rows[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.cols = make([]int, pc)
	for i := 0; i < len(c.cols); i++ {
		c.cols[i] = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
	}

	c.data = make([]float64, pd)
	for i := 0; i < len(c.data); i++ {
		c.data[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
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
//  - an error is returned if the resulting compressed sparse matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *COO) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	var n int
	var buf [8]byte

	nn, err := readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	i := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	j := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	rcnt := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	ccnt := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	datan := int64(binary.LittleEndian.Uint64(buf[:]))

	if int(rcnt) < 0 || rcnt > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(ccnt) < 0 || ccnt > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if int(datan) < 0 || datan > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if i < 0 || j < 0 {
		return n, errors.New("sparse: dimensions/data size mismatch")
	}

	c.r = int(i)
	c.c = int(j)
	c.rows = make([]int, rcnt)
	c.cols = make([]int, ccnt)
	c.data = make([]float64, datan)

	for i := range c.rows {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.rows[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.cols {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.cols[i] = int(binary.LittleEndian.Uint64(buf[:]))
	}

	for i := range c.data {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		c.data[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
	}

	return n, nil
}

// MarshalBinary binary serialises the receiver into a []byte and returns the result.
//
// DOK is little-endian encoded as follows:
//   0 -  7  number of rows    (int64)
//   8 - 15  number of columns (int64)
//  16 - ..  data elements     (key + float64)
func (c *DOK) MarshalBinary() ([]byte, error) {
	bufLen := 3*int64(sizeInt64) + // row and column count plus number of elements
		int64(len(c.elements))*int64(sizeInt64+sizeInt64+sizeFloat64) // key + value entry in elements
	if bufLen <= 0 {
		// bufLen is too big and has wrapped around.
		return nil, errors.New("sparse: buffer for data is too big")
	}
	p := 0
	buf := make([]byte, bufLen)
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.r))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(c.c))
	p += sizeInt64
	binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(len(c.elements)))
	p += sizeInt64

	for k, v := range c.elements {
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(k.i))
		p += sizeInt64
		binary.LittleEndian.PutUint64(buf[p:p+sizeInt64], uint64(k.j))
		p += sizeInt64
		binary.LittleEndian.PutUint64(buf[p:p+sizeFloat64], math.Float64bits(v))
		p += sizeFloat64
	}
	return buf, nil
}

// MarshalBinaryTo binary serialises the receiver and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
//
// See MarshalBinary for the serialised layout.
func (c *DOK) MarshalBinaryTo(w io.Writer) (int, error) {
	var n int
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(c.r))
	nn, err := w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	binary.LittleEndian.PutUint64(buf[:], uint64(c.c))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	binary.LittleEndian.PutUint64(buf[:], uint64(len(c.elements)))
	nn, err = w.Write(buf[:])
	n += nn
	if err != nil {
		return n, err
	}

	for k, v := range c.elements {
		binary.LittleEndian.PutUint64(buf[:], uint64(k.i))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		binary.LittleEndian.PutUint64(buf[:], uint64(k.j))
		nn, err = w.Write(buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
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
//  - an error is returned if the resulting compressed sprase matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *DOK) UnmarshalBinary(data []byte) error {
	if len(data) < 3*sizeInt64 {
		return errors.New("sparse: data is missing required attributes")
	}

	p := 0
	c.r = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	c.c = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64
	cnt := int64(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
	p += sizeInt64

	// if int(nnz) < 0 || nnz > maxLen {
	// 	return errors.New("sparse: data is too big")
	// }
	// if r < 0 || c < 0 || r < nnz || c < nnz {
	// 	return errors.New("sparse: dimensions/data size mismatch")
	// }
	// if len(data) != int(nnz)*sizeFloat64+3*sizeInt64 {
	// 	return errors.New("sparse: data/buffer size mismatch")
	// }

	var k key
	var v float64
	c.elements = make(map[key]float64, cnt)
	for i := 0; i < int(cnt); i++ {
		k.i = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
		k.j = int(binary.LittleEndian.Uint64(data[p : p+sizeInt64]))
		p += sizeInt64
		v = math.Float64frombits(binary.LittleEndian.Uint64(data[p : p+sizeFloat64]))
		p += sizeFloat64
		c.elements[k] = v
	}
	return nil
}

// UnmarshalBinaryFrom binary deserialises the []byte into the receiver and returns
// the number of bytes read and an error if any.
//
// See MarshalBinary for the on-disk layout.
//
// Limited checks on the validity of the binary input are performed:
//  - an error is returned if the resulting compressed sparse matrix is too
//  big for the current architecture (e.g. a 16GB matrix written by a
//  64b application and read back from a 32b application.)
// UnmarshalBinary does not limit the size of the unmarshaled matrix, and so
// it should not be used on untrusted data.
func (c *DOK) UnmarshalBinaryFrom(r io.Reader) (int, error) {
	var n int
	var buf [8]byte

	nn, err := readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	i := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	j := int64(binary.LittleEndian.Uint64(buf[:]))

	nn, err = readUntilFull(r, buf[:])
	n += nn
	if err != nil {
		return n, err
	}
	cnt := int64(binary.LittleEndian.Uint64(buf[:]))

	if int(cnt) < 0 || cnt > maxLen {
		return n, errors.New("sparse: data is too big")
	}
	if i < 0 || j < 0 {
		return n, errors.New("sparse: dimensions/data size mismatch")
	}

	c.r = int(i)
	c.c = int(j)
	c.elements = make(map[key]float64, cnt)

	var k key
	var v float64
	for i := 0; i < int(cnt); i++ {
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		k.i = int(binary.LittleEndian.Uint64(buf[:]))
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		k.j = int(binary.LittleEndian.Uint64(buf[:]))
		nn, err = readUntilFull(r, buf[:])
		n += nn
		if err != nil {
			return n, err
		}
		v = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
		c.elements[k] = v
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
