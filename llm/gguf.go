package llm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

type containerGGUF struct {
	bo binary.ByteOrder

	Version uint32

	V1 struct {
		NumTensor uint32
		NumKV     uint32
	}

	V2 struct {
		NumTensor uint64
		NumKV     uint64
	}
}

func (c *containerGGUF) Name() string {
	return "gguf"
}

func (c *containerGGUF) Decode(rso *readSeekOffset) (model, error) {
	binary.Read(rso, c.bo, &c.Version)

	switch c.Version {
	case 1:
		binary.Read(rso, c.bo, &c.V1)
	default:
		binary.Read(rso, c.bo, &c.V2)
	}

	model := newModelGGUF(c)
	err := model.decode(rso)
	return model, err
}

const (
	ggufTypeUint8 uint32 = iota
	ggufTypeInt8
	ggufTypeUint16
	ggufTypeInt16
	ggufTypeUint32
	ggufTypeInt32
	ggufTypeFloat32
	ggufTypeBool
	ggufTypeString
	ggufTypeArray
	ggufTypeUint64
	ggufTypeInt64
	ggufTypeFloat64
)

type modelGGUF struct {
	*containerGGUF

	kv      KV
	tensors []tensor

	parameters uint64
}

func newModelGGUF(container *containerGGUF) *modelGGUF {
	return &modelGGUF{
		containerGGUF: container,
		kv:            make(KV),
	}
}

func (llm *modelGGUF) KV() KV {
	return llm.kv
}

func (llm *modelGGUF) Tensor() []tensor {
	return llm.tensors
}

func (llm *modelGGUF) NumTensor() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumTensor)
	}

	return llm.V2.NumTensor
}

func (llm *modelGGUF) NumKV() uint64 {
	if llm.Version == 1 {
		return uint64(llm.V1.NumKV)
	}

	return llm.V2.NumKV
}

func (llm *modelGGUF) decode(rso *readSeekOffset) error {
	// decode key-values
	for i := 0; uint64(i) < llm.NumKV(); i++ {
		k, err := llm.readString(rso)
		if err != nil {
			return err
		}

		vtype := llm.readU32(rso)

		var v any
		switch vtype {
		case ggufTypeUint8:
			v = llm.readU8(rso)
		case ggufTypeInt8:
			v = llm.readI8(rso)
		case ggufTypeUint16:
			v = llm.readU16(rso)
		case ggufTypeInt16:
			v = llm.readI16(rso)
		case ggufTypeUint32:
			v = llm.readU32(rso)
		case ggufTypeInt32:
			v = llm.readI32(rso)
		case ggufTypeUint64:
			v = llm.readU64(rso)
		case ggufTypeInt64:
			v = llm.readI64(rso)
		case ggufTypeFloat32:
			v = llm.readF32(rso)
		case ggufTypeFloat64:
			v = llm.readF64(rso)
		case ggufTypeBool:
			v = llm.readBool(rso)
		case ggufTypeString:
			s, err := llm.readString(rso)
			if err != nil {
				return err
			}

			v = s
		case ggufTypeArray:
			a, err := llm.readArray(rso)
			if err != nil {
				return err
			}

			v = a
		default:
			return fmt.Errorf("invalid type: %d", vtype)
		}

		if vtype != ggufTypeArray && k != "tokenizer.chat_template" {
			llm.kv[k] = v
		}
	}

	// decode tensors
	for i := 0; uint64(i) < llm.NumTensor(); i++ {
		name, err := llm.readString(rso)
		if err != nil {
			return err
		}

		// dims is the number of dimensions in the tensor
		dims := llm.readU32(rso)

		shape := make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			shape[i] = llm.readU64(rso)
		}

		tensor := tensor{
			Name:   name,
			Kind:   llm.readU32(rso),
			offset: llm.readU64(rso),
			Shape:  shape,
		}

		llm.tensors = append(llm.tensors, tensor)
		llm.parameters += tensor.parameters()
	}

	// patch KV with parameter count
	llm.kv["general.parameter_count"] = llm.parameters

	alignment, ok := llm.kv["general.alignment"].(uint32)
	if !ok {
		alignment = 32
	}

	rso.Seek(int64(alignment)-rso.offset%int64(alignment), io.SeekCurrent)
	for _, tensor := range llm.tensors {
		padded := (int64(tensor.size()) + int64(alignment) - 1) & ^(int64(alignment) - 1)
		rso.Seek(padded, io.SeekCurrent)
	}

	return nil
}

func (llm modelGGUF) readU8(r io.Reader) uint8 {
	var u8 uint8
	binary.Read(r, llm.bo, &u8)
	return u8
}

func (llm modelGGUF) readI8(r io.Reader) int8 {
	var i8 int8
	binary.Read(r, llm.bo, &i8)
	return i8
}

func (llm modelGGUF) readU16(r io.Reader) uint16 {
	var u16 uint16
	binary.Read(r, llm.bo, &u16)
	return u16
}

func (llm modelGGUF) readI16(r io.Reader) int16 {
	var i16 int16
	binary.Read(r, llm.bo, &i16)
	return i16
}

func (llm modelGGUF) readU32(r io.Reader) uint32 {
	var u32 uint32
	binary.Read(r, llm.bo, &u32)
	return u32
}

func (llm modelGGUF) readI32(r io.Reader) int32 {
	var i32 int32
	binary.Read(r, llm.bo, &i32)
	return i32
}

func (llm modelGGUF) readU64(r io.Reader) uint64 {
	var u64 uint64
	binary.Read(r, llm.bo, &u64)
	return u64
}

func (llm modelGGUF) readI64(r io.Reader) int64 {
	var i64 int64
	binary.Read(r, llm.bo, &i64)
	return i64
}

func (llm modelGGUF) readF32(r io.Reader) float32 {
	var f32 float32
	binary.Read(r, llm.bo, &f32)
	return f32
}

func (llm modelGGUF) readF64(r io.Reader) float64 {
	var f64 float64
	binary.Read(r, llm.bo, &f64)
	return f64
}

func (llm modelGGUF) readBool(r io.Reader) bool {
	var b bool
	binary.Read(r, llm.bo, &b)
	return b
}

func (llm modelGGUF) readStringV1(r io.Reader) (string, error) {
	var nameLength uint32
	binary.Read(r, llm.bo, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	// gguf v1 strings are null-terminated
	b.Truncate(b.Len() - 1)

	return b.String(), nil
}

func (llm modelGGUF) readString(r io.Reader) (string, error) {
	if llm.Version == 1 {
		return llm.readStringV1(r)
	}

	var nameLength uint64
	binary.Read(r, llm.bo, &nameLength)

	var b bytes.Buffer
	if _, err := io.CopyN(&b, r, int64(nameLength)); err != nil {
		return "", err
	}

	return b.String(), nil
}

func (llm *modelGGUF) readArrayV1(r io.Reader) (arr []any, err error) {
	atype := llm.readU32(r)
	n := llm.readU32(r)

	for i := 0; uint32(i) < n; i++ {
		switch atype {
		case ggufTypeUint8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeInt8:
			arr = append(arr, llm.readI8(r))
		case ggufTypeUint16:
			arr = append(arr, llm.readU16(r))
		case ggufTypeInt16:
			arr = append(arr, llm.readI16(r))
		case ggufTypeUint32:
			arr = append(arr, llm.readU32(r))
		case ggufTypeInt32:
			arr = append(arr, llm.readI32(r))
		case ggufTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case ggufTypeBool:
			arr = append(arr, llm.readBool(r))
		case ggufTypeString:
			s, err := llm.readStringV1(r)
			if err != nil {
				return nil, err
			}

			arr = append(arr, s)
		default:
			return nil, fmt.Errorf("invalid array type: %d", atype)
		}
	}

	return
}

func (llm *modelGGUF) readArray(r io.Reader) (arr []any, err error) {
	if llm.Version == 1 {
		return llm.readArrayV1(r)
	}

	atype := llm.readU32(r)
	n := llm.readU64(r)

	for i := 0; uint64(i) < n; i++ {
		switch atype {
		case ggufTypeUint8:
			arr = append(arr, llm.readU8(r))
		case ggufTypeInt8:
			arr = append(arr, llm.readI8(r))
		case ggufTypeUint16:
			arr = append(arr, llm.readU16(r))
		case ggufTypeInt16:
			arr = append(arr, llm.readI16(r))
		case ggufTypeUint32:
			arr = append(arr, llm.readU32(r))
		case ggufTypeInt32:
			arr = append(arr, llm.readI32(r))
		case ggufTypeUint64:
			arr = append(arr, llm.readU64(r))
		case ggufTypeInt64:
			arr = append(arr, llm.readI64(r))
		case ggufTypeFloat32:
			arr = append(arr, llm.readF32(r))
		case ggufTypeFloat64:
			arr = append(arr, llm.readF64(r))
		case ggufTypeBool:
			arr = append(arr, llm.readBool(r))
		case ggufTypeString:
			s, err := llm.readString(r)
			if err != nil {
				return nil, err
			}

			arr = append(arr, s)
		default:
			return nil, fmt.Errorf("invalid array type: %d", atype)
		}
	}

	return
}
