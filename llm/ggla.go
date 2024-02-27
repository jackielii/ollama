package llm

import (
	"encoding/binary"
	"errors"
	"io"
	"slices"
)

type containerGGLA struct {
	version uint32
}

func (c *containerGGLA) Name() string {
	return "ggla"
}

func (c *containerGGLA) Decode(rso *readSeekOffset) (model, error) {
	binary.Read(rso, binary.LittleEndian, &c.version)

	switch c.version {
	case 1:
	default:
		return nil, errors.New("invalid version")
	}

	model := newModelGGLA(c)
	err := model.decode(rso)
	return model, err
}

type modelGGLA struct {
	*containerGGLA

	kv
	tensors []tensor
}

func newModelGGLA(container *containerGGLA) *modelGGLA {
	return &modelGGLA{
		containerGGLA: container,
		kv:            make(kv),
	}
}

func (m *modelGGLA) decode(rso *readSeekOffset) error {
	var r uint32
	if err := binary.Read(rso, binary.LittleEndian, &r); err != nil {
		return err
	}
	m.kv["r"] = r

	var alpha uint32
	if err := binary.Read(rso, binary.LittleEndian, &alpha); err != nil {
		return err
	}
	m.kv["alpha"] = alpha

	for {
		var dims uint32
		if err := binary.Read(rso, binary.LittleEndian, &dims); err != nil {
			return err
		}

		var namesize uint32
		if err := binary.Read(rso, binary.LittleEndian, &namesize); err != nil {
			return err
		}

		var t tensor
		if err := binary.Read(rso, binary.LittleEndian, &t.Kind); err != nil {
			return err
		}

		t.Shape = make([]uint64, dims)
		for i := 0; uint32(i) < dims; i++ {
			var shape32 uint32
			if err := binary.Read(rso, binary.LittleEndian, &shape32); err != nil {
				return err
			}

			t.Shape[i] = uint64(shape32)
		}

		// ggla tensor shape is reversed
		// ref: https://github.com/ggerganov/llama.cpp/blob/29ae62d2ae163e2b68aa0ad3bf2ab4636de0c957/convert-lora-to-ggml.py#L44
		slices.Reverse(t.Shape)

		name := make([]byte, namesize)
		if err := binary.Read(rso, binary.LittleEndian, &name); err != nil {
			return err
		}

		t.Name = string(name)

		if _, err := rso.Seek((rso.offset+31)&-32, io.SeekStart); err != nil {
			return err
		}

		t.offset = uint64(rso.offset)

		if _, err := rso.Seek(int64(t.size()), io.SeekCurrent); err != nil {
			return err
		}

		m.tensors = append(m.tensors, t)
	}
}

func (m *modelGGLA) KV() kv {
	return m.kv
}

func (m *modelGGLA) Tensor() []tensor {
	return m.tensors
}
