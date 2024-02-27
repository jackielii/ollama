package llm

import (
	"encoding/binary"
	"errors"
	"io"
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

	// remaining file contents aren't decoded
	model := newGGLAModel(c)
	if err := model.decode(rso); err != nil {
		return nil, err
	}

	rso.Seek(0, io.SeekEnd)
	return model, nil
}

type gglaModel struct {
	*containerGGLA
}

func newGGLAModel(c *containerGGLA) *gglaModel {
	return &gglaModel{containerGGLA: c}
}

func (m *gglaModel) decode(rso *readSeekOffset) error {
	return nil
}

func (m *gglaModel) KV() kv {
	return kv{}
}

func (m *gglaModel) Tensor() []tensor {
	return nil
}
