package pipeline

type Pipeline[T any] struct {
	data []T
}

func NewPipeline[T any](data []T) *Pipeline[T] {
	return &Pipeline[T]{data: append([]T(nil), data...)}
}

func (p *Pipeline[T]) Filter(predicate func(T) bool) *Pipeline[T] {
	var out []T
	for _, v := range p.data {
		if predicate(v) {
			out = append(out, v)
		}
	}
	return &Pipeline[T]{data: out}
}

func MapPipeline[T any, U any](p *Pipeline[T], fn func(T) U) *Pipeline[U] {
	var boxed []interface{}
	for _, v := range p.data {
		boxed = append(boxed, fn(v))
	}
	out := make([]U, 0, len(boxed))
	for _, b := range boxed {
		out = append(out, b.(U))
	}
	return &Pipeline[U]{data: out}
}

func (p *Pipeline[T]) Execute() []T {
	return append([]T(nil), p.data...)
}
