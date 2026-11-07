package pipeline

import (
	"fmt"
	"reflect"
	"testing"
)

func TestPipeline(t *testing.T) {
	src := []int{1, 2, 3, 4, 5}
	p := NewPipeline(src)
	odd := p.Filter(func(n int) bool { return n%2 == 1 })
	out := MapPipeline(odd, func(n int) string { return fmt.Sprintf("#%d", n) }).Execute()
	if !reflect.DeepEqual(out, []string{"#1", "#3", "#5"}) {
		t.Fatalf("filter/map/execute result wrong: %v", out)
	}
	if !reflect.DeepEqual(p.Execute(), []int{1, 2, 3, 4, 5}) {
		t.Fatalf("Filter must return a new pipeline, not change the original: %v", p.Execute())
	}
	got := p.Execute()
	got[0] = 99
	if p.Execute()[0] != 1 {
		t.Fatal("Execute must return a copy")
	}
	src[1] = 42
	if p.Execute()[1] != 2 {
		t.Fatal("NewPipeline must copy its input (the pipeline is immutable)")
	}
}
