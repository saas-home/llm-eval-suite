package orders

import (
	"fmt"
	"strings"
	"testing"
)

func render(out []ProcessedOrder) string {
	parts := make([]string, 0, len(out))
	for _, o := range out {
		parts = append(parts, fmt.Sprintf("%s:%.9f", o.ID, o.Total))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func TestRefactorPreservesBehavior(t *testing.T) {
	cases := [][]Record{
		nil,
		{},
		{{ID: "a", Qty: 2, Price: 3}},
		{{ID: "", Qty: 1, Price: 1}},
		{{ID: "q0", Qty: 0, Price: 5}, {ID: "qn", Qty: -1, Price: 5}},
		{{ID: "pn", Qty: 2, Price: -1}, {ID: "p0", Qty: 2, Price: 0}},
		{{ID: "d0", Qty: 4, Price: 2.5, Discount: 0}, {ID: "d25", Qty: 4, Price: 2.5, Discount: 0.25}},
		{{ID: "d1", Qty: 3, Price: 7, Discount: 1}, {ID: "d15", Qty: 3, Price: 7, Discount: 1.5}, {ID: "dn", Qty: 3, Price: 7, Discount: -0.2}},
		{{ID: "r", Qty: 7, Price: 1.115, Discount: 0.1}},
	}
	for _, c := range cases {
		before, after := render(legacyProcessOrderData(c)), render(ProcessOrderData(c))
		if before != after {
			t.Fatalf("behavior changed for %+v: legacy %s vs refactored %s", c, before, after)
		}
	}
}
