package payroll

import (
	"math"
	"testing"
)

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestPatchedBehavior(t *testing.T) {
	if got := CalculatePayroll([]*Employee{{Salary: 100}, {Salary: 200}}, 0.1, 5); !near(got, 275) {
		t.Fatalf("off-by-one not fixed (wrong total): got %v", got)
	}
	if got := CalculatePayroll(nil, 0.1, 5); !near(got, 5) {
		t.Fatalf("nil slice must return the bonus pool: got %v", got)
	}
	if got := CalculatePayroll([]*Employee{{Salary: 100}, nil, {Salary: 200}}, 0.1, 5); !near(got, 275) {
		t.Fatalf("nil entries must be skipped: got %v", got)
	}
}
