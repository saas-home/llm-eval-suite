package payment

import "testing"

type g struct{}

func (g) Charge(amount int, currency, token string) (map[string]string, error) {
	return map[string]string{"status": "succeeded", "id": "x"}, nil
}

type n struct{}

func (n) SendReceipt(email, chargeID string) bool { return true }

func TestSuccess(t *testing.T) {
	if ok, _ := (&PaymentProcessor{Gateway: g{}, Notifier: n{}}).ProcessOrder("a", 100, "t"); !ok {
		t.Fatal("expected true")
	}
}

func TestNegative(t *testing.T) {
	if _, err := (&PaymentProcessor{Gateway: g{}, Notifier: n{}}).ProcessOrder("a", -1, "t"); err == nil {
		t.Fatal("expected error")
	}
}

func TestNothing(t *testing.T) {}
