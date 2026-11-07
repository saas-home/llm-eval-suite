package payment

import "testing"

type mockGateway struct {
	status string
	calls  int
}

func (g *mockGateway) Charge(amount int, currency, token string) (map[string]string, error) {
	g.calls++
	return map[string]string{"status": g.status, "id": "ch_1"}, nil
}

type mockNotifier struct{ receipts [][2]string }

func (n *mockNotifier) SendReceipt(email, chargeID string) bool {
	n.receipts = append(n.receipts, [2]string{email, chargeID})
	return true
}

func TestSuccessSendsReceipt(t *testing.T) {
	g, n := &mockGateway{status: "succeeded"}, &mockNotifier{}
	ok, err := (&PaymentProcessor{Gateway: g, Notifier: n}).ProcessOrder("a@x.io", 100, "tok")
	if err != nil || !ok {
		t.Fatalf("expected success, got %v %v", ok, err)
	}
	if len(n.receipts) != 1 || n.receipts[0] != [2]string{"a@x.io", "ch_1"} {
		t.Fatalf("receipt not sent correctly: %v", n.receipts)
	}
}

func TestDeclinedSendsNoReceipt(t *testing.T) {
	g, n := &mockGateway{status: "declined"}, &mockNotifier{}
	ok, err := (&PaymentProcessor{Gateway: g, Notifier: n}).ProcessOrder("a@x.io", 100, "tok")
	if err != nil || ok {
		t.Fatalf("expected false, got %v %v", ok, err)
	}
	if len(n.receipts) != 0 {
		t.Fatalf("receipt must not be sent: %v", n.receipts)
	}
}

func TestNegativeAmountErrors(t *testing.T) {
	g, n := &mockGateway{status: "succeeded"}, &mockNotifier{}
	if _, err := (&PaymentProcessor{Gateway: g, Notifier: n}).ProcessOrder("a@x.io", -5, "tok"); err == nil {
		t.Fatal("expected an error for a negative amount")
	}
	if g.calls != 0 {
		t.Fatal("gateway must not be charged")
	}
}
