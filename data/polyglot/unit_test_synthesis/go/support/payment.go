package payment

import "errors"

type StripeGateway interface {
	Charge(amount int, currency, token string) (map[string]string, error)
}

type NotificationClient interface {
	SendReceipt(email, chargeID string) bool
}

type PaymentProcessor struct {
	Gateway  StripeGateway
	Notifier NotificationClient
}

func (p *PaymentProcessor) ProcessOrder(email string, amount int, token string) (bool, error) {
	if amount <= 0 {
		return false, errors.New("amount must be positive")
	}
	resp, err := p.Gateway.Charge(amount, "USD", token)
	if err != nil {
		return false, err
	}
	if resp["status"] == "succeeded" {
		p.Notifier.SendReceipt(email, resp["id"])
		return true, nil
	}
	return false, nil
}
