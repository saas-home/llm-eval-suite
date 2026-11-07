package com.payment;

public class PaymentProcessor {
    public interface StripeGateway { ChargeResult charge(int amount, String currency, String token); }
    public interface NotificationClient { boolean sendReceipt(String email, String chargeId); }
    public record ChargeResult(String status, String id) {}

    private final StripeGateway gateway;
    private final NotificationClient notifier;

    public PaymentProcessor(StripeGateway gateway, NotificationClient notifier) {
        this.gateway = gateway;
        this.notifier = notifier;
    }

    public boolean processOrder(String email, int amount, String token) {
        if (amount <= 0) throw new IllegalArgumentException("Amount must be positive");
        var res = gateway.charge(amount, "USD", token);
        if ("succeeded".equals(res.status())) {
            notifier.sendReceipt(email, res.id());
            return true;
        }
        return false;
    }
}
