namespace Payments;

public record ChargeResponse(string Status, string Id);

public interface IStripeGateway { Task<ChargeResponse> ChargeAsync(decimal amount, string currency, string token); }

public interface INotificationClient { Task<bool> SendReceiptAsync(string email, string chargeId); }

public class PaymentProcessor {
    private readonly IStripeGateway _gateway;
    private readonly INotificationClient _notifier;

    public PaymentProcessor(IStripeGateway gateway, INotificationClient notifier) {
        _gateway = gateway;
        _notifier = notifier;
    }

    public async Task<bool> ProcessOrderAsync(string email, decimal amount, string token) {
        if (amount <= 0) throw new ArgumentException("Amount must be positive", nameof(amount));
        var resp = await _gateway.ChargeAsync(amount, "USD", token);
        if (resp.Status == "succeeded") {
            await _notifier.SendReceiptAsync(email, resp.Id);
            return true;
        }
        return false;
    }
}
