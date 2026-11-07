using Moq;
using Payments;
using Xunit;

public class PaymentProcessorTests {
    private readonly Mock<IStripeGateway> _gateway = new();
    private readonly Mock<INotificationClient> _notifier = new();

    private PaymentProcessor Create(string status) {
        _gateway.Setup(g => g.ChargeAsync(It.IsAny<decimal>(), It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(new ChargeResponse(status, "ch_1"));
        _notifier.Setup(n => n.SendReceiptAsync(It.IsAny<string>(), It.IsAny<string>())).ReturnsAsync(true);
        return new PaymentProcessor(_gateway.Object, _notifier.Object);
    }

    [Fact]
    public async Task SuccessSendsReceipt() {
        Assert.True(await Create("succeeded").ProcessOrderAsync("a@x.io", 100m, "tok"));
        _notifier.Verify(n => n.SendReceiptAsync("a@x.io", "ch_1"), Times.Once());
    }

    [Fact]
    public async Task DeclinedSendsNoReceipt() {
        Assert.False(await Create("declined").ProcessOrderAsync("a@x.io", 100m, "tok"));
        _notifier.Verify(n => n.SendReceiptAsync(It.IsAny<string>(), It.IsAny<string>()), Times.Never());
    }

    [Fact]
    public async Task NonPositiveAmountThrows() {
        await Assert.ThrowsAsync<ArgumentException>(() => Create("succeeded").ProcessOrderAsync("a@x.io", 0m, "tok"));
        _gateway.Verify(g => g.ChargeAsync(It.IsAny<decimal>(), It.IsAny<string>(), It.IsAny<string>()), Times.Never());
    }
}
