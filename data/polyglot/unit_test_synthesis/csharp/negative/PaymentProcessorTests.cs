using Moq;
using Payments;
using Xunit;

public class PaymentProcessorTests {
    [Fact]
    public async Task Success() {
        var g = new Mock<IStripeGateway>();
        g.Setup(x => x.ChargeAsync(100m, "USD", "tok")).ReturnsAsync(new ChargeResponse("succeeded", "x"));
        Assert.True(await new PaymentProcessor(g.Object, new Mock<INotificationClient>().Object).ProcessOrderAsync("a", 100m, "tok"));
    }

    [Fact]
    public async Task Negative() {
        var p = new PaymentProcessor(new Mock<IStripeGateway>().Object, new Mock<INotificationClient>().Object);
        await Assert.ThrowsAsync<ArgumentException>(() => p.ProcessOrderAsync("a", -1m, "tok"));
    }

    [Fact]
    public void Constructs() { Assert.NotNull(new PaymentProcessor(new Mock<IStripeGateway>().Object, new Mock<INotificationClient>().Object)); }
}
