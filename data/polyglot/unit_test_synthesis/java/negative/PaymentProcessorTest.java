package com.payment;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import org.junit.jupiter.api.Test;

class PaymentProcessorTest {
    @Test
    void success() {
        var gateway = mock(PaymentProcessor.StripeGateway.class);
        when(gateway.charge(100, "USD", "tok")).thenReturn(new PaymentProcessor.ChargeResult("succeeded", "x"));
        assertTrue(new PaymentProcessor(gateway, mock(PaymentProcessor.NotificationClient.class)).processOrder("a", 100, "tok"));
    }

    @Test
    void negative() {
        var p = new PaymentProcessor(mock(PaymentProcessor.StripeGateway.class), mock(PaymentProcessor.NotificationClient.class));
        assertThrows(IllegalArgumentException.class, () -> p.processOrder("a", -1, "tok"));
    }

    @Test
    void constructs() { assertNotNull(new PaymentProcessor(null, null)); }
}
