package com.payment;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

import org.junit.jupiter.api.Test;

class PaymentProcessorTest {
    private final PaymentProcessor.StripeGateway gateway = mock(PaymentProcessor.StripeGateway.class);
    private final PaymentProcessor.NotificationClient notifier = mock(PaymentProcessor.NotificationClient.class);
    private final PaymentProcessor processor = new PaymentProcessor(gateway, notifier);

    @Test
    void successSendsReceipt() {
        when(gateway.charge(100, "USD", "tok")).thenReturn(new PaymentProcessor.ChargeResult("succeeded", "ch_1"));
        assertTrue(processor.processOrder("a@x.io", 100, "tok"));
        verify(notifier).sendReceipt("a@x.io", "ch_1");
    }

    @Test
    void declinedSendsNoReceipt() {
        when(gateway.charge(anyInt(), anyString(), anyString())).thenReturn(new PaymentProcessor.ChargeResult("declined", "ch_2"));
        assertFalse(processor.processOrder("a@x.io", 100, "tok"));
        verify(notifier, never()).sendReceipt(anyString(), anyString());
    }

    @Test
    void nonPositiveAmountThrows() {
        assertThrows(IllegalArgumentException.class, () -> processor.processOrder("a@x.io", 0, "tok"));
        verifyNoInteractions(gateway);
    }
}
