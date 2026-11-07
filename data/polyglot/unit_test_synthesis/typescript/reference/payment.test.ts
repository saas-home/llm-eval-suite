import { describe, it, expect, vi } from 'vitest';
import { PaymentProcessor } from './paymentProcessor';

function setup(status: string) {
    const gateway = { charge: vi.fn().mockResolvedValue({ status, id: 'ch_1' }) };
    const notifier = { sendReceipt: vi.fn().mockResolvedValue(true) };
    return { gateway, notifier, processor: new PaymentProcessor(gateway, notifier) };
}

describe('PaymentProcessor.processOrder', () => {
    it('sends a receipt and returns true on success', async () => {
        const { gateway, notifier, processor } = setup('succeeded');
        await expect(processor.processOrder('a@x.io', 100, 'tok')).resolves.toBe(true);
        expect(gateway.charge).toHaveBeenCalledWith(100, 'USD', 'tok');
        expect(notifier.sendReceipt).toHaveBeenCalledWith('a@x.io', 'ch_1');
    });
    it('does not send a receipt and returns false when declined', async () => {
        const { notifier, processor } = setup('declined');
        await expect(processor.processOrder('a@x.io', 100, 'tok')).resolves.toBe(false);
        expect(notifier.sendReceipt).not.toHaveBeenCalled();
    });
    it('rejects a non-positive amount', async () => {
        const { gateway, processor } = setup('succeeded');
        await expect(processor.processOrder('a@x.io', -5, 'tok')).rejects.toThrow();
        expect(gateway.charge).not.toHaveBeenCalled();
    });
});
