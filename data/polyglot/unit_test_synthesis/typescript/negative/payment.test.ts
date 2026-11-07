import { describe, it, expect, vi } from 'vitest';
import { PaymentProcessor } from './paymentProcessor';

describe('PaymentProcessor (weak)', () => {
    it('returns true on success', async () => {
        const p = new PaymentProcessor({ charge: vi.fn().mockResolvedValue({ status: 'succeeded', id: 'x' }) },
                                       { sendReceipt: vi.fn().mockResolvedValue(true) });
        expect(await p.processOrder('a@x.io', 100, 'tok')).toBe(true);
    });
    it('rejects negative amounts', async () => {
        const p = new PaymentProcessor({ charge: vi.fn() }, { sendReceipt: vi.fn() });
        await expect(p.processOrder('a@x.io', -1, 'tok')).rejects.toThrow();
    });
    it('constructs', () => { expect(new PaymentProcessor({ charge: vi.fn() }, { sendReceipt: vi.fn() })).toBeTruthy(); });
});
