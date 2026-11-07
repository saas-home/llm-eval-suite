export interface StripeGateway { charge(amount: number, currency: string, token: string): Promise<{ status: string; id: string }>; }
export interface NotificationClient { sendReceipt(email: string, chargeId: string): Promise<boolean>; }
export class PaymentProcessor {
    constructor(private gateway: StripeGateway, private notifier: NotificationClient) {}
    async processOrder(email: string, amount: number, token: string): Promise<boolean> {
        if (amount <= 0) throw new Error('Amount must be positive');
        const resp = await this.gateway.charge(amount, 'USD', token);
        if (resp.status === 'succeeded') {
            await this.notifier.sendReceipt(email, resp.id);
            return true;
        }
        return false;
    }
}
