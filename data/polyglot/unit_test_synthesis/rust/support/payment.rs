pub struct ChargeResult { pub status: String, pub id: String }

pub trait StripeGateway {
    fn charge(&self, amount: i64, currency: &str, token: &str) -> ChargeResult;
}

pub trait NotificationClient {
    fn send_receipt(&self, email: &str, charge_id: &str) -> bool;
}

pub struct PaymentProcessor<'a, G: StripeGateway, N: NotificationClient> {
    gateway: &'a G,
    notifier: &'a N,
}

impl<'a, G: StripeGateway, N: NotificationClient> PaymentProcessor<'a, G, N> {
    pub fn new(gateway: &'a G, notifier: &'a N) -> Self {
        Self { gateway, notifier }
    }

    pub fn process_order(&self, email: &str, amount: i64, token: &str) -> Result<bool, String> {
        if amount <= 0 {
            return Err("amount must be positive".to_string());
        }
        let res = self.gateway.charge(amount, "USD", token);
        if res.status == "succeeded" {
            self.notifier.send_receipt(email, &res.id);
            return Ok(true);
        }
        Ok(false)
    }
}
