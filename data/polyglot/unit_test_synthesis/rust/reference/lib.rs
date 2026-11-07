#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    struct MockGateway { status: &'static str, calls: Cell<u32> }
    impl StripeGateway for MockGateway {
        fn charge(&self, _amount: i64, _currency: &str, _token: &str) -> ChargeResult {
            self.calls.set(self.calls.get() + 1);
            ChargeResult { status: self.status.to_string(), id: "ch_1".to_string() }
        }
    }

    struct MockNotifier { receipts: RefCell<Vec<(String, String)>> }
    impl NotificationClient for MockNotifier {
        fn send_receipt(&self, email: &str, charge_id: &str) -> bool {
            self.receipts.borrow_mut().push((email.to_string(), charge_id.to_string()));
            true
        }
    }

    fn mocks(status: &'static str) -> (MockGateway, MockNotifier) {
        (MockGateway { status, calls: Cell::new(0) }, MockNotifier { receipts: RefCell::new(Vec::new()) })
    }

    #[test]
    fn test_process_order_success() {
        let (g, n) = mocks("succeeded");
        assert_eq!(PaymentProcessor::new(&g, &n).process_order("a@x.io", 100, "tok"), Ok(true));
        assert_eq!(*n.receipts.borrow(), vec![("a@x.io".to_string(), "ch_1".to_string())]);
    }

    #[test]
    fn test_process_order_declined() {
        let (g, n) = mocks("declined");
        assert_eq!(PaymentProcessor::new(&g, &n).process_order("a@x.io", 100, "tok"), Ok(false));
        assert!(n.receipts.borrow().is_empty());
    }

    #[test]
    fn test_process_order_negative_amount() {
        let (g, n) = mocks("succeeded");
        assert!(PaymentProcessor::new(&g, &n).process_order("a@x.io", -5, "tok").is_err());
        assert_eq!(g.calls.get(), 0);
    }
}
