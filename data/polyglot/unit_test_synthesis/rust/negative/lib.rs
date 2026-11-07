#[cfg(test)]
mod tests {
    use super::*;
    struct G;
    impl StripeGateway for G {
        fn charge(&self, _a: i64, _c: &str, _t: &str) -> ChargeResult { ChargeResult { status: "succeeded".into(), id: "x".into() } }
    }
    struct N;
    impl NotificationClient for N { fn send_receipt(&self, _e: &str, _c: &str) -> bool { true } }

    #[test]
    fn success() { assert_eq!(PaymentProcessor::new(&G, &N).process_order("a", 100, "t"), Ok(true)); }
    #[test]
    fn negative() { assert!(PaymentProcessor::new(&G, &N).process_order("a", -1, "t").is_err()); }
    #[test]
    fn nothing() {}
}
