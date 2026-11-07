pub struct Record { pub id: Option<String>, pub qty: i32, pub price: f64, pub discount: Option<f64> }

pub struct ProcessedOrder { pub id: String, pub total: f64 }

fn valid_discount(d: Option<f64>) -> Option<f64> {
    d.filter(|d| *d > 0.0 && *d <= 1.0)
}

pub fn process_order_data(records: &[Record]) -> Vec<ProcessedOrder> {
    records
        .iter()
        .filter(|r| r.qty > 0 && r.price >= 0.0)
        .map(|r| {
            let base = r.qty as f64 * r.price;
            let total = match valid_discount(r.discount) {
                Some(d) => base * (1.0 - d),
                None => base,
            };
            let id = r.id.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "N/A".to_string());
            ProcessedOrder { id, total: (total * 100.0).round() / 100.0 }
        })
        .collect()
}
