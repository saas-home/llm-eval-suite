pub struct Record { pub id: Option<String>, pub qty: i32, pub price: f64, pub discount: Option<f64> }

pub struct ProcessedOrder { pub id: String, pub total: f64 }

pub fn process_order_data(records: &[Record]) -> Vec<ProcessedOrder> {
    let mut res = Vec::new();
    for r in records {
        if r.qty > 0 && r.price >= 0.0 {
            let mut total = r.qty as f64 * r.price;
            if let Some(d) = r.discount {
                if d > 0.0 && d <= 1.0 {
                    total *= 1.0 - d;
                }
            }
            let id = match &r.id {
                Some(s) if !s.is_empty() => s.clone(),
                _ => "N/A".to_string(),
            };
            res.push(ProcessedOrder { id, total: (total * 100.0).round() / 100.0 });
        }
    }
    res
}
