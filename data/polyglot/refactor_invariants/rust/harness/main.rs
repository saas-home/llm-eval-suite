#![allow(dead_code)]
mod legacy;
mod refactored;

macro_rules! rec {
    ($m:ident, $id:expr, $qty:expr, $price:expr, $disc:expr) => {
        $m::Record { id: $id.map(|s: &str| s.to_string()), qty: $qty, price: $price, discount: $disc }
    };
}

macro_rules! cases {
    ($m:ident) => {
        vec![
            vec![],
            vec![rec!($m, Some("a"), 2, 3.0, None)],
            vec![rec!($m, None, 2, 3.0, None), rec!($m, Some(""), 1, 1.0, None)],
            vec![rec!($m, Some("q0"), 0, 5.0, None), rec!($m, Some("qn"), -1, 5.0, None)],
            vec![rec!($m, Some("pn"), 2, -1.0, None), rec!($m, Some("p0"), 2, 0.0, None)],
            vec![rec!($m, Some("d0"), 4, 2.5, Some(0.0)), rec!($m, Some("d25"), 4, 2.5, Some(0.25))],
            vec![rec!($m, Some("d1"), 3, 7.0, Some(1.0)), rec!($m, Some("d15"), 3, 7.0, Some(1.5)), rec!($m, Some("dn"), 3, 7.0, Some(-0.2))],
            vec![rec!($m, Some("r1"), 1, 1.005, None), rec!($m, Some("r2"), 3, 0.335, None), rec!($m, Some("r3"), 7, 1.115, Some(0.1))],
        ]
    };
}

fn main() {
    let before: Vec<Vec<String>> = cases!(legacy).iter().map(|c| legacy::process_order_data(c).iter().map(|o| format!("{}:{:.9}", o.id, o.total)).collect()).collect();
    let after: Vec<Vec<String>> = cases!(refactored).iter().map(|c| refactored::process_order_data(c).iter().map(|o| format!("{}:{:.9}", o.id, o.total)).collect()).collect();
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(b, a, "behavior changed for case {}: legacy {:?} vs refactored {:?}", i, b, a);
    }
    println!("POLYGLOT_OK {}", std::env::var("EVAL_RUN_ID").unwrap_or_default());
}
