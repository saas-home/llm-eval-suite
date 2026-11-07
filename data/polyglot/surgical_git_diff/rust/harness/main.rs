mod payroll;
use payroll::{calculate_payroll, Employee};

fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }

fn main() {
    let staff = vec![Some(Employee { salary: 100.0 }), None, Some(Employee { salary: 200.0 })];
    assert!(near(calculate_payroll(&staff, 0.1, 5.0), 275.0), "off-by-one not fixed (wrong total) or None not skipped");
    assert!(near(calculate_payroll(&[], 0.1, 5.0), 5.0), "empty slice must return the bonus pool");
    println!("POLYGLOT_OK {}", std::env::var("EVAL_RUN_ID").unwrap_or_default());
}
