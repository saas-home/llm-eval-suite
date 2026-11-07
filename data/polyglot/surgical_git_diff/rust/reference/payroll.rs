pub struct Employee { pub salary: f64 }

pub fn calculate_payroll(employees: &[Option<Employee>], tax_rate: f64, bonus_pool: f64) -> f64 {
    let mut total = 0.0;
    for i in 0..employees.len() {
        let emp = &employees[i];
        if let Some(e) = emp {
            let net = e.salary * (1.0 - tax_rate);
            total += net;
        }
    }
    total + bonus_pool
}
