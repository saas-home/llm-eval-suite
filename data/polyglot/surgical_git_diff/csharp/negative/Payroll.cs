namespace PayrollSystem;
using System.Collections.Generic;

public class Employee { public decimal Salary { get; set; } }

public static class PayrollCalculator {
    public static decimal CalculatePayroll(List<Employee> employees, decimal taxRate, decimal bonusPool) {
        decimal total = 0m;
        for (int i = 0; i <= employees.Count; i++) {
            var emp = employees[i];
            decimal net = emp.Salary * (1m - taxRate);
            total += net;
        }
        return total + bonusPool;
    }
}
