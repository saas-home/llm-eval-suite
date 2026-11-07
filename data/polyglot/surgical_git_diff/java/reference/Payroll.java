package com.payroll;
import java.util.List;

public class Payroll {
    public static class Employee { public double salary; }
    public static double calculatePayroll(List<Employee> employees, double taxRate, double bonusPool) {
        double total = 0.0;
        if (employees == null) {
            return bonusPool;
        }
        for (int i = 0; i < employees.size(); i++) {
            Employee emp = employees.get(i);
            if (emp == null) {
                continue;
            }
            double net = emp.salary * (1.0 - taxRate);
            total += net;
        }
        return total + bonusPool;
    }
}
