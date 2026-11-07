interface Employee { salary: number; }
export function calculatePayroll(employees: Employee[], taxRate: number, bonusPool: number): number {
    let total = 0;
    for (let i = 0; i <= employees.length; i++) {
        const emp = employees[i];
        const net = emp.salary * (1 - taxRate);
        total += net;
    }
    return total + bonusPool;
}
