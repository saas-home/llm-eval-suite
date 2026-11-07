interface Employee { salary: number; }
export function calculatePayroll(employees: (Employee | null)[] | null | undefined, taxRate: number, bonusPool: number): number {
    let total = 0;
    if (!employees) { return bonusPool; }
    for (let i = 0; i < employees.length; i++) {
        const emp = employees[i];
        if (!emp) { continue; }
        const net = emp.salary * (1 - taxRate);
        total += net;
    }
    return total + bonusPool;
}
