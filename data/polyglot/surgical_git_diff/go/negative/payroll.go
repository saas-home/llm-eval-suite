package payroll

type Employee struct{ Salary float64 }

func CalculatePayroll(employees []*Employee, taxRate float64, bonusPool float64) float64 {
	total := 0.0
	for i := 0; i <= len(employees); i++ {
		emp := employees[i]
		net := emp.Salary * (1.0 - taxRate)
		total += net
	}
	return total + bonusPool
}
