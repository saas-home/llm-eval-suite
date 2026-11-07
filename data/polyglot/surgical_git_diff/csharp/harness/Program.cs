using PayrollSystem;

static Employee Emp(decimal salary) => new Employee { Salary = salary };
static void Check(bool cond, string msg) { if (!cond) throw new Exception(msg); }

Check(PayrollCalculator.CalculatePayroll(new List<Employee> { Emp(100m), Emp(200m) }, 0.1m, 5m) == 275m, "off-by-one not fixed (wrong total)");
Check(PayrollCalculator.CalculatePayroll(new List<Employee>(), 0.1m, 5m) == 5m, "empty list must return the bonus pool");
Check(PayrollCalculator.CalculatePayroll(null!, 0.1m, 5m) == 5m, "null list must be treated as an empty list");
Check(PayrollCalculator.CalculatePayroll(new List<Employee> { Emp(100m), null!, Emp(200m) }, 0.1m, 5m) == 275m, "null entries must be skipped");
Console.WriteLine("POLYGLOT_OK " + Environment.GetEnvironmentVariable("EVAL_RUN_ID"));
