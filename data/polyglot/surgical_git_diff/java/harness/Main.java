import com.payroll.Payroll;
import java.util.ArrayList;
import java.util.List;

public class Main {
    static Payroll.Employee emp(double salary) {
        Payroll.Employee e = new Payroll.Employee();
        e.salary = salary;
        return e;
    }

    static void check(boolean cond, String msg) {
        if (!cond) throw new AssertionError(msg);
    }

    static boolean near(double a, double b) { return Math.abs(a - b) < 1e-9; }

    public static void main(String[] args) {
        check(near(Payroll.calculatePayroll(List.of(emp(100), emp(200)), 0.1, 5), 275), "off-by-one not fixed (wrong total)");
        check(near(Payroll.calculatePayroll(new ArrayList<>(), 0.1, 5), 5), "empty list must return the bonus pool");
        check(near(Payroll.calculatePayroll(null, 0.1, 5), 5), "null list must be treated as an empty list");
        List<Payroll.Employee> withNull = new ArrayList<>();
        withNull.add(emp(100));
        withNull.add(null);
        withNull.add(emp(200));
        check(near(Payroll.calculatePayroll(withNull, 0.1, 5), 275), "null entries must be skipped");
        System.out.println("POLYGLOT_OK " + System.getenv("EVAL_RUN_ID"));
    }
}
