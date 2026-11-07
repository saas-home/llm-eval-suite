import { calculatePayroll } from './payroll';

type E = { salary: number };
const near = (a: number, b: number): boolean => Math.abs(a - b) < 1e-9;
function check(cond: boolean, msg: string): void {
    if (!cond) { throw new Error(msg); }
}

check(near(calculatePayroll([{ salary: 100 }, { salary: 200 }], 0.1, 5), 275), 'off-by-one not fixed (wrong total)');
check(near(calculatePayroll([], 0.1, 5), 5), 'empty list must return the bonus pool');
check(near(calculatePayroll(null as unknown as E[], 0.1, 5), 5), 'null employees must be treated as an empty list');
check(near(calculatePayroll([{ salary: 100 }, null as unknown as E, { salary: 200 }], 0.1, 5), 275), 'null entries must be skipped');
console.log(`POLYGLOT_OK ${process.env.EVAL_RUN_ID}`);
