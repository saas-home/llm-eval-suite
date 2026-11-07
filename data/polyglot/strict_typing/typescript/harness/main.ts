import { Pipeline } from './pipeline';

function check(cond: boolean, msg: string): void { if (!cond) { throw new Error(msg); } }

const src: number[] = [1, 2, 3, 4, 5];
const out: string[] = new Pipeline(src).filter(n => n % 2 === 1).map(n => `#${n}`).execute();
check(JSON.stringify(out) === '["#1","#3","#5"]', 'filter/map/execute result wrong');
check(JSON.stringify(src) === '[1,2,3,4,5]', 'the source array was mutated');

const base = new Pipeline<number>([1, 2, 3]);
base.filter(n => n > 1);
check(JSON.stringify(base.execute()) === '[1,2,3]', 'filter must return a new pipeline, not change the original');
const first = base.execute();
first.push(99);
check(JSON.stringify(base.execute()) === '[1,2,3]', 'execute() must return a new array');

// Type-level probes: each line below must be a compile error under correct generics.
// @ts-expect-error map<U> must infer U = string, so the result is not number[]
const wrongMap: number[] = new Pipeline([1]).map(n => String(n)).execute();
// @ts-expect-error filter must preserve T = number, so the result is not string[]
const wrongFilter: string[] = new Pipeline([1]).filter(n => n > 0).execute();
void wrongMap;
void wrongFilter;
console.log(`POLYGLOT_OK ${process.env.EVAL_RUN_ID}`);
