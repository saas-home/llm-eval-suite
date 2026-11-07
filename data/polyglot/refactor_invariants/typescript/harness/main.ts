import { processOrderData as legacy } from './legacy';
import { processOrderData as refactored } from './refactored';

const cases: unknown[] = [
    null, undefined, [], [null], [undefined], [{}],
    [{ id: 'a', qty: 2, price: 3 }],
    [{ qty: 2, price: 3 }],
    [{ id: '', qty: 1, price: 1 }],
    [{ id: 'q0', qty: 0, price: 5 }, { id: 'qn', qty: -1, price: 5 }],
    [{ id: 'pn', qty: 2, price: -1 }, { id: 'p0', qty: 2, price: 0 }],
    [{ id: 'mq', price: 4 }, { id: 'mp', qty: 4 }],
    [{ id: 'd0', qty: 4, price: 2.5, discount: 0 }, { id: 'd25', qty: 4, price: 2.5, discount: 0.25 }],
    [{ id: 'd1', qty: 3, price: 7, discount: 1 }, { id: 'd15', qty: 3, price: 7, discount: 1.5 }, { id: 'dn', qty: 3, price: 7, discount: -0.2 }],
    [{ id: 'r1', qty: 1, price: 1.005 }, { id: 'r2', qty: 3, price: 0.335 }, { id: 'r3', qty: 7, price: 1.115, discount: 0.1 }],
];

for (const c of cases) {
    const before = JSON.stringify(legacy(c as never));
    const after = JSON.stringify(refactored(c as never));
    if (before !== after) {
        throw new Error(`behavior changed for ${JSON.stringify(c)}: legacy ${before} vs refactored ${after}`);
    }
}
console.log(`POLYGLOT_OK ${process.env.EVAL_RUN_ID}`);
