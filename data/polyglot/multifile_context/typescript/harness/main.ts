import { AuthenticationService } from './authService';
import type { User } from './models';
import type { IUserRepository } from './repository';

class FakeRepo implements IUserRepository {
    calls: string[] = [];
    constructor(private readonly users: Map<string, User>) {}
    async getById(userId: string): Promise<User | null> { this.calls.push(userId); return this.users.get(userId) ?? null; }
    async save(user: User): Promise<void> { this.users.set(user.id, user); }
    async listActive(): Promise<User[]> { return [...this.users.values()].filter(u => u.isActive); }
}

function check(cond: boolean, msg: string): void { if (!cond) { throw new Error(msg); } }

async function main(): Promise<void> {
    const repo = new FakeRepo(new Map<string, User>([
        ['u1', { id: 'u1', email: 'a@x.io', isActive: true }],
        ['u2', { id: 'u2', email: 'b@x.io', isActive: false }],
    ]));
    const svc = new AuthenticationService(repo, { tenantId: 't1', maxTokenTtlSeconds: 900 });
    const tok = await svc.authenticate('u1');
    check(tok !== null && tok.token === 'tok_u1' && tok.expiresIn === 900, 'active user token wrong');
    check((await svc.authenticate('u2')) === null, 'inactive user must get null');
    check((await svc.authenticate('nope')) === null, 'unknown user must get null');
    check(JSON.stringify(repo.calls) === JSON.stringify(['u1', 'u2', 'nope']), 'must look users up via the injected repository');
    console.log(`POLYGLOT_OK ${process.env.EVAL_RUN_ID}`);
}

main().catch((e: unknown) => { console.error(e); process.exit(1); });
