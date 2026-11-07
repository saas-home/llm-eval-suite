import { AuthToken, TenantConfig } from './models';
import { IUserRepository } from './repository';

export class AuthenticationService {
    constructor(private readonly userRepo: IUserRepository, private readonly config: TenantConfig) {}

    async authenticate(userId: string): Promise<AuthToken | null> {
        const user = await this.userRepo.getById(userId);
        if (!user) {
            return null;
        }
        return { token: `tok_${user.id}`, expiresIn: this.config.maxTokenTtlSeconds };
    }
}
