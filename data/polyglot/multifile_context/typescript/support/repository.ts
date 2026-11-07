import { User } from './models';
export interface IUserRepository {
    getById(userId: string): Promise<User | null>;
    save(user: User): Promise<void>;
    listActive(): Promise<User[]>;
}
