export interface User { id: string; email: string; isActive: boolean; }
export interface TenantConfig { tenantId: string; maxTokenTtlSeconds: number; }
export interface AuthToken { token: string; expiresIn: number; }
