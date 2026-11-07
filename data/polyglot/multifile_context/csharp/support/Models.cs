public record User(string Id, string Email, bool IsActive);
public record TenantConfig(string TenantId, int MaxTokenTtlSeconds);
public record AuthToken(string Token, int ExpiresIn);
