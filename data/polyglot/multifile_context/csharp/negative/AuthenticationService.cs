public class AuthenticationService {
    private readonly IUserRepository _userRepo;
    private readonly TenantConfig _config;

    public AuthenticationService(IUserRepository userRepo, TenantConfig config) {
        _userRepo = userRepo;
        _config = config;
    }

    public async Task<AuthToken?> AuthenticateAsync(string userId) {
        var user = await _userRepo.GetByIdAsync(userId);
        if (user is null) {
            return null;
        }
        return new AuthToken($"tok_{user.Id}", _config.MaxTokenTtlSeconds);
    }
}
