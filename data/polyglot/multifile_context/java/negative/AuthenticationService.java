import java.util.Optional;

public class AuthenticationService {
    private final UserRepository userRepo;
    private final TenantConfig config;

    public AuthenticationService(UserRepository userRepo, TenantConfig config) {
        this.userRepo = userRepo;
        this.config = config;
    }

    public Optional<AuthToken> authenticate(String userId) {
        return userRepo.getById(userId)
                .map(user -> new AuthToken("tok_" + user.id(), config.maxTokenTtlSeconds()));
    }
}
