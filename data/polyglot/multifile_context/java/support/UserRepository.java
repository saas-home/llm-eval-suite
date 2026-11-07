import java.util.Optional;

public interface UserRepository {
    Optional<User> getById(String userId);
    void save(User user);
}
