import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class Main {
    static final class FakeRepo implements UserRepository {
        final Map<String, User> users = new HashMap<>();
        final List<String> calls = new ArrayList<>();
        public Optional<User> getById(String userId) { calls.add(userId); return Optional.ofNullable(users.get(userId)); }
        public void save(User user) { users.put(user.id(), user); }
    }

    static void check(boolean cond, String msg) { if (!cond) throw new AssertionError(msg); }

    public static void main(String[] args) {
        FakeRepo repo = new FakeRepo();
        repo.save(new User("u1", "a@x.io", true));
        repo.save(new User("u2", "b@x.io", false));
        AuthenticationService svc = new AuthenticationService(repo, new TenantConfig("t1", 900));
        check(svc.authenticate("u1").equals(Optional.of(new AuthToken("tok_u1", 900))), "active user token wrong");
        check(svc.authenticate("u2").isEmpty(), "inactive user must get Optional.empty()");
        check(svc.authenticate("nope").isEmpty(), "unknown user must get Optional.empty()");
        check(repo.calls.equals(List.of("u1", "u2", "nope")), "must look users up via the injected repository");
        System.out.println("POLYGLOT_OK " + System.getenv("EVAL_RUN_ID"));
    }
}
