static void Check(bool cond, string msg) { if (!cond) throw new Exception(msg); }

var repo = new FakeRepo();
await repo.SaveAsync(new User("u1", "a@x.io", true));
await repo.SaveAsync(new User("u2", "b@x.io", false));
var svc = new AuthenticationService(repo, new TenantConfig("t1", 900));
Check(await svc.AuthenticateAsync("u1") == new AuthToken("tok_u1", 900), "active user token wrong");
Check(await svc.AuthenticateAsync("u2") is null, "inactive user must get null");
Check(await svc.AuthenticateAsync("nope") is null, "unknown user must get null");
Check(repo.Calls.SequenceEqual(new[] { "u1", "u2", "nope" }), "must look users up via the injected repository");
Console.WriteLine("POLYGLOT_OK " + Environment.GetEnvironmentVariable("EVAL_RUN_ID"));

sealed class FakeRepo : IUserRepository {
    private readonly Dictionary<string, User> _users = new();
    public List<string> Calls { get; } = new();
    public Task<User?> GetByIdAsync(string userId) {
        Calls.Add(userId);
        return Task.FromResult(_users.TryGetValue(userId, out var u) ? u : null);
    }
    public Task SaveAsync(User user) { _users[user.Id] = user; return Task.CompletedTask; }
}
