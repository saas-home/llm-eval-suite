public interface IUserRepository {
    Task<User?> GetByIdAsync(string userId);
    Task SaveAsync(User user);
}
