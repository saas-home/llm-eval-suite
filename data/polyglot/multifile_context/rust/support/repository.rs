use crate::models::User;

pub trait UserRepository {
    fn get_by_id(&self, user_id: &str) -> Option<User>;
}
