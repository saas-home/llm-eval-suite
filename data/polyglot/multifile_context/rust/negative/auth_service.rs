use crate::models::{AuthToken, TenantConfig};
use crate::repository::UserRepository;

pub struct AuthenticationService<R: UserRepository> {
    user_repo: R,
    config: TenantConfig,
}

impl<R: UserRepository> AuthenticationService<R> {
    pub fn new(user_repo: R, config: TenantConfig) -> Self {
        Self { user_repo, config }
    }

    pub fn authenticate(&self, user_id: &str) -> Option<AuthToken> {
        let user = self.user_repo.get_by_id(user_id)?;
        Some(AuthToken { token: format!("tok_{}", user.id), expires_in: self.config.max_token_ttl_seconds })
    }
}
