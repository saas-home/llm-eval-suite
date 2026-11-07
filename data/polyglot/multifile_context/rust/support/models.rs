#[derive(Clone, Debug)]
pub struct User { pub id: String, pub email: String, pub is_active: bool }

#[derive(Clone, Debug)]
pub struct TenantConfig { pub tenant_id: String, pub max_token_ttl_seconds: u32 }

#[derive(Clone, Debug)]
pub struct AuthToken { pub token: String, pub expires_in: u32 }
