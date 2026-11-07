#![allow(dead_code)]
mod auth_service;
mod models;
mod repository;

use auth_service::AuthenticationService;
use models::{TenantConfig, User};
use repository::UserRepository;
use std::cell::RefCell;
use std::rc::Rc;

struct FakeRepo { users: Vec<User>, calls: Rc<RefCell<Vec<String>>> }

impl UserRepository for FakeRepo {
    fn get_by_id(&self, user_id: &str) -> Option<User> {
        self.calls.borrow_mut().push(user_id.to_string());
        self.users.iter().find(|u| u.id == user_id).cloned()
    }
}

fn main() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let repo = FakeRepo {
        users: vec![
            User { id: "u1".into(), email: "a@x.io".into(), is_active: true },
            User { id: "u2".into(), email: "b@x.io".into(), is_active: false },
        ],
        calls: Rc::clone(&calls),
    };
    let svc = AuthenticationService::new(repo, TenantConfig { tenant_id: "t1".into(), max_token_ttl_seconds: 900 });
    let tok = svc.authenticate("u1").expect("active user must get a token");
    assert_eq!(tok.token, "tok_u1", "active user token wrong");
    assert_eq!(tok.expires_in, 900, "token TTL must come from the injected config");
    assert!(svc.authenticate("u2").is_none(), "inactive user must get None");
    assert!(svc.authenticate("nope").is_none(), "unknown user must get None");
    assert_eq!(*calls.borrow(), vec!["u1", "u2", "nope"], "must look users up via the injected repository");
    println!("POLYGLOT_OK {}", std::env::var("EVAL_RUN_ID").unwrap_or_default());
}
