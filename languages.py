"""
Polyglot Language Registry & Evaluation Fixtures for SWE Benchmark Suite (Tests 20-25)

Supports the 6 major industry software engineering ecosystems:
  1. Python      (AI/ML, Data Engineering, Backend APIs, DevOps)
  2. TypeScript  (Frontend, React/Vue/Next.js, Fullstack Web, Node.js)
  3. Go          (Cloud Microservices, Kubernetes, High-Throughput Distributed Systems)
  4. Java        (Enterprise Systems, Spring Boot, Monoliths)
  5. Rust        (High-Performance Engines, Systems, Memory Safety)
  6. C#          (.NET Core, Enterprise Web, Async Services)

Zero runtime dependencies: 100% Python standard library.
Adaptive execution:
  - If language compiler/runner is present on host and execution enabled: compiles and runs test harness.
  - If language compiler is absent: performs rigorous static structural AST/grammar verification.
"""

import os
import re
import sys
import ast
import shutil
import tempfile
import subprocess
from typing import Any, Dict, Optional

from client import DATA_DIR, assemble_python_script, harness_passed, run_model_code

SUPPORTED_LANGUAGES = ["python", "typescript", "go", "java", "rust", "csharp"]

LANGUAGE_ALIASES = {
    "py": "python",
    "python": "python",
    "ts": "typescript",
    "js": "typescript",
    "javascript": "typescript",
    "typescript": "typescript",
    "golang": "go",
    "go": "go",
    "java": "java",
    "rs": "rust",
    "rust": "rust",
    "cs": "csharp",
    "c#": "csharp",
    "csharp": "csharp",
    "polyglot": "polyglot",
    "all": "polyglot",
}

LANGUAGE_METADATA = {
    "python": {
        "name": "Python",
        "ext": ".py",
        "tool": "python3",
        "framework": "unittest",
        "typechecker": "mypy",
    },
    "typescript": {
        "name": "TypeScript",
        "ext": ".ts",
        "tool": "tsc",
        "framework": "jest",
        "typechecker": "tsc",
    },
    "go": {
        "name": "Go",
        "ext": ".go",
        "tool": "go",
        "framework": "testing",
        "typechecker": "go vet",
    },
    "java": {
        "name": "Java",
        "ext": ".java",
        "tool": "javac",
        "framework": "junit",
        "typechecker": "javac",
    },
    "rust": {
        "name": "Rust",
        "ext": ".rs",
        "tool": "rustc",
        "framework": "cargo test",
        "typechecker": "rustc",
    },
    "csharp": {
        "name": "C#",
        "ext": ".cs",
        "tool": "dotnet",
        "framework": "xUnit",
        "typechecker": "dotnet build",
    },
}


def normalize_language(lang_str: str) -> str:
    """Normalizes language string alias to canonical name."""
    if not lang_str:
        return "python"
    cleaned = str(lang_str).strip().lower().replace("-", "_")
    return LANGUAGE_ALIASES.get(cleaned, "python")


# ============================================================================
# 1. TEST 20: SURGICAL UNIFIED DIFF FIXTURES
# ============================================================================

_GO_PAYROLL = (
    "package payroll\n\n"
    "type Employee struct{ Salary float64 }\n\n"
    "func CalculatePayroll(employees []*Employee, taxRate float64, bonusPool float64) float64 {\n"
    "\ttotal := 0.0\n"
    "\tfor i := 0; i <= len(employees); i++ {\n"
    "\t\temp := employees[i]\n"
    "\t\tnet := emp.Salary * (1.0 - taxRate)\n"
    "\t\ttotal += net\n"
    "\t}\n"
    "\treturn total + bonusPool\n"
    "}\n"
)

DIFF_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "file_name": "payroll.py",
        "code": (
            "def calculate_payroll(employees, tax_rate, bonus_pool):\n"
            "    total = 0.0\n"
            "    for i in range(len(employees) + 1):\n"
            "        emp = employees[i]\n"
            "        salary = emp['salary']\n"
            "        net = salary * (1 - tax_rate)\n"
            "        total += net\n"
            "    return total + bonus_pool\n"
        ),
        "behavior_check": (
            "assert abs(calculate_payroll([{'salary': 100.0}, {'salary': 200.0}], 0.1, 5.0) - 275.0) < 1e-9, "
            "'off-by-one not fixed (wrong total)'\n"
            "for empty in (None, []):\n"
            "    r = calculate_payroll(empty, 0.1, 5.0)\n"
            "    assert r in (0.0, 5.0), f'None/empty employees not handled safely: {r!r}'\n"
            "print('DIFF_BEHAVIOR_VERIFIED', __import__('os').environ['EVAL_RUN_ID'])\n"
        ),
        "prompt": (
            "Here is an existing Python file `payroll.py` with an off-by-one index error and missing null check:\n\n"
            "```python\n"
            "def calculate_payroll(employees, tax_rate, bonus_pool):\n"
            "    total = 0.0\n"
            "    for i in range(len(employees) + 1):\n"
            "        emp = employees[i]\n"
            "        salary = emp['salary']\n"
            "        net = salary * (1 - tax_rate)\n"
            "        total += net\n"
            "    return total + bonus_pool\n"
            "```\n\n"
            "Fix the off-by-one bug and add a check to safely handle None/empty employees.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/payroll.py` and `+++ b/payroll.py`."
        ),
    },
    "typescript": {
        "file_name": "payroll.ts",
        "code": (
            "interface Employee { salary: number; }\n"
            "export function calculatePayroll(employees: Employee[], taxRate: number, bonusPool: number): number {\n"
            "    let total = 0;\n"
            "    for (let i = 0; i <= employees.length; i++) {\n"
            "        const emp = employees[i];\n"
            "        const net = emp.salary * (1 - taxRate);\n"
            "        total += net;\n"
            "    }\n"
            "    return total + bonusPool;\n"
            "}\n"
        ),
        "prompt": (
            "Here is an existing TypeScript file `payroll.ts` with an off-by-one loop error (`<= employees.length`) and missing null check:\n\n"
            "```typescript\n"
            "interface Employee { salary: number; }\n"
            "export function calculatePayroll(employees: Employee[], taxRate: number, bonusPool: number): number {\n"
            "    let total = 0;\n"
            "    for (let i = 0; i <= employees.length; i++) {\n"
            "        const emp = employees[i];\n"
            "        const net = emp.salary * (1 - taxRate);\n"
            "        total += net;\n"
            "    }\n"
            "    return total + bonusPool;\n"
            "}\n"
            "```\n\n"
            "Fix the off-by-one loop boundary and add null safety.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/payroll.ts` and `+++ b/payroll.ts`."
        ),
    },
    "go": {
        "file_name": "payroll.go",
        # gofmt output (tab indentation), as every real Go file is. With
        # 4-space indentation, a diff written in normal Go style (tabs) did
        # not match the file and failed git apply.
        "code": _GO_PAYROLL,
        "prompt": (
            "Here is an existing Go file `payroll.go` with an out-of-bounds index error (`i <= len(employees)`) and missing nil check:\n\n"
            "```go\n" + _GO_PAYROLL + "```\n\n"
            "Fix the out-of-bounds loop and handle nil pointers safely.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/payroll.go` and `+++ b/payroll.go`."
        ),
    },
    "java": {
        "file_name": "Payroll.java",
        "code": (
            "package com.payroll;\n"
            "import java.util.List;\n\n"
            "public class Payroll {\n"
            "    public static class Employee { public double salary; }\n"
            "    public static double calculatePayroll(List<Employee> employees, double taxRate, double bonusPool) {\n"
            "        double total = 0.0;\n"
            "        for (int i = 0; i <= employees.size(); i++) {\n"
            "            Employee emp = employees.get(i);\n"
            "            double net = emp.salary * (1.0 - taxRate);\n"
            "            total += net;\n"
            "        }\n"
            "        return total + bonusPool;\n"
            "    }\n"
            "}\n"
        ),
        "prompt": (
            "Here is an existing Java file `Payroll.java` with an `IndexOutOfBoundsException` (`i <= employees.size()`) and missing null check:\n\n"
            "```java\n"
            "package com.payroll;\n"
            "import java.util.List;\n\n"
            "public class Payroll {\n"
            "    public static class Employee { public double salary; }\n"
            "    public static double calculatePayroll(List<Employee> employees, double taxRate, double bonusPool) {\n"
            "        double total = 0.0;\n"
            "        for (int i = 0; i <= employees.size(); i++) {\n"
            "            Employee emp = employees.get(i);\n"
            "            double net = emp.salary * (1.0 - taxRate);\n"
            "            total += net;\n"
            "        }\n"
            "        return total + bonusPool;\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "Fix the off-by-one loop and add null safety.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/Payroll.java` and `+++ b/Payroll.java`."
        ),
    },
    "rust": {
        "file_name": "payroll.rs",
        "code": (
            "pub struct Employee { pub salary: f64 }\n\n"
            "pub fn calculate_payroll(employees: &[Option<Employee>], tax_rate: f64, bonus_pool: f64) -> f64 {\n"
            "    let mut total = 0.0;\n"
            "    for i in 0..=employees.len() {\n"
            "        let emp = &employees[i];\n"
            "        if let Some(e) = emp {\n"
            "            let net = e.salary * (1.0 - tax_rate);\n"
            "            total += net;\n"
            "        }\n"
            "    }\n"
            "    total + bonus_pool\n"
            "}\n"
        ),
        "prompt": (
            "Here is an existing Rust file `payroll.rs` with a panic on out-of-bounds indexing (`0..=employees.len()`):\n\n"
            "```rust\n"
            "pub struct Employee { pub salary: f64 }\n\n"
            "pub fn calculate_payroll(employees: &[Option<Employee>], tax_rate: f64, bonus_pool: f64) -> f64 {\n"
            "    let mut total = 0.0;\n"
            "    for i in 0..=employees.len() {\n"
            "        let emp = &employees[i];\n"
            "        if let Some(e) = emp {\n"
            "            let net = e.salary * (1.0 - tax_rate);\n"
            "            total += net;\n"
            "        }\n"
            "    }\n"
            "    total + bonus_pool\n"
            "}\n"
            "```\n\n"
            "Fix the range slice index to prevent out-of-bounds panic.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/payroll.rs` and `+++ b/payroll.rs`."
        ),
    },
    "csharp": {
        "file_name": "Payroll.cs",
        "code": (
            "namespace PayrollSystem;\n"
            "using System.Collections.Generic;\n\n"
            "public class Employee { public decimal Salary { get; set; } }\n\n"
            "public static class PayrollCalculator {\n"
            "    public static decimal CalculatePayroll(List<Employee> employees, decimal taxRate, decimal bonusPool) {\n"
            "        decimal total = 0m;\n"
            "        for (int i = 0; i <= employees.Count; i++) {\n"
            "            var emp = employees[i];\n"
            "            decimal net = emp.Salary * (1m - taxRate);\n"
            "            total += net;\n"
            "        }\n"
            "        return total + bonusPool;\n"
            "    }\n"
            "}\n"
        ),
        "prompt": (
            "Here is an existing C# file `Payroll.cs` with an `ArgumentOutOfRangeException` (`i <= employees.Count`) and missing null check:\n\n"
            "```csharp\n"
            "namespace PayrollSystem;\n"
            "using System.Collections.Generic;\n\n"
            "public class Employee { public decimal Salary { get; set; } }\n\n"
            "public static class PayrollCalculator {\n"
            "    public static decimal CalculatePayroll(List<Employee> employees, decimal taxRate, decimal bonusPool) {\n"
            "        decimal total = 0m;\n"
            "        for (int i = 0; i <= employees.Count; i++) {\n"
            "            var emp = employees[i];\n"
            "            decimal net = emp.Salary * (1m - taxRate);\n"
            "            total += net;\n"
            "        }\n"
            "        return total + bonusPool;\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "Fix the index boundary and null handling.\n"
            "Provide a surgical unified git diff (patch) to fix this. "
            "Your response MUST contain a valid unified diff block starting with `--- a/Payroll.cs` and `+++ b/Payroll.cs`."
        ),
    },
}


# ============================================================================
# 2. TEST 21: MULTI-FILE CROSS-REFERENCE & DEPENDENCY INJECTION FIXTURES
# ============================================================================

MULTIFILE_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "prompt": (
            "You are working across multiple files in a repository.\n\n"
            "# File 1: models.py\n"
            "from dataclasses import dataclass\n"
            "from typing import Optional\n"
            "@dataclass(frozen=True)\n"
            "class User:\n"
            "    id: str\n"
            "    email: str\n"
            "    is_active: bool\n\n"
            "@dataclass(frozen=True)\n"
            "class TenantConfig:\n"
            "    tenant_id: str\n"
            "    max_token_ttl_seconds: int = 3600\n\n"
            "@dataclass(frozen=True)\n"
            "class AuthToken:\n"
            "    token: str\n"
            "    expires_in: int\n\n"
            "# File 2: repository.py\n"
            "from abc import ABC, abstractmethod\n"
            "from typing import Optional, List\n"
            "from models import User\n"
            "class AbstractUserRepository(ABC):\n"
            "    @abstractmethod\n"
            "    def get_by_id(self, user_id: str) -> Optional[User]: ...\n"
            "    @abstractmethod\n"
            "    def save(self, user: User) -> None: ...\n"
            "    @abstractmethod\n"
            "    def list_active(self) -> List[User]: ...\n\n"
            "# File 3: auth_service.py (INCOMPLETE)\n"
            "# Complete AuthenticationService with:\n"
            "# 1. Constructor dependency injection accepting `user_repo: AbstractUserRepository` and `config: TenantConfig`.\n"
            "# 2. `authenticate(user_id: str) -> Optional[AuthToken]`:\n"
            "#    - Retrieves user using `user_repo.get_by_id(user_id)`.\n"
            "#    - If user exists and is active, returns `AuthToken(token=f'tok_{user.id}', expires_in=config.max_token_ttl_seconds)`.\n"
            "#    - Otherwise returns None.\n"
            "# Output the complete, syntactically valid `auth_service.py` in a Python code block."
        ),
        "behavior_check": (
            "from typing import Optional, List\n"
            "from models import User, TenantConfig, AuthToken\n"
            "from repository import AbstractUserRepository\n"
            "from auth_service import AuthenticationService\n"
            "class FakeRepo(AbstractUserRepository):\n"
            "    def __init__(self, users): self.users = {u.id: u for u in users}; self.calls = []\n"
            "    def get_by_id(self, user_id): self.calls.append(user_id); return self.users.get(user_id)\n"
            "    def save(self, user): self.users[user.id] = user\n"
            "    def list_active(self): return [u for u in self.users.values() if u.is_active]\n"
            "repo = FakeRepo([User('u1', 'a@x.io', True), User('u2', 'b@x.io', False)])\n"
            "svc = AuthenticationService(user_repo=repo, config=TenantConfig('t1', max_token_ttl_seconds=900))\n"
            "assert svc.authenticate('u1') == AuthToken(token='tok_u1', expires_in=900), 'active user token wrong'\n"
            "assert svc.authenticate('u2') is None, 'inactive user must get None'\n"
            "assert svc.authenticate('nope') is None, 'unknown user must get None'\n"
            "assert repo.calls == ['u1', 'u2', 'nope'], 'must look users up via the injected repository'\n"
            "print('MULTIFILE_BEHAVIOR_VERIFIED', __import__('os').environ['EVAL_RUN_ID'])\n"
        ),
        "required_symbols": ["AuthenticationService", "get_by_id", "AuthToken"],
    },
    "typescript": {
        "prompt": (
            "You are working across multiple files in a TypeScript repository.\n\n"
            "// File 1: models.ts\n"
            "export interface User { id: string; email: string; isActive: boolean; }\n"
            "export interface TenantConfig { tenantId: string; maxTokenTtlSeconds: number; }\n"
            "export interface AuthToken { token: string; expiresIn: number; }\n\n"
            "// File 2: repository.ts\n"
            "import { User } from './models';\n"
            "export interface IUserRepository {\n"
            "    getById(userId: string): Promise<User | null>;\n"
            "    save(user: User): Promise<void>;\n"
            "    listActive(): Promise<User[]>;\n"
            "}\n\n"
            "// File 3: authService.ts (INCOMPLETE)\n"
            "// Implement `AuthenticationService` with:\n"
            "// 1. Constructor dependency injection accepting `userRepo: IUserRepository` and `config: TenantConfig`.\n"
            "// 2. `async authenticate(userId: string): Promise<AuthToken | null>`:\n"
            "//    - Resolves user via `userRepo.getById(userId)`.\n"
            "//    - If user exists and is active, returns `{ token: `tok_${user.id}`, expiresIn: config.maxTokenTtlSeconds }`.\n"
            "//    - Otherwise returns null.\n"
            "// Output the complete `authService.ts` in a ```typescript code block."
        ),
        "required_symbols": ["AuthenticationService", "getById", "AuthToken"],
    },
    "go": {
        "prompt": (
            "You are working across multiple packages in a Go repository.\n\n"
            "// File 1: models.go\n"
            "package models\n"
            "type User struct { ID string; Email string; IsActive bool }\n"
            "type TenantConfig struct { TenantID string; MaxTokenTTLSeconds int }\n"
            "type AuthToken struct { Token string; ExpiresIn int }\n\n"
            "// File 2: repository.go\n"
            "package repository\n"
            "import \"models\"\n"
            "type UserRepository interface {\n"
            "    GetByID(userID string) (*models.User, error)\n"
            "    Save(user *models.User) error\n"
            "    ListActive() ([]*models.User, error)\n"
            "}\n\n"
            "// File 3: auth_service.go (INCOMPLETE)\n"
            "// Implement `AuthService` struct and constructor with:\n"
            "// 1. Dependency injection accepting `repo UserRepository` and `config models.TenantConfig`.\n"
            "// 2. `Authenticate(userID string) (*models.AuthToken, error)`:\n"
            "//    - Resolves user via `repo.GetByID(userID)`.\n"
            "//    - If user exists and IsActive, returns `&models.AuthToken{Token: \"tok_\" + user.ID, ExpiresIn: config.MaxTokenTTLSeconds}, nil`.\n"
            "//    - Otherwise returns nil, nil (or error).\n"
            "// Output complete Go code in a ```go block."
        ),
        "required_symbols": ["AuthService", "GetByID", "AuthToken"],
    },
    "java": {
        "prompt": (
            "You are working across multiple files in a Java Spring Boot repository.\n\n"
            "// File 1: Models.java\n"
            "public record User(String id, String email, boolean isActive) {}\n"
            "public record TenantConfig(String tenantId, int maxTokenTtlSeconds) {}\n"
            "public record AuthToken(String token, int expiresIn) {}\n\n"
            "// File 2: UserRepository.java\n"
            "import java.util.Optional;\n"
            "public interface UserRepository {\n"
            "    Optional<User> getById(String userId);\n"
            "    void save(User user);\n"
            "}\n\n"
            "// File 3: AuthenticationService.java (INCOMPLETE)\n"
            "// Implement `AuthenticationService` with:\n"
            "// 1. Constructor dependency injection accepting `UserRepository userRepo` and `TenantConfig config`.\n"
            "// 2. `public Optional<AuthToken> authenticate(String userId)`:\n"
            "//    - Resolves user via `userRepo.getById(userId)`.\n"
            "//    - If user exists and isActive(), returns Optional of AuthToken with token \"tok_\" + user.id() and config.maxTokenTtlSeconds().\n"
            "//    - Otherwise returns Optional.empty().\n"
            "// Output complete Java code in a ```java block."
        ),
        "required_symbols": ["AuthenticationService", "getById", "AuthToken"],
    },
    "rust": {
        "prompt": (
            "You are working across multiple files in a Rust repository.\n\n"
            "// File 1: models.rs\n"
            "#[derive(Clone, Debug)]\n"
            "pub struct User { pub id: String, pub email: String, pub is_active: bool }\n"
            "#[derive(Clone, Debug)]\n"
            "pub struct TenantConfig { pub tenant_id: String, pub max_token_ttl_seconds: u32 }\n"
            "#[derive(Clone, Debug)]\n"
            "pub struct AuthToken { pub token: String, pub expires_in: u32 }\n\n"
            "// File 2: repository.rs\n"
            "use crate::models::User;\n"
            "pub trait UserRepository {\n"
            "    fn get_by_id(&self, user_id: &str) -> Option<User>;\n"
            "}\n\n"
            "// File 3: auth_service.rs (INCOMPLETE)\n"
            "// Implement `AuthenticationService<R: UserRepository>` with:\n"
            "// 1. `new(user_repo: R, config: TenantConfig) -> Self` constructor DI.\n"
            "// 2. `pub fn authenticate(&self, user_id: &str) -> Option<AuthToken>`:\n"
            "//    - Resolves user via `self.user_repo.get_by_id(user_id)`.\n"
            "//    - If user exists and is_active is true, returns Some(AuthToken { token: format!(\"tok_{}\", user.id), expires_in: self.config.max_token_ttl_seconds }).\n"
            "//    - Otherwise returns None.\n"
            "// Output complete Rust code in a ```rust block."
        ),
        "required_symbols": ["AuthenticationService", "get_by_id", "AuthToken"],
    },
    "csharp": {
        "prompt": (
            "You are working across multiple files in a C# .NET solution.\n\n"
            "// File 1: Models.cs\n"
            "public record User(string Id, string Email, bool IsActive);\n"
            "public record TenantConfig(string TenantId, int MaxTokenTtlSeconds);\n"
            "public record AuthToken(string Token, int ExpiresIn);\n\n"
            "// File 2: IUserRepository.cs\n"
            "public interface IUserRepository {\n"
            "    Task<User?> GetByIdAsync(string userId);\n"
            "    Task SaveAsync(User user);\n"
            "}\n\n"
            "// File 3: AuthenticationService.cs (INCOMPLETE)\n"
            "// Implement `AuthenticationService` with:\n"
            "// 1. Constructor dependency injection accepting `IUserRepository userRepo` and `TenantConfig config`.\n"
            "// 2. `public async Task<AuthToken?> AuthenticateAsync(string userId)`:\n"
            "//    - Resolves user via `await _userRepo.GetByIdAsync(userId)`.\n"
            "//    - If user exists and IsActive, returns new AuthToken($\"tok_{user.Id}\", _config.MaxTokenTtlSeconds).\n"
            "//    - Otherwise returns null.\n"
            "// Output complete C# code in a ```csharp block."
        ),
        "required_symbols": ["AuthenticationService", "GetByIdAsync", "AuthToken"],
    },
}


# ============================================================================
# 3. TEST 22: SECURITY REVIEW & OWASP VULNERABILITY AUDIT FIXTURES
# ============================================================================

SECURITY_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "prompt": (
            "Perform a rigorous security code review of the following three Python pull request snippets.\n\n"
            "Snippet A:\n"
            "```python\n"
            "def query_user(cursor, username):\n"
            "    query = f\"SELECT id, email FROM users WHERE username = '{username}'\"\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchone()\n"
            "```\n\n"
            "Snippet B:\n"
            "```python\n"
            "def transfer_funds(accounts, from_acc, to_acc, amount):\n"
            "    if accounts[from_acc] >= amount:\n"
            "        time.sleep(0.001)  # simulated network I/O\n"
            "        accounts[from_acc] -= amount\n"
            "        accounts[to_acc] += amount\n"
            "        return True\n"
            "    return False\n"
            "```\n\n"
            "Snippet C:\n"
            "```python\n"
            "def get_user_email(cursor, user_id: int):\n"
            "    cursor.execute(\"SELECT email FROM users WHERE id = %s\", (user_id,))\n"
            "    row = cursor.fetchone()\n"
            "    return row[0] if row else None\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
    "typescript": {
        "prompt": (
            "Perform a rigorous security code review of the following three TypeScript/Node.js pull request snippets.\n\n"
            "Snippet A:\n"
            "```typescript\n"
            "async function findUser(db: any, username: string) {\n"
            "    const query = `SELECT id, email FROM users WHERE username = '${username}'`;\n"
            "    return await db.query(query);\n"
            "}\n"
            "```\n\n"
            "Snippet B:\n"
            "```typescript\n"
            "async function transferBalance(accounts: Map<string, number>, fromId: string, toId: string, amt: number) {\n"
            "    if ((accounts.get(fromId) || 0) >= amt) {\n"
            "        await new Promise(r => setTimeout(r, 1)); // async gap\n"
            "        accounts.set(fromId, accounts.get(fromId)! - amt);\n"
            "        accounts.set(toId, (accounts.get(toId) || 0) + amt);\n"
            "        return true;\n"
            "    }\n"
            "    return false;\n"
            "}\n"
            "```\n\n"
            "Snippet C:\n"
            "```typescript\n"
            "async function getUserEmail(db: Pool, userId: number): Promise<string | null> {\n"
            "    const res = await db.query('SELECT email FROM users WHERE id = $1', [userId]);\n"
            "    return res.rows.length ? res.rows[0].email : null;\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
    "go": {
        "prompt": (
            "Perform a rigorous security code review of the following three Go pull request snippets.\n\n"
            "Snippet A:\n"
            "```go\n"
            "func GetUser(db *sql.DB, username string) (*User, error) {\n"
            "    query := fmt.Sprintf(\"SELECT id, email FROM users WHERE username = '%s'\", username)\n"
            "    return db.QueryRow(query).Scan(&id, &email)\n"
            "}\n"
            "```\n\n"
            "Snippet B:\n"
            "```go\n"
            "var balances = make(map[string]int)\n"
            "func Transfer(from, to string, amount int) bool {\n"
            "    if balances[from] >= amount {\n"
            "        time.Sleep(1 * time.Millisecond)\n"
            "        balances[from] -= amount\n"
            "        balances[to] += amount\n"
            "        return true\n"
            "    }\n"
            "    return false\n"
            "}\n"
            "```\n\n"
            "Snippet C:\n"
            "```go\n"
            "func GetUserEmail(db *sql.DB, userID int64) (string, error) {\n"
            "    var email string\n"
            "    err := db.QueryRow(\"SELECT email FROM users WHERE id = $1\", userID).Scan(&email)\n"
            "    return email, err\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
    "java": {
        "prompt": (
            "Perform a rigorous security code review of the following three Java pull request snippets.\n\n"
            "Snippet A:\n"
            "```java\n"
            "public User findUser(Connection conn, String username) throws SQLException {\n"
            "    String sql = \"SELECT id, email FROM users WHERE username = '\" + username + \"'\";\n"
            "    Statement stmt = conn.createStatement();\n"
            "    return stmt.executeQuery(sql);\n"
            "}\n"
            "```\n\n"
            "Snippet B:\n"
            "```java\n"
            "private Map<String, Double> accounts = new HashMap<>();\n"
            "public boolean transfer(String from, String to, double amount) {\n"
            "    if (accounts.get(from) >= amount) {\n"
            "        try { Thread.sleep(1); } catch (Exception ignored) {}\n"
            "        accounts.put(from, accounts.get(from) - amount);\n"
            "        accounts.put(to, accounts.get(to) + amount);\n"
            "        return true;\n"
            "    }\n"
            "    return false;\n"
            "}\n"
            "```\n\n"
            "Snippet C:\n"
            "```java\n"
            "public Optional<String> getUserEmail(Connection conn, long userId) throws SQLException {\n"
            "    try (PreparedStatement ps = conn.prepareStatement(\"SELECT email FROM users WHERE id = ?\")) {\n"
            "        ps.setLong(1, userId);\n"
            "        try (ResultSet rs = ps.executeQuery()) {\n"
            "            return rs.next() ? Optional.of(rs.getString(\"email\")) : Optional.empty();\n"
            "        }\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
    "rust": {
        "prompt": (
            "Perform a rigorous security code review of the following three Rust pull request snippets.\n\n"
            "Snippet A:\n"
            "```rust\n"
            "pub fn query_user(conn: &Connection, username: &str) -> Result<User, Error> {\n"
            "    let query = format!(\"SELECT id, email FROM users WHERE username = '{}'\", username);\n"
            "    conn.execute(&query, [])\n"
            "}\n"
            "```\n\n"
            "Snippet B:\n"
            "```rust\n"
            "static mut BALANCES: Option<HashMap<String, i64>> = None;\n"
            "pub unsafe fn transfer(from: &str, to: &str, amount: i64) -> bool {\n"
            "    if let Some(ref mut map) = BALANCES {\n"
            "        if *map.get(from).unwrap_or(&0) >= amount {\n"
            "            std::thread::sleep(std::time::Duration::from_millis(1));\n"
            "            *map.entry(from.to_string()).or_default() -= amount;\n"
            "            *map.entry(to.to_string()).or_default() += amount;\n"
            "            return true;\n"
            "        }\n"
            "    }\n"
            "    false\n"
            "}\n"
            "```\n\n"
            "Snippet C:\n"
            "```rust\n"
            "use rusqlite::{Connection, OptionalExtension};\n"
            "\n"
            "pub fn get_user_email(conn: &Connection, user_id: i64) -> rusqlite::Result<Option<String>> {\n"
            "    conn.query_row(\"SELECT email FROM users WHERE id = ?1\", [user_id], |row| row.get(0))\n"
            "        .optional()\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
    "csharp": {
        "prompt": (
            "Perform a rigorous security code review of the following three C# pull request snippets.\n\n"
            "Snippet A:\n"
            "```csharp\n"
            "public async Task<User?> GetUserAsync(SqlConnection conn, string username) {\n"
            "    string query = $\"SELECT id, email FROM users WHERE username = '{username}'\";\n"
            "    using var cmd = new SqlCommand(query, conn);\n"
            "    return await cmd.ExecuteReaderAsync();\n"
            "}\n"
            "```\n\n"
            "Snippet B:\n"
            "```csharp\n"
            "private Dictionary<string, decimal> _balances = new();\n"
            "public async Task<bool> TransferAsync(string from, string to, decimal amount) {\n"
            "    if (_balances[from] >= amount) {\n"
            "        await Task.Delay(1);\n"
            "        _balances[from] -= amount;\n"
            "        _balances[to] += amount;\n"
            "        return true;\n"
            "    }\n"
            "    return false;\n"
            "}\n"
            "```\n\n"
            "Snippet C:\n"
            "```csharp\n"
            "public string? GetUserEmail(SqlConnection conn, int userId) {\n"
            "    using var cmd = new SqlCommand(\"SELECT Email FROM Users WHERE Id = @id\", conn);\n"
            "    cmd.Parameters.AddWithValue(\"@id\", userId);\n"
            "    return cmd.ExecuteScalar() as string;\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
}


# ============================================================================
# 4. TEST 23: UNIT TEST SYNTHESIS WITH MOCKS FIXTURES
# ============================================================================

UNIT_TEST_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "source": (
            "class StripeGateway:\n"
            "    def charge(self, amount: int, currency: str, token: str) -> dict:\n"
            "        raise NotImplementedError('Network call to Stripe API')\n\n"
            "class NotificationClient:\n"
            "    def send_receipt(self, email: str, charge_id: str) -> bool:\n"
            "        raise NotImplementedError('Network call to Email service')\n\n"
            "class PaymentProcessor:\n"
            "    def __init__(self, gateway: StripeGateway, notifier: NotificationClient):\n"
            "        self.gateway = gateway\n"
            "        self.notifier = notifier\n\n"
            "    def process_order(self, email: str, amount: int, token: str) -> bool:\n"
            "        if amount <= 0:\n"
            "            raise ValueError('Amount must be positive')\n"
            "        resp = self.gateway.charge(amount, 'USD', token)\n"
            "        if resp.get('status') == 'succeeded':\n"
            "            self.notifier.send_receipt(email, resp['id'])\n"
            "            return True\n"
            "        return False\n"
        ),
        "prompt": (
            "Given the following Python production code:\n\n"
            "```python\n"
            "class StripeGateway:\n"
            "    def charge(self, amount: int, currency: str, token: str) -> dict:\n"
            "        raise NotImplementedError('Network call to Stripe API')\n\n"
            "class NotificationClient:\n"
            "    def send_receipt(self, email: str, charge_id: str) -> bool:\n"
            "        raise NotImplementedError('Network call to Email service')\n\n"
            "class PaymentProcessor:\n"
            "    def __init__(self, gateway: StripeGateway, notifier: NotificationClient):\n"
            "        self.gateway = gateway\n"
            "        self.notifier = notifier\n\n"
            "    def process_order(self, email: str, amount: int, token: str) -> bool:\n"
            "        if amount <= 0:\n"
            "            raise ValueError('Amount must be positive')\n"
            "        resp = self.gateway.charge(amount, 'USD', token)\n"
            "        if resp.get('status') == 'succeeded':\n"
            "            self.notifier.send_receipt(email, resp['id'])\n"
            "            return True\n"
            "        return False\n"
            "```\n\n"
            "Generate a complete Python `unittest.TestCase` suite testing `PaymentProcessor.process_order`.\n"
            "Requirements:\n"
            "1. Mock `StripeGateway` and `NotificationClient` using `unittest.mock.MagicMock` or `patch`.\n"
            "2. Test successful charge (`status == 'succeeded'`) verifies `send_receipt` was called and returns True.\n"
            "3. Test failed charge (`status == 'declined'`) verifies `send_receipt` was NOT called and returns False.\n"
            "4. Test negative amount raises `ValueError`.\n"
            "Output ONLY the test script in a ```python block."
        ),
        "mock_keywords": ["mock", "magicmock", "patch"],
    },
    "typescript": {
        "source": (
            "export interface StripeGateway { charge(amount: number, currency: string, token: string): Promise<{ status: string; id: string }>; }\n"
            "export interface NotificationClient { sendReceipt(email: string, chargeId: string): Promise<boolean>; }\n"
            "export class PaymentProcessor {\n"
            "    constructor(private gateway: StripeGateway, private notifier: NotificationClient) {}\n"
            "    async processOrder(email: string, amount: number, token: string): Promise<boolean> {\n"
            "        if (amount <= 0) throw new Error('Amount must be positive');\n"
            "        const resp = await this.gateway.charge(amount, 'USD', token);\n"
            "        if (resp.status === 'succeeded') {\n"
            "            await this.notifier.sendReceipt(email, resp.id);\n"
            "            return true;\n"
            "        }\n"
            "        return false;\n"
            "    }\n"
            "}\n"
        ),
        "prompt": (
            "Given the following TypeScript code:\n\n"
            "```typescript\n"
            "export interface StripeGateway { charge(amount: number, currency: string, token: string): Promise<{ status: string; id: string }>; }\n"
            "export interface NotificationClient { sendReceipt(email: string, chargeId: string): Promise<boolean>; }\n"
            "export class PaymentProcessor {\n"
            "    constructor(private gateway: StripeGateway, private notifier: NotificationClient) {}\n"
            "    async processOrder(email: string, amount: number, token: string): Promise<boolean> {\n"
            "        if (amount <= 0) throw new Error('Amount must be positive');\n"
            "        const resp = await this.gateway.charge(amount, 'USD', token);\n"
            "        if (resp.status === 'succeeded') {\n"
            "            await this.notifier.sendReceipt(email, resp.id);\n"
            "            return true;\n"
            "        }\n"
            "        return false;\n"
            "    }\n"
            "}\n"
            "```\n\n"
            "Generate a complete Jest / Vitest unit test suite testing `PaymentProcessor.processOrder`.\n"
            "Requirements:\n"
            "1. Mock `StripeGateway` and `NotificationClient` using `jest.fn()` or `vi.fn()`.\n"
            "2. Test successful charge verifies `sendReceipt` was called with correct args and returns true.\n"
            "3. Test declined charge verifies `sendReceipt` was NOT called and returns false.\n"
            "4. Test negative amount throws Error.\n"
            "Output ONLY the test script in a ```typescript block."
        ),
        "mock_keywords": ["jest.fn", "vi.fn", "mock", "spyond"],
    },
    "go": {
        "source": (
            "package payment\n"
            "import \"errors\"\n"
            "type StripeGateway interface { Charge(amount int, currency, token string) (map[string]string, error) }\n"
            "type NotificationClient interface { SendReceipt(email, chargeID string) bool }\n"
            "type PaymentProcessor struct { Gateway StripeGateway; Notifier NotificationClient }\n"
            "func (p *PaymentProcessor) ProcessOrder(email string, amount int, token string) (bool, error) {\n"
            "    if amount <= 0 { return false, errors.New(\"amount must be positive\") }\n"
            "    resp, err := p.Gateway.Charge(amount, \"USD\", token)\n"
            "    if err != nil { return false, err }\n"
            "    if resp[\"status\"] == \"succeeded\" {\n"
            "        p.Notifier.SendReceipt(email, resp[\"id\"])\n"
            "        return true, nil\n"
            "    }\n"
            "    return false, nil\n"
            "}\n"
        ),
        "prompt": (
            "Given the following Go code:\n\n"
            "```go\n"
            "package payment\n"
            "import \"errors\"\n"
            "type StripeGateway interface { Charge(amount int, currency, token string) (map[string]string, error) }\n"
            "type NotificationClient interface { SendReceipt(email, chargeID string) bool }\n"
            "type PaymentProcessor struct { Gateway StripeGateway; Notifier NotificationClient }\n"
            "func (p *PaymentProcessor) ProcessOrder(email string, amount int, token string) (bool, error) {\n"
            "    if amount <= 0 { return false, errors.New(\"amount must be positive\") }\n"
            "    resp, err := p.Gateway.Charge(amount, \"USD\", token)\n"
            "    if err != nil { return false, err }\n"
            "    if resp[\"status\"] == \"succeeded\" {\n"
            "        p.Notifier.SendReceipt(email, resp[\"id\"])\n"
            "        return true, nil\n"
            "    }\n"
            "    return false, nil\n"
            "}\n"
            "```\n\n"
            "Generate a complete Go unit test file using standard `testing.T`.\n"
            "Requirements:\n"
            "1. Implement mock structs for `StripeGateway` and `NotificationClient`.\n"
            "2. Test successful charge verifies SendReceipt call and returns true.\n"
            "3. Test failed charge verifies SendReceipt was NOT called and returns false.\n"
            "4. Test negative amount returns an error.\n"
            "Output ONLY the Go test code in a ```go block."
        ),
        "mock_keywords": ["mock", "struct", "test", "testing.t"],
    },
    "java": {
        "source": (
            "package com.payment;\n"
            "public class PaymentProcessor {\n"
            "    public interface StripeGateway { ChargeResult charge(int amount, String currency, String token); }\n"
            "    public interface NotificationClient { boolean sendReceipt(String email, String chargeId); }\n"
            "    public record ChargeResult(String status, String id) {}\n"
            "    private final StripeGateway gateway; private final NotificationClient notifier;\n"
            "    public PaymentProcessor(StripeGateway gateway, NotificationClient notifier) { this.gateway = gateway; this.notifier = notifier; }\n"
            "    public boolean processOrder(String email, int amount, String token) {\n"
            "        if (amount <= 0) throw new IllegalArgumentException(\"Amount must be positive\");\n"
            "        var res = gateway.charge(amount, \"USD\", token);\n"
            "        if (\"succeeded\".equals(res.status())) { notifier.sendReceipt(email, res.id()); return true; }\n"
            "        return false;\n"
            "    }\n"
            "}\n"
        ),
        "prompt": (
            "Given the `PaymentProcessor` class in Java with `StripeGateway` and `NotificationClient` dependencies,\n"
            "generate a complete JUnit 5 test class using Mockito (`Mockito.mock()`, `when(...).thenReturn(...)`, `verify(...)`).\n"
            "Requirements:\n"
            "1. Test successful charge returns true and verifies `sendReceipt` called.\n"
            "2. Test declined charge returns false and verifies `sendReceipt` never called.\n"
            "3. Test non-positive amount throws `IllegalArgumentException`.\n"
            "Output ONLY the Java test class in a ```java block."
        ),
        "mock_keywords": ["mockito", "@test", "mock", "verify"],
    },
    "rust": {
        "source": (
            "pub trait StripeGateway { fn charge(&self, amount: u32, currency: &str, token: &str) -> Result<String, String>; }\n"
            "pub trait NotificationClient { fn send_receipt(&self, email: &str, charge_id: &str) -> bool; }\n"
            "pub struct PaymentProcessor<'a, G: StripeGateway, N: NotificationClient> { gateway: &'a G, notifier: &'a N }\n"
        ),
        "prompt": (
            "Given the `PaymentProcessor` in Rust taking `StripeGateway` and `NotificationClient` trait references,\n"
            "generate a complete Rust `#[cfg(test)] mod tests` suite.\n"
            "Requirements:\n"
            "1. Implement mock structs for `StripeGateway` and `NotificationClient`.\n"
            "2. `test_process_order_success`: verifies receipt sent and returns true.\n"
            "3. `test_process_order_declined`: verifies receipt not sent.\n"
            "4. `test_process_order_negative_amount`: verifies error returned.\n"
            "Output ONLY the Rust test code in a ```rust block."
        ),
        "mock_keywords": ["#[test]", "assert", "mock", "impl"],
    },
    "csharp": {
        "source": (
            "public interface IStripeGateway { Task<ChargeResponse> ChargeAsync(decimal amount, string currency, string token); }\n"
            "public interface INotificationClient { Task<bool> SendReceiptAsync(string email, string chargeId); }\n"
            "public class PaymentProcessor { ... }\n"
        ),
        "prompt": (
            "Given the `PaymentProcessor` in C# depending on `IStripeGateway` and `INotificationClient`,\n"
            "generate an xUnit test class using `Moq` (`new Mock<IStripeGateway>()`, `.Setup(...)`, `.Verify(...)`).\n"
            "Requirements:\n"
            "1. Test successful charge returns true and verifies SendReceiptAsync was called.\n"
            "2. Test declined charge returns false and verifies SendReceiptAsync was NOT called.\n"
            "3. Test non-positive amount throws `ArgumentException`.\n"
            "Output ONLY the C# test code in a ```csharp block."
        ),
        "mock_keywords": ["mock", "fact", "setup", "verify"],
    },
}


# ============================================================================
# 5. TEST 24: STRICT TYPING & GENERICS FIXTURES
# ============================================================================

TYPING_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "prompt": (
            "Write a generic, immutable data transformation pipeline class in Python with 100% strict type annotations.\n"
            "Requirements:\n"
            "1. Define generic `TypeVar('T')` and `TypeVar('U')`.\n"
            "2. Implement `Pipeline(Generic[T])` with `__init__(self, data: List[T]) -> None`.\n"
            "3. Implement `map(self, fn: Callable[[T], U]) -> 'Pipeline[U]'`.\n"
            "4. Implement `filter(self, predicate: Callable[[T], bool]) -> 'Pipeline[T]'`.\n"
            "5. Implement `execute(self) -> List[T]`.\n"
            "All methods, parameters, and return types must be fully typed without Any or untyped defs.\n"
            "Output ONLY the python code in a ```python block."
        ),
        "required_patterns": [r"TypeVar", r"Generic\[", r"Callable\[", r"->"],
    },
    "typescript": {
        "prompt": (
            "Write a generic, immutable data transformation pipeline class in TypeScript with 100% strict type annotations.\n"
            "Requirements:\n"
            "1. Implement `class Pipeline<T>`.\n"
            "2. Constructor accepts `private readonly data: ReadonlyArray<T>`.\n"
            "3. Implement `map<U>(fn: (item: T) => U): Pipeline<U>`.\n"
            "4. Implement `filter(predicate: (item: T) => boolean): Pipeline<T>`.\n"
            "5. Implement `execute(): T[]`.\n"
            "No `any` types allowed; strictly typed generics.\n"
            "Output ONLY the TypeScript code in a ```typescript block."
        ),
        "required_patterns": [r"class Pipeline<T>", r"map<U>", r"filter", r"execute\(\)"],
    },
    "go": {
        "prompt": (
            "Write a generic data transformation pipeline in Go (1.18+) with full generic type safety.\n"
            "Requirements:\n"
            "1. Define `type Pipeline[T any] struct { data []T }`.\n"
            "2. Constructor `NewPipeline[T any](data []T) *Pipeline[T]`.\n"
            "3. Method `Filter(predicate func(T) bool) *Pipeline[T]`.\n"
            "4. Generic function `MapPipeline[T any, U any](p *Pipeline[T], fn func(T) U) *Pipeline[U]`.\n"
            "5. Method `Execute() []T`.\n"
            "Output ONLY the Go code in a ```go block."
        ),
        "required_patterns": [r"Pipeline\[T any\]", r"func", r"Execute\(\)"],
    },
    "java": {
        "prompt": (
            "Write a generic, immutable data transformation pipeline class in Java.\n"
            "Requirements:\n"
            "1. `public class Pipeline<T>` with an immutable `List<T> data`.\n"
            "2. `public <U> Pipeline<U> map(Function<? super T, ? extends U> mapper)`.\n"
            "3. `public Pipeline<T> filter(Predicate<? super T> predicate)`.\n"
            "4. `public List<T> execute()`.\n"
            "All generic bounds and wildcards must be strictly typed without raw types.\n"
            "Output ONLY the Java code in a ```java block."
        ),
        "required_patterns": [r"class Pipeline<T>", r"map", r"filter", r"execute\(\)"],
    },
    "rust": {
        "prompt": (
            "Write a generic, immutable data transformation pipeline in Rust with strict generic bounds.\n"
            "Requirements:\n"
            "1. `pub struct Pipeline<T> { data: Vec<T> }`.\n"
            "2. `impl<T: Clone> Pipeline<T>` with `pub fn new(data: Vec<T>) -> Self`.\n"
            "3. `pub fn map<U, F: Fn(&T) -> U>(&self, f: F) -> Pipeline<U>`.\n"
            "4. `pub fn filter<F: Fn(&T) -> bool>(&self, f: F) -> Pipeline<T>`.\n"
            "5. `pub fn execute(self) -> Vec<T>`.\n"
            "Output ONLY the Rust code in a ```rust block."
        ),
        "required_patterns": [r"struct Pipeline<T>", r"impl<T", r"fn map<", r"fn execute"],
    },
    "csharp": {
        "prompt": (
            "Write a generic, immutable data transformation pipeline class in C# (.NET) with 100% strict typing.\n"
            "Requirements:\n"
            "1. `public class Pipeline<T>` with `IReadOnlyList<T>`.\n"
            "2. `public Pipeline<U> Map<U>(Func<T, U> mapper)`.\n"
            "3. `public Pipeline<T> Filter(Func<T, bool> predicate)`.\n"
            "4. `public IReadOnlyList<T> Execute()`.\n"
            "All parameters, return types, and generic parameters must be explicitly typed.\n"
            "Output ONLY the C# code in a ```csharp block."
        ),
        "required_patterns": [r"class Pipeline<T>", r"Map<U>", r"Filter\(", r"Execute\(\)"],
    },
}


# ============================================================================
# 6. TEST 25: SAFE REFACTORING INVARIANT FIXTURES
# ============================================================================

REFACTOR_FIXTURES: Dict[str, Dict[str, Any]] = {
    "python": {
        "legacy": (
            "def process_order_data(records):\n"
            "    res = []\n"
            "    if records is not None:\n"
            "        for r in records:\n"
            "            if r and 'qty' in r and 'price' in r:\n"
            "                q = r['qty']\n"
            "                p = r['price']\n"
            "                if q > 0 and p >= 0:\n"
            "                    total = q * p\n"
            "                    if 'discount' in r:\n"
            "                        d = r['discount']\n"
            "                        if d > 0 and d <= 1:\n"
            "                            total = total * (1 - d)\n"
            "                    res.append({'id': r.get('id', 'N/A'), 'total': round(total, 2)})\n"
            "    return res\n"
        ),
        "prompt": (
            "Refactor the following deeply nested legacy Python function into clean, modern, idiomatic Python with early returns and guard clauses.\n\n"
            "```python\n"
            "def process_order_data(records):\n"
            "    res = []\n"
            "    if records is not None:\n"
            "        for r in records:\n"
            "            if r and 'qty' in r and 'price' in r:\n"
            "                q = r['qty']\n"
            "                p = r['price']\n"
            "                if q > 0 and p >= 0:\n"
            "                    total = q * p\n"
            "                    if 'discount' in r:\n"
            "                        d = r['discount']\n"
            "                        if d > 0 and d <= 1:\n"
            "                            total = total * (1 - d)\n"
            "                    res.append({'id': r.get('id', 'N/A'), 'total': round(total, 2)})\n"
            "    return res\n"
            "```\n\n"
            "Requirements:\n"
            "1. Maintain EXACT input/output behavior for all cases (None input, missing keys, invalid qty/price, discounts).\n"
            "2. Function name must remain `process_order_data(records)`.\n"
            "Output ONLY the refactored python code in a ```python block."
        ),
        "function_name": "process_order_data",
    },
    "typescript": {
        "legacy": (
            "export interface OrderRecord { id?: string; qty?: number; price?: number; discount?: number; }\n"
            "export interface ProcessedOrder { id: string; total: number; }\n"
            "export function processOrderData(records: OrderRecord[] | null | undefined): ProcessedOrder[] {\n"
            "    const res: ProcessedOrder[] = [];\n"
            "    if (records) {\n"
            "        for (const r of records) {\n"
            "            if (r && r.qty !== undefined && r.price !== undefined) {\n"
            "                if (r.qty > 0 && r.price >= 0) {\n"
            "                    let total = r.qty * r.price;\n"
            "                    if (r.discount !== undefined && r.discount > 0 && r.discount <= 1) {\n"
            "                        total *= (1 - r.discount);\n"
            "                    }\n"
            "                    res.push({ id: r.id || 'N/A', total: Math.round(total * 100) / 100 });\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return res;\n"
            "}\n"
        ),
        "prompt": (
            "Refactor the following nested legacy TypeScript function into clean, modern TypeScript with early returns and guard clauses.\n"
            "Maintain EXACT input/output behavior for null/empty/discount cases.\n"
            "Output ONLY the TypeScript code in a ```typescript block."
        ),
        "function_name": "processOrderData",
    },
    "go": {
        "legacy": (
            "package orders\n"
            "type Record struct { ID string; Qty int; Price float64; Discount float64 }\n"
            "type ProcessedOrder struct { ID string; Total float64 }\n"
            "func ProcessOrderData(records []Record) []ProcessedOrder {\n"
            "    var res []ProcessedOrder\n"
            "    for _, r := range records {\n"
            "        if r.Qty > 0 && r.Price >= 0 {\n"
            "            total := float64(r.Qty) * r.Price\n"
            "            if r.Discount > 0 && r.Discount <= 1 {\n"
            "                total *= (1 - r.Discount)\n"
            "            }\n"
            "            res = append(res, ProcessedOrder{ID: r.ID, Total: total})\n"
            "        }\n"
            "    }\n"
            "    return res\n"
            "}\n"
        ),
        "prompt": (
            "Refactor the following Go function using idiomatic Go style, early returns, and guard clauses.\n"
            "Maintain exact input/output behavior.\n"
            "Output ONLY the Go code in a ```go block."
        ),
        "function_name": "ProcessOrderData",
    },
    "java": {
        "legacy": (
            "public record Record(String id, Integer qty, Double price, Double discount) {}\n"
            "public record ProcessedOrder(String id, double total) {}\n"
            "public static List<ProcessedOrder> processOrderData(List<Record> records) {\n"
            "    List<ProcessedOrder> res = new ArrayList<>();\n"
            "    if (records != null) {\n"
            "        for (Record r : records) {\n"
            "            if (r != null && r.qty() != null && r.price() != null) {\n"
            "                if (r.qty() > 0 && r.price() >= 0) {\n"
            "                    double total = r.qty() * r.price();\n"
            "                    if (r.discount() != null && r.discount() > 0 && r.discount() <= 1) {\n"
            "                        total *= (1 - r.discount());\n"
            "                    }\n"
            "                    res.add(new ProcessedOrder(r.id() != null ? r.id() : \"N/A\", total));\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return res;\n"
            "}\n"
        ),
        "prompt": (
            "Refactor the following Java method into clean, modern Java using early returns and guard clauses.\n"
            "Maintain exact input/output behavior.\n"
            "Output ONLY the Java code in a ```java block."
        ),
        "function_name": "processOrderData",
    },
    "rust": {
        "legacy": (
            "pub struct Record { pub id: Option<String>, pub qty: i32, pub price: f64, pub discount: Option<f64> }\n"
            "pub struct ProcessedOrder { pub id: String, pub total: f64 }\n"
            "pub fn process_order_data(records: &[Record]) -> Vec<ProcessedOrder> { ... }\n"
        ),
        "prompt": (
            "Refactor the following legacy Rust function using idiomatic Rust iterators and filter_map.\n"
            "Maintain exact input/output behavior.\n"
            "Output ONLY the Rust code in a ```rust block."
        ),
        "function_name": "process_order_data",
    },
    "csharp": {
        "legacy": (
            "public record Record(string? Id, int Qty, decimal Price, decimal? Discount);\n"
            "public record ProcessedOrder(string Id, decimal Total);\n"
            "public static List<ProcessedOrder> ProcessOrderData(IEnumerable<Record>? records) { ... }\n"
        ),
        "prompt": (
            "Refactor the following C# method using idiomatic C# guard clauses and early returns.\n"
            "Maintain exact input/output behavior.\n"
            "Output ONLY the C# code in a ```csharp block."
        ),
        "function_name": "ProcessOrderData",
    },
}


# ============================================================================
# POLYGLOT PROMPTS (generated from the execution test packs)
# ============================================================================
# For TypeScript, Go, Java, Rust and C#, Tests 21, 23, 24 and 25 are graded by
# compiling and running the answer against data/polyglot/<test>/<lang>/. The
# prompts are built from those same pack files so the model always sees the
# exact code the harness compiles against.

_POLYGLOT_DIR = os.path.join(DATA_DIR, "polyglot")
_FENCE = {"typescript": "typescript", "go": "go", "java": "java", "rust": "rust", "csharp": "csharp"}


def _pack_file(test_key: str, lang: str, part: str, name: str) -> str:
    with open(os.path.join(_POLYGLOT_DIR, test_key, lang, part, name), encoding="utf-8") as f:
        return f.read().strip()


def _files_block(test_key: str, lang: str, names) -> str:
    fence = _FENCE[lang]
    return "\n\n".join(f"File `{n}`:\n```{fence}\n{_pack_file(test_key, lang, 'support', n)}\n```" for n in names)


# Test 20: the null-safety requirement is stated precisely so a behavior check is fair.
DIFF_FIXTURES["typescript"]["prompt"] = DIFF_FIXTURES["typescript"]["prompt"].replace(
    "Fix the off-by-one loop boundary and add null safety.",
    "Fix the off-by-one loop boundary and add null safety: a null/undefined `employees` argument must be treated as an "
    "empty list (return only the bonus pool), and null entries in the list must be skipped.")
DIFF_FIXTURES["go"]["prompt"] = DIFF_FIXTURES["go"]["prompt"].replace(
    "Fix the out-of-bounds loop and handle nil pointers safely.",
    "Fix the out-of-bounds loop and handle nil pointers safely: nil `*Employee` entries must be skipped, and a nil "
    "slice must return only the bonus pool.")
DIFF_FIXTURES["java"]["prompt"] = DIFF_FIXTURES["java"]["prompt"].replace(
    "Fix the off-by-one loop and add null safety.",
    "Fix the off-by-one loop and add null safety: a null `employees` list must be treated as empty (return only the "
    "bonus pool), and null entries in the list must be skipped.")
DIFF_FIXTURES["csharp"]["prompt"] = DIFF_FIXTURES["csharp"]["prompt"].replace(
    "Fix the index boundary and null handling.",
    "Fix the index boundary and null handling: a null `employees` list must be treated as empty (return only the "
    "bonus pool), and null entries in the list must be skipped.")

# Test 21: multi-file dependency injection.
_MULTIFILE_SPECS = {
    "typescript": (["models.ts", "repository.ts"], "authService.ts",
        "Implement and export `class AuthenticationService` with:\n"
        "1. Constructor dependency injection: `constructor(userRepo: IUserRepository, config: TenantConfig)`.\n"
        "2. `async authenticate(userId: string): Promise<AuthToken | null>`: resolve the user via `userRepo.getById(userId)`; "
        "if the user exists and `isActive`, return `{ token: `tok_${user.id}`, expiresIn: config.maxTokenTtlSeconds }`; otherwise return null.\n"
        "Import the types from './models' and './repository'."),
    "go": (["models.go", "repository.go"], "auth_service.go",
        "In package `authsvc` (same package as the files above), implement:\n"
        "1. `type AuthService struct` and the constructor `NewAuthService(repo UserRepository, config TenantConfig) *AuthService`.\n"
        "2. `func (s *AuthService) Authenticate(userID string) (*AuthToken, error)`: resolve the user via `repo.GetByID(userID)`; "
        "if that returns an error, return nil and the error; if the user exists and `IsActive`, return "
        "`&AuthToken{Token: \"tok_\" + user.ID, ExpiresIn: config.MaxTokenTTLSeconds}, nil`; otherwise return nil, nil."),
    "java": (["User.java", "TenantConfig.java", "AuthToken.java", "UserRepository.java"], "AuthenticationService.java",
        "In the same (default) package, implement `public class AuthenticationService` with:\n"
        "1. Constructor dependency injection: `public AuthenticationService(UserRepository userRepo, TenantConfig config)`.\n"
        "2. `public Optional<AuthToken> authenticate(String userId)`: resolve the user via `userRepo.getById(userId)`; if present "
        "and `isActive()`, return an AuthToken with token `\"tok_\" + user.id()` and `config.maxTokenTtlSeconds()`; otherwise `Optional.empty()`."),
    "rust": (["models.rs", "repository.rs"], "auth_service.rs",
        "The crate declares `mod models; mod repository; mod auth_service;`. Implement `AuthenticationService<R: UserRepository>` with:\n"
        "1. `pub fn new(user_repo: R, config: TenantConfig) -> Self` (constructor dependency injection).\n"
        "2. `pub fn authenticate(&self, user_id: &str) -> Option<AuthToken>`: resolve the user via `self.user_repo.get_by_id(user_id)`; "
        "if it exists and `is_active`, return `Some(AuthToken { token: format!(\"tok_{}\", user.id), expires_in: self.config.max_token_ttl_seconds })`; "
        "otherwise `None`. Use `crate::models` and `crate::repository` for the types."),
    "csharp": (["Models.cs", "IUserRepository.cs"], "AuthenticationService.cs",
        "Implement `public class AuthenticationService` (global namespace, like the files above) with:\n"
        "1. Constructor dependency injection: `public AuthenticationService(IUserRepository userRepo, TenantConfig config)`.\n"
        "2. `public async Task<AuthToken?> AuthenticateAsync(string userId)`: resolve the user via `GetByIdAsync(userId)`; "
        "if the user exists and `IsActive`, return `new AuthToken($\"tok_{user.Id}\", config.MaxTokenTtlSeconds)`; otherwise null."),
}
for _lang, (_support, _answer, _spec) in _MULTIFILE_SPECS.items():
    MULTIFILE_FIXTURES[_lang]["prompt"] = (
        "You are working across multiple files in a repository. These files already exist:\n\n"
        + _files_block("multifile_context", _lang, _support)
        + f"\n\nWrite the missing file `{_answer}`.\n{_spec}\n"
        f"Output only the complete `{_answer}` in a ```{_FENCE[_lang]} code block."
    )

# Test 23: unit-test synthesis (graded by running the tests and by mutation testing).
_UNIT_TEST_SPECS = {
    "typescript": ("paymentProcessor.ts", "payment.test.ts",
        "Generate a complete Vitest test file `payment.test.ts` (import `describe`, `it`, `expect`, `vi` from 'vitest' and "
        "`PaymentProcessor` from './paymentProcessor'). Use `vi.fn()` mocks for the gateway and notifier."),
    "go": ("payment.go", "payment_test.go",
        "Generate a complete Go test file `payment_test.go` in package `payment` using the standard `testing` package. "
        "Implement your own mock structs for `StripeGateway` and `NotificationClient`."),
    "java": ("PaymentProcessor.java", "PaymentProcessorTest.java",
        "Generate a complete JUnit 5 test class `PaymentProcessorTest` in package `com.payment`, using Mockito "
        "(`mock(...)`, `when(...).thenReturn(...)`, `verify(...)`)."),
    "rust": ("payment.rs", "tests",
        "Generate a `#[cfg(test)] mod tests { use super::*; ... }` block that will be appended to the end of this file. "
        "Implement your own mock structs for `StripeGateway` and `NotificationClient` (use `Cell`/`RefCell` to record calls)."),
    "csharp": ("PaymentProcessor.cs", "PaymentProcessorTests.cs",
        "Generate a complete xUnit test class `PaymentProcessorTests` using Moq (`new Mock<IStripeGateway>()`, "
        "`.Setup(...)`, `.ReturnsAsync(...)`, `.Verify(...)`). Add `using Payments;`."),
}
_UNIT_TEST_REQUIREMENTS = (
    "Requirements (each must be a separate test):\n"
    "1. Successful charge (status \"succeeded\"): returns true AND the receipt is sent exactly once with the customer's "
    "email and the charge id returned by the gateway.\n"
    "2. Declined charge (status \"declined\"): returns false AND no receipt is sent.\n"
    "3. Non-positive amount: fails with the documented error AND the gateway is never charged.\n"
    "Your tests are run against this code and against subtly broken versions of it; they must pass on the correct code "
    "and catch the broken ones."
)
for _lang, (_src, _answer, _framework) in _UNIT_TEST_SPECS.items():
    UNIT_TEST_FIXTURES[_lang]["source"] = _pack_file("unit_test_synthesis", _lang, "support", _src)
    UNIT_TEST_FIXTURES[_lang]["prompt"] = (
        f"Given the production code in `{_src}`:\n\n```{_FENCE[_lang]}\n{UNIT_TEST_FIXTURES[_lang]['source']}\n```\n\n"
        f"{_framework}\n{_UNIT_TEST_REQUIREMENTS}\n"
        f"Output ONLY the test code in a ```{_FENCE[_lang]} block."
    )
UNIT_TEST_FIXTURES["python"]["prompt"] = UNIT_TEST_FIXTURES["python"]["prompt"].replace(
    "2. Test successful charge (`status == 'succeeded'`) verifies `send_receipt` was called and returns True.",
    "2. Test successful charge (`status == 'succeeded'`) verifies `send_receipt` was called exactly once with the "
    "customer email and the charge id, and returns True.").replace(
    "4. Test negative amount raises `ValueError`.",
    "4. Test negative amount raises `ValueError` and the gateway is never charged.\n"
    "The code above is the module `payment_processor` (`from payment_processor import PaymentProcessor`).\n"
    "Your tests are run against this code and against subtly broken versions of it; they must pass on the correct "
    "code and catch the broken ones.")
UNIT_TEST_FIXTURES["python"]["mutants"] = [
    {"name": "any_status_treated_as_success", "replace": ["if resp.get('status') == 'succeeded':", "if resp.get('status'):"]},
    {"name": "amount_guard_removed", "replace": ["if amount <= 0:", "if amount < -10**9:"]},
    {"name": "receipt_sent_with_wrong_charge_id", "replace": ["self.notifier.send_receipt(email, resp['id'])", "self.notifier.send_receipt(email, token)"]},
]

# Test 24: strict typing (compiled in each language's strictest practical mode).
TYPING_FIXTURES["typescript"]["prompt"] = TYPING_FIXTURES["typescript"]["prompt"].replace(
    "5. Implement `execute(): T[]`.",
    "5. Implement `execute(): T[]` returning a new array (callers must not be able to mutate the pipeline).\n"
    "Export the class from `pipeline.ts` (`export class Pipeline<T>`). It is compiled with `tsc --strict`.")
TYPING_FIXTURES["go"]["prompt"] = TYPING_FIXTURES["go"]["prompt"].replace(
    "1. Define `type Pipeline[T any] struct { data []T }`.",
    "Use package `pipeline`. The pipeline is immutable: the constructor copies its input, Filter returns a new pipeline, "
    "and Execute returns a copy. Do not use `interface{}`/`any` values or reflection outside type-parameter constraints.\n"
    "1. Define `type Pipeline[T any] struct { data []T }`.").replace(
    "Output ONLY the Go code", "The code must pass `go vet`. Output ONLY the Go code")
TYPING_FIXTURES["java"]["prompt"] = TYPING_FIXTURES["java"]["prompt"].replace(
    "1. `public class Pipeline<T>` with an immutable `List<T> data`.",
    "1. `public class Pipeline<T>` with an immutable `List<T> data` and constructor `public Pipeline(List<T> data)` that "
    "copies its input; `execute()` must not let callers mutate the pipeline.").replace(
    "Output ONLY the Java code", "It is compiled with `javac -Xlint:all -Werror`. Output ONLY the Java code")
TYPING_FIXTURES["rust"]["prompt"] = TYPING_FIXTURES["rust"]["prompt"].replace(
    "Output ONLY the Rust code", "It is compiled as module `pipeline` with `rustc -D warnings`; no `unsafe`. Output ONLY the Rust code")
TYPING_FIXTURES["csharp"]["prompt"] = TYPING_FIXTURES["csharp"]["prompt"].replace(
    "1. `public class Pipeline<T>` with `IReadOnlyList<T>`.",
    "1. `public class Pipeline<T>` with `IReadOnlyList<T>` and constructor `public Pipeline(IEnumerable<T> data)` that "
    "copies its input; `Execute()` must not let callers mutate the pipeline.").replace(
    "Output ONLY the C# code", "It is compiled with nullable reference types enabled and warnings as errors; no `dynamic`. "
    "Output ONLY the C# code")

# Test 25: safe refactoring (graded by differential testing against the legacy code).
_REFACTOR_SPECS = {
    "typescript": (_pack_file("refactor_invariants", "typescript", "support", "legacy.ts"),
        "Keep the exported names `OrderRecord`, `ProcessedOrder` and `processOrderData` and the same signature."),
    "go": ("package orders\n\n" + _pack_file("refactor_invariants", "go", "reference", "orders.go").split("func ProcessOrderData")[0]
           .replace("package orders\n\n", "")
           + _pack_file("refactor_invariants", "go", "support", "legacy.go").split("\n", 3)[3].replace("legacyProcessOrderData", "ProcessOrderData"),
        "Keep package `orders`, the `Record` and `ProcessedOrder` types and the `ProcessOrderData` signature."),
    "java": (_pack_file("refactor_invariants", "java", "support", "OrderRecord.java") + "\n"
             + _pack_file("refactor_invariants", "java", "support", "ProcessedOrder.java") + "\n\n"
             + _pack_file("refactor_invariants", "java", "support", "LegacyOrderProcessor.java")
               .replace("/** The original implementation, kept for differential testing. */\n", "")
               .replace("public final class LegacyOrderProcessor", "public final class OrderProcessor"),
        "The records `OrderRecord` and `ProcessedOrder` already exist in their own files; output only `OrderProcessor.java` "
        "with the same `public static List<ProcessedOrder> processOrderData(List<OrderRecord> records)` method."),
    "rust": (_pack_file("refactor_invariants", "rust", "support", "legacy.rs"),
        "Keep the `Record` and `ProcessedOrder` structs (same public fields) and the `process_order_data` signature."),
    "csharp": (_pack_file("refactor_invariants", "csharp", "support", "Legacy.cs").replace("namespace Legacy;", "namespace Orders;"),
        "Keep `namespace Orders;`, the `Record` and `ProcessedOrder` records and the "
        "`OrderProcessor.ProcessOrderData` signature."),
}
for _lang, (_legacy, _keep) in _REFACTOR_SPECS.items():
    REFACTOR_FIXTURES[_lang]["legacy"] = _legacy
    REFACTOR_FIXTURES[_lang]["prompt"] = (
        f"Refactor the following deeply nested legacy code into clean, idiomatic, modern code with guard clauses and early "
        f"returns:\n\n```{_FENCE[_lang]}\n{_legacy}\n```\n\n"
        f"Requirements:\n1. Maintain EXACT input/output behavior for every input (null/empty inputs, missing or invalid "
        f"fields, discount boundaries, rounding).\n2. {_keep}\n"
        f"Output ONLY the complete refactored code in a ```{_FENCE[_lang]} block."
    )


# ============================================================================
# CODE EXTRACTION & EVALUATION VERIFICATION HELPERS
# ============================================================================

LANGUAGE_FENCE_MAP = {
    "python": r"python|python3|py",
    "typescript": r"typescript|ts|javascript|js",
    "go": r"go|golang",
    "java": r"java",
    "rust": r"rust|rs",
    "csharp": r"csharp|cs|c#",
}


def extract_code_by_language(raw_content: str, lang: str = "python", prefer_class: str = None) -> str:
    """Extracts code block matching the specified language syntax fence."""
    lang_norm = normalize_language(lang)
    tag_pattern = LANGUAGE_FENCE_MAP.get(lang_norm, r"[a-zA-Z0-9_\-]+")
    fence_pattern = rf"```(?:{tag_pattern})?\s*\n(.*?)```"
    blocks = re.findall(fence_pattern, raw_content, re.DOTALL | re.IGNORECASE)

    # Check for unclosed block at end of response (e.g. truncated generation)
    unclosed_match = re.search(rf"```(?:{tag_pattern})?\s*\n((?:(?!```).)*)$", raw_content, re.DOTALL | re.IGNORECASE)
    unclosed_code = unclosed_match.group(1).strip() if unclosed_match else ""
    if unclosed_code:
        unclosed_code = re.sub(r"```\s*$", "", unclosed_code).strip()

    all_blocks = list(blocks)
    if unclosed_code and unclosed_code not in all_blocks:
        all_blocks.append(unclosed_code)

    if prefer_class:
        pref_blocks = [b.strip() for b in all_blocks if prefer_class in b]
        if pref_blocks:
            return max(pref_blocks, key=len)
        if prefer_class in unclosed_code:
            return unclosed_code
        if prefer_class not in raw_content:
            return ""

    if blocks:
        return max(blocks, key=len).strip()
    if unclosed_code:
        return unclosed_code

    # Generic fence fallback if language-tagged fence wasn't found
    generic_blocks = re.findall(r"```\s*\n(.*?)```", raw_content, re.DOTALL)
    if generic_blocks:
        return max(generic_blocks, key=len).strip()

    # Raw content fallback
    for marker in ("def ", "class ", "fn ", "pub ", "func ", "package ", "import ", "interface "):
        if marker in raw_content:
            code = raw_content.strip()
            code = re.sub(r"```\s*$", "", code).strip()
            return code

    return ""


def extract_unified_diff(text: str, file_name: str = "") -> str:
    """Extracts clean unified git diff text, stripping conversational text and markdown fences."""
    if not text:
        return ""

    fence_pattern = r"```(?:diff|patch)?\s*\n([\s\S]*?)\n```"
    matches = re.findall(fence_pattern, text)
    for m in matches:
        if ("--- " in m and "+++ " in m) or "@@" in m:
            text = m
            break

    target_hdr = f"--- a/{file_name}" if file_name else "--- a/"
    start_idx = text.find(target_hdr)
    if start_idx == -1:
        m = re.search(r"^--- (?:a/|[^\s]+)", text, re.MULTILINE)
        start_idx = m.start() if m else -1

    if start_idx == -1:
        return text.strip()

    diff_body = text[start_idx:]
    lines = diff_body.splitlines()
    cleaned_lines = []
    in_hunk = False
    for line in lines:
        if line.startswith(("--- ", "+++ ")):
            cleaned_lines.append(line)
        elif line.startswith("@@ "):
            in_hunk = True
            cleaned_lines.append(line)
        elif in_hunk:
            if line.startswith(("+", "-", " ", "\\")):
                cleaned_lines.append(line)
            elif not line.strip():
                cleaned_lines.append(" " + line)
            else:
                break
        else:
            cleaned_lines.append(line)

    # Not .strip(): a hunk may end with a blank context line (" "), and
    # stripping it made a correct patch "corrupt" (one line short).
    return "\n".join(fit_hunk_lengths(cleaned_lines)).strip("\n")


# Longest error text sent back to the model on a retry (Tests 14, 17, 20-21, 23-25).
MAX_FEEDBACK_CHARS = 2000
NO_CODE_FEEDBACK = "No code block was found in your answer."


def timeout_feedback(seconds: float) -> str:
    """What the model is told when its code ran past the time limit."""
    return f"Your code did not finish within {seconds:g} s (an infinite loop, deadlock or blocking call?)."


def tool_feedback(res: dict) -> str:
    """The error a developer would read from a failed compile-and-run grade."""
    return f"{res['stage']} failed:\n" + str(res.get("output") or res.get("detail") or "")[:MAX_FEEDBACK_CHARS]


def output_tail(text: str, lines: int = 20) -> str:
    """The last lines of a Python run's output (the traceback ends there)."""
    return "\n".join(text.strip().splitlines()[-lines:])[:MAX_FEEDBACK_CHARS]

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
MAX_BLANK_PAD = 3


def fit_hunk_lengths(lines: list) -> list:
    """Repairs only the blank lines at the end of each hunk, so the hunk has
    the line counts its header declares: extra trailing blank context lines
    (from the blank line before a closing code fence) are dropped, and up to
    MAX_BLANK_PAD missing ones (lost when the fence was cut off) are added
    back. Nothing else changes, and git apply still checks every context line
    against the file, so this cannot turn a wrong patch into one that applies."""
    out: list = []
    i = 0
    while i < len(lines):
        m = _HUNK_HEADER.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        want_old = int(m.group(1)) if m.group(1) is not None else 1
        want_new = int(m.group(2)) if m.group(2) is not None else 1
        out.append(lines[i])
        i += 1
        body = []
        while i < len(lines) and not _HUNK_HEADER.match(lines[i]) and not lines[i].startswith(("--- ", "+++ ")):
            body.append(lines[i])
            i += 1

        def counts(b):
            return (sum(1 for x in b if x[:1] in (" ", "-")), sum(1 for x in b if x[:1] in (" ", "+")))
        old, new = counts(body)
        while body and body[-1].strip() == "" and old > want_old and new > want_new:
            body.pop()
            old, new = counts(body)
        pad = want_old - old
        if 0 < pad <= MAX_BLANK_PAD and want_new - new == pad:
            body += [" "] * pad
        out += body
    return out


def _polyglot_grade(test_key: str, lang: str, code: str, execute_code: bool):
    """Compile-and-run grading for a non-Python answer, or None when it cannot
    run (execution disabled, no test pack, toolchain not installed, or a build
    step timed out). Callers treat None as unverified and never award a PASS
    for it. A build timeout is also recorded for eval's infra guard
    (polyglot.take_infra_events), which marks the test an infra failure."""
    if not execute_code:
        return None
    import polyglot
    if not polyglot.has_pack(test_key, lang) or not polyglot.toolchain_available(lang):
        return None
    res = polyglot.grade_answer(test_key, lang, code)
    return None if res["stage"] == "infra" else res


_UNVERIFIED_NOTE = "toolchain not available (run inside the Docker image); structural check only, capped at PARTIAL"
_EXECUTION_DISABLED_NOTE = "execution disabled (--no-execute / EVAL_SAFE=1); structural check only, capped at PARTIAL"


def verify_git_diff(diff_text: str, original_code: str, file_name: str,
                    behavior_check: Optional[str] = None, execute_code: bool = True, language: str = "python") -> dict:
    """Verifies a unified diff: it must apply cleanly with `git apply`, and when
    the fixture has a behavior check (Python), the patched file must actually
    fix the bug. A patch that applies but changes nothing useful is PARTIAL.

    "feedback" is the error to show the model when the failure is the
    model's (no diff, git apply rejected it, or the patched code failed to
    compile or failed its tests), so it can try again; None otherwise."""
    valid_syntax = bool(re.search(r"^--- .+?\n\+\+\+ .+?\n@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", diff_text.strip(), re.MULTILINE))
    git_apply_passed = False
    behavior_verified = None  # None = not checked (no check for this language / execution disabled)
    detail = ""
    feedback = None

    if not valid_syntax:
        detail = "no well-formed unified diff hunk"
        feedback = "No unified diff was found in your answer (expected ---/+++ headers and an @@ hunk)."
    elif not shutil.which("git"):
        detail = "git not installed; patch application unverified"
    else:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file_name)
                with open(file_path, "w") as f:
                    f.write(original_code)
                with open(os.path.join(tmpdir, "fix.patch"), "w") as f:
                    f.write(diff_text.strip("\n") + "\n")  # keep a final blank context line (" ")
                git = ["git", "-c", "user.email=eval@bench.local", "-c", "user.name=Eval Bench"]
                subprocess.run(git + ["init"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                subprocess.run(git + ["add", file_name], cwd=tmpdir, check=True)
                subprocess.run(git + ["commit", "-m", "init"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                proc = subprocess.run(["git", "apply", "fix.patch"], cwd=tmpdir, capture_output=True, text=True, timeout=10)
                git_apply_passed = (proc.returncode == 0)
                if not git_apply_passed:
                    detail = (proc.stderr.strip().splitlines() or ["git apply failed"])[0][:200]
                    feedback = "`git apply` rejected your patch:\n" + (proc.stderr.strip() or "git apply failed")[:MAX_FEEDBACK_CHARS]
                elif normalize_language(language) != "python":
                    with open(file_path) as f:
                        patched = f.read()
                    res = _polyglot_grade("surgical_git_diff", normalize_language(language), patched, execute_code)
                    if res is None:
                        detail = _UNVERIFIED_NOTE if execute_code else _EXECUTION_DISABLED_NOTE
                    else:
                        behavior_verified = res["ok"]
                        if not behavior_verified:
                            detail = f"{res['stage']}: {res['detail']}"[:200]
                            feedback = "The patch applied, but the patched code failed. " + tool_feedback(res)
                elif behavior_check and not execute_code:
                    detail = _EXECUTION_DISABLED_NOTE
                elif behavior_check:
                    with open(file_path) as f:
                        patched = f.read()
                    try:
                        run = run_model_code([sys.executable, "-c", patched + "\n\n" + behavior_check], 15, cwd=tmpdir)
                    except subprocess.TimeoutExpired:
                        run = subprocess.CompletedProcess([], 1, "", timeout_feedback(15))
                    behavior_verified = run.returncode == 0 and harness_passed(run, "DIFF_BEHAVIOR_VERIFIED")
                    if not behavior_verified:
                        lines = (run.stderr or run.stdout).strip().splitlines()
                        detail = (lines[-1] if lines else "behavior check failed")[:200]
                        feedback = ("The patch applied, but the patched code failed its tests:\n"
                                    + output_tail(run.stderr or run.stdout))
        except Exception as e:
            git_apply_passed = False
            detail = f"verification error: {e}"[:200]

    # A behavior check that could not run (no toolchain, or execution disabled)
    # leaves the patch unverified: PARTIAL at most.
    unverified = behavior_verified is None and (normalize_language(language) != "python" or bool(behavior_check))
    if git_apply_passed and behavior_verified is not False and not unverified:
        status = "PASS"
    elif valid_syntax:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "valid_diff_syntax": valid_syntax,
        "git_apply_passed": git_apply_passed,
        "behavior_verified": behavior_verified,
        "detail": detail,
        "feedback": feedback,
        "status": status,
    }


def _prompt_support_files(prompt: str) -> dict:
    """Splits the '# File N: name.py' sections of a multi-file prompt into
    {file_name: source}, skipping the file the model is asked to write."""
    files = {}
    parts = re.split(r"^# File \d+: (\S+)(.*)$", prompt, flags=re.MULTILINE)
    for k in range(1, len(parts) - 2, 3):
        name, suffix, body = parts[k], parts[k + 1], parts[k + 2]
        if "INCOMPLETE" not in suffix:
            files[name] = body.strip() + "\n"
    return files


def verify_multifile_context(code: str, language: str, execute_code: bool = True) -> dict:
    """Verifies multi-file context cross-referencing and dependency injection.
    For Python with execution enabled, the generated module is imported next
    to the prompt's real sibling files and exercised against a fake repository;
    otherwise the required symbols must be defined or called (AST)."""
    lang = normalize_language(language)
    fixture = MULTIFILE_FIXTURES.get(lang, MULTIFILE_FIXTURES["python"])
    req_symbols = fixture.get("required_symbols", [])

    syntax_valid = False
    symbols_resolved = 0
    behavior_verified = None
    feedback = None  # the error to show the model on a retry (see verify_git_diff)

    if lang == "python":
        try:
            tree = ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            tree = None
            feedback = f"SyntaxError: {e}" if code.strip() else NO_CODE_FEEDBACK
        if tree is not None:
            defined_or_called = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    defined_or_called.add(node.name)
                elif isinstance(node, ast.Call):
                    defined_or_called.add(getattr(node.func, "id", "") or getattr(node.func, "attr", ""))
            symbols_resolved = sum(1 for sym in req_symbols if sym in defined_or_called)
            check = fixture.get("behavior_check")
            if check and execute_code:
                behavior_verified = False
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for name, body in _prompt_support_files(fixture["prompt"]).items():
                            with open(os.path.join(tmpdir, name), "w") as f:
                                f.write(body)
                        with open(os.path.join(tmpdir, "auth_service.py"), "w") as f:
                            f.write(code)
                        run = run_model_code([sys.executable, "-c", check], 15, cwd=tmpdir)
                        behavior_verified = harness_passed(run, "MULTIFILE_BEHAVIOR_VERIFIED")
                        if not behavior_verified:
                            feedback = ("Running your module against the real sibling files failed:\n"
                                        + output_tail(run.stderr + run.stdout))
                except subprocess.TimeoutExpired:
                    behavior_verified = False
                    feedback = timeout_feedback(15)
                except Exception:
                    behavior_verified = False
    else:
        symbols_resolved = sum(1 for sym in req_symbols if re.search(rf"\b{re.escape(sym)}\b", code))
        res = _polyglot_grade("multifile_context", lang, code, execute_code)
        if res is None:
            open_braces = code.count("{")
            close_braces = code.count("}")
            syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        else:
            syntax_valid = res["stage"] not in ("compile", "unavailable")
            behavior_verified = res["ok"]
            if not res["ok"]:
                feedback = tool_feedback(res) if code.strip() else NO_CODE_FEEDBACK

    total_symbols = len(req_symbols)
    if behavior_verified is not None:
        # Execution is authoritative: the module must actually work.
        status = "PASS" if behavior_verified else ("PARTIAL" if (syntax_valid and symbols_resolved >= 1) else "FAIL")
    elif lang != "python" or fixture.get("behavior_check"):
        # Unverified (no toolchain, or execution disabled): never PASS.
        status = "PARTIAL" if (syntax_valid and symbols_resolved >= 1) else "FAIL"
    else:
        status = "PASS" if (syntax_valid and symbols_resolved >= total_symbols) else ("PARTIAL" if (syntax_valid and symbols_resolved >= 1) else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "symbols_resolved": symbols_resolved,
        "total_symbols": total_symbols,
        "behavior_verified": behavior_verified,
        "feedback": None if status == "PASS" else feedback,
        "status": status,
    }


_SNIPPET_MENTION = re.compile(r"snippet\s*\(?([abc])\b", re.IGNORECASE)
# A mention that opens a line ('### Snippet A', '**Snippet B**', '2. Snippet C:').
_SNIPPET_HEADING = re.compile(r"^[ \t>#*_\-\d.)|]*snippet\s*\(?([abc])\b", re.IGNORECASE | re.MULTILINE)
# Verdict words in priority order of appearance. Word boundaries keep 'safe'
# from matching inside 'unsafe' and 'secure' inside 'insecure'.
_VERDICT = re.compile(
    r"\b(not\s+vulnerable|no\s+vulnerabilit(?:y|ies)|not\s+safe|vulnerable|unsafe|insecure|safe|secure)\b",
    re.IGNORECASE,
)
_SAFE_VERDICTS = {"not vulnerable", "no vulnerability", "no vulnerabilities", "safe", "secure"}
_SQLI_EVIDENCE = re.compile(r"sql[\s-]*injection|\bsqli\b", re.IGNORECASE)
# The vulnerability must be named: generic words like 'lock' or 'atomic' also
# appear in remediation advice and do not show the race was identified.
_RACE_EVIDENCE = re.compile(
    r"race[\s-]*condition|data[\s-]*race|\btoctou\b|time[\s-]*of[\s-]*check|check[\s-]*then[\s-]*act",
    re.IGNORECASE,
)


def _snippet_sections(text: str) -> dict:
    """Splits a review into per-snippet sections keyed 'a'/'b'/'c'.

    Line-opening headings are preferred over passing mentions, so an intro
    such as "I'll review Snippet A, Snippet B and Snippet C." does not make
    the Snippet C section swallow the whole review. Each snippet's section
    starts at its first boundary and runs to the next boundary of another snippet.
    """
    boundaries = [(m.start(), m.group(1).lower()) for m in _SNIPPET_HEADING.finditer(text)]
    if not boundaries:
        boundaries = [(m.start(), m.group(1).lower()) for m in _SNIPPET_MENTION.finditer(text)]
    sections = {}
    for i, (pos, key) in enumerate(boundaries):
        if key in sections:
            continue
        end = next((p for p, k in boundaries[i + 1:] if k != key), len(text))
        sections[key] = text[pos:end]
    return sections


def _snippet_verdict(section: str):
    """'SAFE', 'VULNERABLE', or None from the first verdict word in a section."""
    m = _VERDICT.search(section)
    if not m:
        return None
    word = re.sub(r"\s+", " ", m.group(1).lower())
    return "SAFE" if word in _SAFE_VERDICTS else "VULNERABLE"


def verify_security_review(text: str) -> dict:
    """Grades the three-snippet OWASP review per snippet: A must be flagged
    VULNERABLE as SQL injection, B VULNERABLE as a race condition, and the
    clean snippet C must be judged SAFE (the false-positive check)."""
    sections = _snippet_sections(text)
    sec_a = sections.get("a", "")
    sec_b = sections.get("b", "")
    sec_c = sections.get("c")

    # Naming the vulnerability is a finding even without the word VULNERABLE;
    # an explicit SAFE verdict for the snippet overrides it.
    sqli_detected = _snippet_verdict(sec_a) != "SAFE" and bool(_SQLI_EVIDENCE.search(sec_a))
    race_detected = _snippet_verdict(sec_b) != "SAFE" and bool(_RACE_EVIDENCE.search(sec_b))
    # An unlabeled review cannot show snippet C was cleared.
    safe_passed = sec_c is not None and _snippet_verdict(sec_c) == "SAFE"

    status = "PASS" if (sqli_detected and race_detected and safe_passed) else ("PARTIAL" if (sqli_detected or race_detected) else "FAIL")
    return {
        "sqli_detected": sqli_detected,
        "race_detected": race_detected,
        "safe_snippet_passed": safe_passed,
        "status": status,
    }


def parse_unittest_summary(output: str):
    """Parses unittest's final summary into (tests_ran, tests_passed).

    'Ran 0 tests ... OK' (Python < 3.12) and 'NO TESTS RAN' both mean zero
    tests. Skipped and expected-failure tests are not counted as passed.
    """
    m = re.search(r"Ran (\d+) tests? in", output)
    ran = int(m.group(1)) if m else 0
    not_passed = 0
    for key in ("failures", "errors", "skipped", "expected failures", "unexpected successes"):
        km = re.search(rf"{key}=(\d+)", output)
        if km:
            not_passed += int(km.group(1))
    return ran, max(0, ran - not_passed)


_SOURCE_CLASSES = ("PaymentProcessor", "StripeGateway", "NotificationClient")


def _imported_source_modules(code: str):
    """Module names the tests import the production classes from
    (`from production_code import PaymentProcessor`, `import payment`...)."""
    names = {"payment_processor"}
    for module, imported in re.findall(r"^\s*from\s+([\w.]+)\s+import\s+\(?([^)\n]+)", code, re.MULTILINE):
        if any(cls in imported for cls in _SOURCE_CLASSES):
            names.add(module)
    return sorted(names)


def _run_python_unit_tests(code: str, source_code: str, output: Optional[list] = None):
    """Runs a generated unittest suite against `source_code`.
    Returns (tests_ran, tests_passed), or (None, 0) if it could not run.
    The run's output is appended to `output` when given.
    The production classes are served under whatever module name the tests
    import them from, so a reasonable module name is never a failure."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            registrations = "".join(f"sys.modules[{m!r}] = pp_mod\n" for m in _imported_source_modules(code))
            preamble = (
                "import sys, os, types\n"
                f"sys.path.insert(0, r'{tmpdir}')\n"
                f"{source_code}\n"
                "pp_mod = types.ModuleType('payment_processor')\n"
                "pp_mod.PaymentProcessor = PaymentProcessor\n"
                "pp_mod.StripeGateway = StripeGateway\n"
                "pp_mod.NotificationClient = NotificationClient\n"
                + registrations
            )
            harness = (
                "if __name__ == '__main__':\n"
                "    import unittest\n"
                "    unittest.main(exit=False)\n"
            )
            t_file = os.path.join(tmpdir, "test_suite.py")
            with open(t_file, "w") as f:
                f.write(assemble_python_script(code, preamble=preamble, harness=harness))
            try:
                proc = run_model_code([sys.executable, t_file], 15, cwd=tmpdir)
            except subprocess.TimeoutExpired:
                if output is not None:
                    output.append(timeout_feedback(15))
                return 0, 0  # ran, but no test finished
            if output is not None:
                output.append(proc.stderr + proc.stdout)
            return parse_unittest_summary(proc.stderr + proc.stdout)
    except Exception:
        return None, 0


def verify_unit_test(code: str, language: str, source_code: str, execute_code: bool = True) -> dict:
    """Verifies unit test synthesis by running the generated suite.

    PASS requires at least 3 tests, all passing against the real code, AND
    every seeded mutant (a subtly broken version of the code) caught by at
    least one failing test -- tests that pass but assert nothing do not pass.
    """
    lang = normalize_language(language)
    fixture = UNIT_TEST_FIXTURES.get(lang, UNIT_TEST_FIXTURES["python"])
    mock_keywords = fixture.get("mock_keywords", [])
    code_lower = code.lower()

    syntax_valid = False
    tests_passed = 0
    tests_ran = None  # set only when the suite was actually executed
    total_tests = 3  # minimum number of passing tests the prompt asks for
    mutants_total = 0
    mutants_killed = None  # None = mutation testing not performed
    execution_verified = False
    run_output: list = []
    feedback = None  # the error to show the model on a retry (see verify_git_diff)

    if lang == "python":
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            feedback = f"SyntaxError: {e}" if code.strip() else NO_CODE_FEEDBACK

        if syntax_valid and execute_code:
            tests_ran, tests_passed = _run_python_unit_tests(code, source_code, run_output)
            execution_verified = tests_ran is not None
            mutants = fixture.get("mutants", [])
            mutants_total = len(mutants)
            if tests_ran and tests_passed == tests_ran:
                mutants_killed = 0
                for mutant in mutants:
                    old, new = mutant["replace"]
                    m_ran, m_passed = _run_python_unit_tests(code, source_code.replace(old, new, 1))
                    # A mutant the suite hangs or crashes on (nothing ran) is caught.
                    mutants_killed += (m_ran is None) or (m_ran == 0) or (m_passed < m_ran)
        elif syntax_valid:
            has_mock = any(k.lower() in code_lower for k in mock_keywords)
            if has_mock and "test" in code_lower:
                tests_passed = total_tests
    else:
        res = _polyglot_grade("unit_test_synthesis", lang, code, execute_code)
        if res is None:
            open_braces = code.count("{")
            close_braces = code.count("}")
            syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
            has_mock = any(k.lower() in code_lower for k in mock_keywords)
            has_assertions = any(k in code_lower for k in ["assert", "expect", "verify", "should"])
            tests_passed = 1 if (has_mock and has_assertions) else 0
        else:
            execution_verified = True
            syntax_valid = res["stage"] != "compile"
            tests_ran = res["tests_ran"] or 0
            tests_passed = res["tests_passed"] or 0
            mutants_total = res.get("mutants_total", 0)
            mutants_killed = len(res.get("mutants_killed", [])) if res["ok"] else None
            if not res["ok"]:
                feedback = tool_feedback(res) if code.strip() else NO_CODE_FEEDBACK

    all_green = tests_ran is None or tests_passed == tests_ran
    mutation_ok = mutants_total == 0 or mutants_killed == mutants_total
    if execution_verified and feedback is None:
        # What a test run and a mutation-testing tool report; never which
        # bug was seeded.
        if lang == "python" and run_output and not (tests_ran and all_green):
            feedback = "Your test suite did not pass against the real code:\n" + output_tail(run_output[0])
        elif tests_ran is not None and tests_passed < total_tests and all_green:
            feedback = f"Only {tests_passed} test(s) ran; at least {total_tests} are required."
        elif not mutation_ok and mutants_killed is not None:
            feedback = (f"Mutation testing: your tests pass on the real code, but {mutants_total - mutants_killed} of "
                        f"{mutants_total} subtly broken versions of the code also pass them, so the tests do not "
                        "check enough of the behaviour.")
    if not execution_verified:
        status = "PARTIAL" if tests_passed > 0 else "FAIL"  # unverified: never PASS
    elif tests_passed >= total_tests and all_green and mutation_ok:
        status = "PASS"
    elif syntax_valid and tests_passed > 0:
        status = "PARTIAL"
    else:
        status = "FAIL"
    return {
        "syntax_valid": syntax_valid,
        "unit_tests_passed_count": tests_passed,
        "unit_tests_total_count": total_tests if tests_ran is None else max(total_tests, tests_ran),
        "mutants_total": mutants_total,
        "mutants_killed": mutants_killed,
        "execution_verified": execution_verified,
        "feedback": None if status == "PASS" else feedback,
        "status": status,
    }


def verify_strict_typing(code: str, language: str, execute_code: bool = True) -> dict:
    """Verifies strict typing compliance and generic declarations."""
    lang = normalize_language(language)
    fixture = TYPING_FIXTURES.get(lang, TYPING_FIXTURES["python"])
    req_patterns = fixture.get("required_patterns", [])

    syntax_valid = False
    all_annotated = False
    annotated_methods = 0
    total_methods = 0
    mypy_ran = False
    mypy_passed = False
    tool_detail = ""
    feedback = None  # the error to show the model on a retry (see verify_git_diff)

    if lang == "python":
        try:
            tree = ast.parse(code)
            syntax_valid = True
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_methods += 1
                    has_return = node.returns is not None
                    all_args = list(getattr(node.args, "posonlyargs", [])) + node.args.args + node.args.kwonlyargs
                    args_typed = all(arg.annotation is not None for arg in all_args if arg.arg not in ("self", "cls"))
                    if node.args.vararg and node.args.vararg.annotation is None:
                        args_typed = False
                    if node.args.kwarg and node.args.kwarg.annotation is None:
                        args_typed = False
                    if has_return and args_typed:
                        annotated_methods += 1

            # mypy only reads the code; it runs even with execution disabled.
            if shutil.which("mypy"):
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(code)
                    tmp_path = f.name
                try:
                    proc = subprocess.run(["mypy", "--strict", "--ignore-missing-imports", tmp_path],
                                          capture_output=True, text=True, timeout=15)
                    mypy_ran = True
                    mypy_passed = (proc.returncode == 0)
                    if not mypy_passed:
                        feedback = "mypy --strict reported:\n" + proc.stdout.replace(tmp_path, "module.py")[:MAX_FEEDBACK_CHARS]
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        except SyntaxError as e:
            feedback = f"SyntaxError: {e}" if code.strip() else NO_CODE_FEEDBACK
        except Exception:
            pass

        all_annotated = (total_methods > 0 and annotated_methods == total_methods)
        # When mypy --strict ran, its verdict is authoritative: full annotation
        # coverage does not make code with type errors compliant. Without mypy
        # (not installed) the AST annotation audit is the check.
        type_compliant = mypy_passed if mypy_ran else all_annotated
    else:
        matches = [bool(re.search(pat, code)) for pat in req_patterns]
        all_annotated = all(matches) if matches else False
        res = _polyglot_grade("strict_typing", lang, code, execute_code)
        if res is None:
            # Unverified: never PASS (see _UNVERIFIED_NOTE).
            open_braces = code.count("{")
            close_braces = code.count("}")
            syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
            type_compliant = False
        else:
            # Compiled in the language's strict mode, behavior-checked, and
            # scanned for type escape hatches (any / interface{} / dynamic / raw types).
            syntax_valid = res["stage"] != "compile"
            type_compliant = res["ok"]
            tool_detail = f"{res['stage']}: {res['detail']}"
            if not res["ok"]:
                feedback = tool_feedback(res) if code.strip() else NO_CODE_FEEDBACK
        if res is None and syntax_valid and all_annotated:
            annotated_methods = 1  # structural evidence only -> PARTIAL below

    status = "PASS" if type_compliant else ("PARTIAL" if (syntax_valid and (annotated_methods > 0 or all_annotated)) else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "all_annotated": all_annotated,
        "type_compliant": type_compliant,
        "mypy_verified": mypy_ran and mypy_passed,
        "compiler_verified": lang != "python" and type_compliant,
        "detail": tool_detail,
        "feedback": None if status == "PASS" else feedback,
        "status": status,
    }


_PY_REFACTOR_CASES = (
    "[None, [], [None], [{}], [{'qty': 0, 'price': 10}], [{'id': '1', 'qty': 2, 'price': 50.0}],"
    " [{'qty': 2, 'price': 3}], [{'id': 'm', 'price': 3}], [{'id': 'n', 'qty': 3}],"
    " [{'id': 'q', 'qty': -1, 'price': 5}], [{'id': 'p', 'qty': 2, 'price': -1}], [{'id': 'z', 'qty': 2, 'price': 0}],"
    " [{'id': '2', 'qty': 4, 'price': 25.0, 'discount': 0.1}], [{'id': 'd0', 'qty': 4, 'price': 2.5, 'discount': 0}],"
    " [{'id': 'd1', 'qty': 3, 'price': 7, 'discount': 1}], [{'id': 'd15', 'qty': 3, 'price': 7, 'discount': 1.5}],"
    " [{'id': 'dn', 'qty': 3, 'price': 7, 'discount': -0.2}], [{'id': 'r', 'qty': 1, 'price': 2.675}],"
    " [{'id': 'r2', 'qty': 7, 'price': 1.115, 'discount': 0.1}, {'id': None, 'qty': 1, 'price': 1}]]"
)


def verify_refactoring(code: str, language: str, execute_code: bool = True) -> dict:
    """Verifies safe refactoring by differential testing: the refactored code
    and the original legacy code must return identical results for every
    input in a battery covering null/empty input, missing and invalid fields,
    discount boundaries and rounding."""
    lang = normalize_language(language)

    syntax_valid = False
    invariants_passed = False
    execution_verified = False
    feedback = None  # the error to show the model on a retry (see verify_git_diff)

    if lang == "python":
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as e:
            feedback = f"SyntaxError: {e}" if code.strip() else NO_CODE_FEEDBACK

        if syntax_valid and execute_code:
            legacy = REFACTOR_FIXTURES["python"]["legacy"].replace("def process_order_data(", "def _legacy_process_order_data(")
            harness = (
                legacy + "\n"
                f"for _case in {_PY_REFACTOR_CASES}:\n"
                "    _before = _legacy_process_order_data(_case)\n"
                "    _after = process_order_data(_case)\n"
                "    assert _before == _after, f'behavior changed for {_case!r}: legacy {_before!r} vs refactored {_after!r}'\n"
                "print('INVARIANTS_OK', __import__('os').environ['EVAL_RUN_ID'])\n"
            )
            test_harness = assemble_python_script(code, harness=harness)
            try:
                # Its own work folder: when the suite runs as root, the
                # model-code user must own the script it runs.
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, "refactored.py")
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        f.write(test_harness)
                    proc = run_model_code([sys.executable, tmp_path], 10, cwd=tmpdir)
                    execution_verified = True
                    if harness_passed(proc, "INVARIANTS_OK"):
                        invariants_passed = True
                    else:
                        feedback = ("The refactored code does not behave like the original:\n"
                                    + output_tail(proc.stderr + proc.stdout, lines=6).replace(tmp_path, "refactored.py"))
            except subprocess.TimeoutExpired:
                execution_verified = True  # it ran: the model's code never finished
                feedback = timeout_feedback(10)
            except Exception:
                pass
        # Without execution the invariants are unverified: invariants_passed
        # stays False, so a parseable answer is PARTIAL at most.
    else:
        res = _polyglot_grade("refactor_invariants", lang, code, execute_code)
        if res is None:
            # Unverified: never PASS (see _UNVERIFIED_NOTE).
            open_braces = code.count("{")
            close_braces = code.count("}")
            syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        else:
            execution_verified = True
            syntax_valid = res["stage"] != "compile"
            invariants_passed = res["ok"]
            if not res["ok"]:
                feedback = tool_feedback(res) if code.strip() else NO_CODE_FEEDBACK

    status = "PASS" if invariants_passed else ("PARTIAL" if syntax_valid else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "invariants_preserved": invariants_passed,
        "execution_verified": execution_verified,
        "feedback": None if status == "PASS" else feedback,
        "status": status,
    }
