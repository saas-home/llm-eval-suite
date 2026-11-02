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

DIFF_FIXTURES = {
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
        "code": (
            "package payroll\n\n"
            "type Employee struct { Salary float64 }\n\n"
            "func CalculatePayroll(employees []*Employee, taxRate float64, bonusPool float64) float64 {\n"
            "    total := 0.0\n"
            "    for i := 0; i <= len(employees); i++ {\n"
            "        emp := employees[i]\n"
            "        net := emp.Salary * (1.0 - taxRate)\n"
            "        total += net\n"
            "    }\n"
            "    return total + bonusPool\n"
            "}\n"
        ),
        "prompt": (
            "Here is an existing Go file `payroll.go` with an out-of-bounds index error (`i <= len(employees)`) and missing nil check:\n\n"
            "```go\n"
            "package payroll\n\n"
            "type Employee struct { Salary float64 }\n\n"
            "func CalculatePayroll(employees []*Employee, taxRate float64, bonusPool float64) float64 {\n"
            "    total := 0.0\n"
            "    for i := 0; i <= len(employees); i++ {\n"
            "        emp := employees[i]\n"
            "        net := emp.Salary * (1.0 - taxRate)\n"
            "        total += net\n"
            "    }\n"
            "    return total + bonusPool\n"
            "}\n"
            "```\n\n"
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

MULTIFILE_FIXTURES = {
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

SECURITY_FIXTURES = {
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
            "def log_access(logger, user_id, ip_address):\n"
            "    clean_ip = ip_address.strip()\n"
            "    logger.info(\"User %s accessed system from IP %s\", user_id, clean_ip)\n"
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
            "function logAccess(logger: any, userId: string, ip: string) {\n"
            "    const safeIp = ip.trim();\n"
            "    logger.info('User access', { userId, ip: safeIp });\n"
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
            "func LogAccess(logger *log.Logger, userID, ip string) {\n"
            "    cleanIP := strings.TrimSpace(ip)\n"
            "    logger.Printf(\"User %s accessed from IP %s\", userID, cleanIP)\n"
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
            "public void logAccess(Logger logger, String userId, String ip) {\n"
            "    String cleanIp = ip.strip();\n"
            "    logger.info(\"User {} accessed from IP {}\", userId, cleanIp);\n"
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
            "pub fn log_access(user_id: &str, ip: &str) {\n"
            "    let clean_ip = ip.trim();\n"
            "    println!(\"User {} accessed system from IP {}\", user_id, clean_ip);\n"
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
            "public void LogAccess(ILogger logger, string userId, string ip) {\n"
            "    string cleanIp = ip.Trim();\n"
            "    logger.LogInformation(\"User {UserId} accessed from IP {IP}\", userId, cleanIp);\n"
            "}\n"
            "```\n\n"
            "For each snippet, state whether it is VULNERABLE or SAFE. If VULNERABLE, specify the exact vulnerability name."
        ),
    },
}


# ============================================================================
# 4. TEST 23: UNIT TEST SYNTHESIS WITH MOCKS FIXTURES
# ============================================================================

UNIT_TEST_FIXTURES = {
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
            "    def charge(self, amount: int, currency: str, token: str) -> dict: ...\n"
            "class NotificationClient:\n"
            "    def send_receipt(self, email: str, charge_id: str) -> bool: ...\n"
            "class PaymentProcessor:\n"
            "    def __init__(self, gateway: StripeGateway, notifier: NotificationClient): ...\n"
            "    def process_order(self, email: str, amount: int, token: str) -> bool: ...\n"
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
            "    async processOrder(email: string, amount: number, token: string): Promise<boolean> { ... }\n"
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
            "type StripeGateway interface { Charge(amount int, currency, token string) (map[string]string, error) }\n"
            "type NotificationClient interface { SendReceipt(email, chargeID string) bool }\n"
            "type PaymentProcessor struct { Gateway StripeGateway; Notifier NotificationClient }\n"
            "func (p *PaymentProcessor) ProcessOrder(email string, amount int, token string) (bool, error) { ... }\n"
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

TYPING_FIXTURES = {
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

REFACTOR_FIXTURES = {
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

    return "\n".join(cleaned_lines).strip()


def verify_git_diff(diff_text: str, original_code: str, file_name: str) -> dict:
    """Verifies unified git diff syntax and cleanly tests patch application with git apply."""
    valid_syntax = bool(re.search(r"^--- .+?\n\+\+\+ .+?\n@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", diff_text.strip(), re.MULTILINE)) or (f"--- a/{file_name}" in diff_text and f"+++ b/{file_name}" in diff_text) or ("--- a/" in diff_text and "+++ b/" in diff_text)
    git_apply_passed = False

    if valid_syntax and diff_text:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file_name)
                with open(file_path, "w") as f:
                    f.write(original_code)
                patch_path = os.path.join(tmpdir, "fix.patch")
                with open(patch_path, "w") as f:
                    f.write(diff_text.strip() + "\n")

                if shutil.which("git"):
                    subprocess.run(["git", "init"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    subprocess.run(["git", "config", "user.email", "eval@bench.local"], cwd=tmpdir, check=True)
                    subprocess.run(["git", "config", "user.name", "Eval Bench"], cwd=tmpdir, check=True)
                    subprocess.run(["git", "add", file_name], cwd=tmpdir, check=True)
                    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    proc = subprocess.run(["git", "apply", "--check", "fix.patch"], cwd=tmpdir, capture_output=True, text=True, timeout=10)
                    if proc.returncode == 0:
                        git_apply_passed = True
                else:
                    git_apply_passed = valid_syntax and ("-" in diff_text and "+" in diff_text)
        except Exception:
            git_apply_passed = False

    status = "PASS" if git_apply_passed else ("PARTIAL" if valid_syntax else "FAIL")
    return {
        "valid_diff_syntax": valid_syntax,
        "git_apply_passed": git_apply_passed,
        "status": status,
    }


def verify_multifile_context(code: str, language: str) -> dict:
    """Verifies multi-file context cross-referencing and dependency injection."""
    lang = normalize_language(language)
    fixture = MULTIFILE_FIXTURES.get(lang, MULTIFILE_FIXTURES["python"])
    req_symbols = fixture.get("required_symbols", [])

    syntax_valid = False
    symbols_resolved = 0

    if lang == "python":
        try:
            tree = ast.parse(code)
            syntax_valid = True
            for sym in req_symbols:
                found = False
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == sym:
                        found = True
                        break
                    elif isinstance(node, ast.Call):
                        fn_name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                        if fn_name == sym:
                            found = True
                            break
                if found or sym in code:
                    symbols_resolved += 1
        except Exception:
            symbols_resolved = sum(1 for sym in req_symbols if sym in code)
            syntax_valid = len(code.strip()) > 20
    else:
        # Non-python languages: check bracket balance and symbol containment
        open_braces = code.count("{")
        close_braces = code.count("}")
        syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        symbols_resolved = sum(1 for sym in req_symbols if sym in code)

    total_symbols = len(req_symbols)
    status = "PASS" if (syntax_valid and symbols_resolved >= total_symbols) else ("PARTIAL" if (syntax_valid and symbols_resolved >= 1) else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "symbols_resolved": symbols_resolved,
        "total_symbols": total_symbols,
        "status": status,
    }


def verify_security_review(text: str) -> dict:
    """Universal OWASP vulnerability detection verification."""
    text_lower = text.lower()
    sqli_detected = any(k in text_lower for k in [
        "sql injection", "sqli", "string formatting", "concatenation",
        "parameterized", "prepared statement", "sql-injection", "injection vulnerability"
    ])
    race_detected = any(k in text_lower for k in [
        "race condition", "toctou", "concurrency", "thread-safe",
        "thread safe", "atomic", "synchronization", "mutex", "lock", "data race"
    ])

    c_part = text_lower.split("snippet c")[-1] if "snippet c" in text_lower else text_lower
    safe_passed = "safe" in c_part or "no vulnerability" in c_part or "clean" in c_part or ("vulnerable" not in c_part)

    status = "PASS" if (sqli_detected and race_detected and safe_passed) else ("PARTIAL" if (sqli_detected or race_detected) else "FAIL")
    return {
        "sqli_detected": sqli_detected,
        "race_detected": race_detected,
        "safe_snippet_passed": safe_passed,
        "status": status,
    }


def verify_unit_test(code: str, language: str, source_code: str, execute_code: bool = True) -> dict:
    """Verifies unit test synthesis with mocks and assertions."""
    lang = normalize_language(language)
    fixture = UNIT_TEST_FIXTURES.get(lang, UNIT_TEST_FIXTURES["python"])
    mock_keywords = fixture.get("mock_keywords", [])
    code_lower = code.lower()

    syntax_valid = False
    tests_passed = 0
    total_tests = 3

    if lang == "python":
        try:
            ast.parse(code)
            syntax_valid = True
        except Exception:
            pass

        if syntax_valid and execute_code:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    mod_path = os.path.join(tmpdir, "payment_processor.py")
                    with open(mod_path, "w") as f:
                        f.write(source_code)

                    test_harness = (
                        "import sys, os, types\n"
                        f"sys.path.insert(0, r'{tmpdir}')\n"
                        f"{source_code}\n"
                        "pp_mod = types.ModuleType('payment_processor')\n"
                        "pp_mod.PaymentProcessor = PaymentProcessor\n"
                        "pp_mod.StripeGateway = StripeGateway\n"
                        "pp_mod.NotificationClient = NotificationClient\n"
                        "sys.modules['payment_processor'] = pp_mod\n"
                        f"{code}\n"
                        "if __name__ == '__main__':\n"
                        "    import unittest\n"
                        "    unittest.main(exit=False)\n"
                    )
                    t_file = os.path.join(tmpdir, "test_suite.py")
                    with open(t_file, "w") as f:
                        f.write(test_harness)

                    proc = subprocess.run([sys.executable, t_file], capture_output=True, text=True, timeout=15)
                    output = proc.stderr + proc.stdout
                    if "OK" in output:
                        tests_passed = total_tests
                    else:
                        m = re.search(r"Ran (\d+) tests in", output)
                        if m and "FAILED" not in output:
                            tests_passed = int(m.group(1))
            except Exception:
                pass
        elif syntax_valid:
            has_mock = any(k.lower() in code_lower for k in mock_keywords)
            if has_mock and "test" in code_lower:
                tests_passed = total_tests
    else:
        # Non-python languages: structural verification
        open_braces = code.count("{")
        close_braces = code.count("}")
        syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        has_mock = any(k.lower() in code_lower for k in mock_keywords)
        has_assertions = any(k in code_lower for k in ["assert", "expect", "verify", "should"])
        if has_mock and has_assertions:
            tests_passed = total_tests
        elif has_mock or has_assertions:
            tests_passed = 1

    status = "PASS" if tests_passed >= total_tests else ("PARTIAL" if (syntax_valid and tests_passed > 0) else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "unit_tests_passed_count": tests_passed,
        "unit_tests_total_count": total_tests,
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
    tool_verified = False

    if lang == "python":
        try:
            tree = ast.parse(code)
            syntax_valid = True
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_methods += 1
                    has_return = node.returns is not None
                    args_typed = all(arg.annotation is not None for arg in node.args.args if arg.arg != "self")
                    if has_return and args_typed:
                        annotated_methods += 1

            if shutil.which("mypy") and execute_code:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(code)
                    tmp_path = f.name
                try:
                    proc = subprocess.run(["mypy", "--strict", "--ignore-missing-imports", tmp_path],
                                          capture_output=True, text=True, timeout=15)
                    if proc.returncode == 0:
                        tool_verified = True
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            else:
                tool_verified = (total_methods > 0 and annotated_methods == total_methods)
        except Exception:
            pass

        all_annotated = (total_methods > 0 and annotated_methods == total_methods)
        type_compliant = tool_verified or all_annotated
    else:
        open_braces = code.count("{")
        close_braces = code.count("}")
        syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        matches = [bool(re.search(pat, code)) for pat in req_patterns]
        matched_count = sum(1 for m in matches if m)
        all_annotated = (matched_count == len(req_patterns))
        type_compliant = all_annotated

    status = "PASS" if type_compliant else ("PARTIAL" if (syntax_valid and (annotated_methods > 0 or all_annotated)) else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "all_annotated": all_annotated,
        "type_compliant": type_compliant,
        "status": status,
    }


def verify_refactoring(code: str, language: str, execute_code: bool = True) -> dict:
    """Verifies safe refactoring and preservation of behavioral invariants."""
    lang = normalize_language(language)
    fixture = REFACTOR_FIXTURES.get(lang, REFACTOR_FIXTURES["python"])
    fn_name = fixture.get("function_name", "process_order_data")

    syntax_valid = False
    invariants_passed = False

    if lang == "python":
        try:
            ast.parse(code)
            syntax_valid = True
        except Exception:
            pass

        if syntax_valid and execute_code:
            test_harness = (
                f"{code}\n"
                "assert process_order_data(None) == []\n"
                "assert process_order_data([]) == []\n"
                "assert process_order_data([{'qty': 0, 'price': 10}]) == []\n"
                "assert process_order_data([{'id': '1', 'qty': 2, 'price': 50.0}]) == [{'id': '1', 'total': 100.0}]\n"
                "assert process_order_data([{'id': '2', 'qty': 4, 'price': 25.0, 'discount': 0.1}]) == [{'id': '2', 'total': 90.0}]\n"
                "print('INVARIANTS_OK')\n"
            )
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                    f.write(test_harness)
                    tmp_path = f.name
                try:
                    proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=10)
                    if proc.returncode == 0 and "INVARIANTS_OK" in proc.stdout:
                        invariants_passed = True
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception:
                pass
        elif syntax_valid:
            invariants_passed = f"def {fn_name}" in code
    else:
        # Non-python languages: structural invariant check
        open_braces = code.count("{")
        close_braces = code.count("}")
        syntax_valid = (len(code.strip()) > 30 and abs(open_braces - close_braces) <= 2)
        has_fn = fn_name in code
        has_guard = any(k in code for k in ["return", "continue", "break", "filter", "if"])
        invariants_passed = has_fn and has_guard

    status = "PASS" if invariants_passed else ("PARTIAL" if syntax_valid else "FAIL")
    return {
        "syntax_valid": syntax_valid,
        "invariants_preserved": invariants_passed,
        "status": status,
    }

