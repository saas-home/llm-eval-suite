package authsvc

type User struct {
	ID       string
	Email    string
	IsActive bool
}

type TenantConfig struct {
	TenantID           string
	MaxTokenTTLSeconds int
}

type AuthToken struct {
	Token     string
	ExpiresIn int
}
