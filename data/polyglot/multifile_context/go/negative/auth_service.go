package authsvc

type AuthService struct {
	repo   UserRepository
	config TenantConfig
}

func NewAuthService(repo UserRepository, config TenantConfig) *AuthService {
	return &AuthService{repo: repo, config: config}
}

func (s *AuthService) Authenticate(userID string) (*AuthToken, error) {
	user, err := s.repo.GetByID(userID)
	if err != nil {
		return nil, err
	}
	if user == nil {
		return nil, nil
	}
	return &AuthToken{Token: "tok_" + user.ID, ExpiresIn: s.config.MaxTokenTTLSeconds}, nil
}
