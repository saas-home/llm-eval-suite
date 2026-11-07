package authsvc

type UserRepository interface {
	GetByID(userID string) (*User, error)
	Save(user *User) error
	ListActive() ([]*User, error)
}
