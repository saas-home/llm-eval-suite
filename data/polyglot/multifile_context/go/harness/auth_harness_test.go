package authsvc

import (
	"errors"
	"reflect"
	"testing"
)

type fakeRepo struct {
	users map[string]*User
	calls []string
	err   error
}

func (r *fakeRepo) GetByID(id string) (*User, error) {
	r.calls = append(r.calls, id)
	if r.err != nil {
		return nil, r.err
	}
	return r.users[id], nil
}
func (r *fakeRepo) Save(u *User) error            { r.users[u.ID] = u; return nil }
func (r *fakeRepo) ListActive() ([]*User, error) { return nil, nil }

func TestAuthServiceBehavior(t *testing.T) {
	repo := &fakeRepo{users: map[string]*User{
		"u1": {ID: "u1", Email: "a@x.io", IsActive: true},
		"u2": {ID: "u2", Email: "b@x.io", IsActive: false},
	}}
	svc := NewAuthService(repo, TenantConfig{TenantID: "t1", MaxTokenTTLSeconds: 900})
	tok, err := svc.Authenticate("u1")
	if err != nil || tok == nil || tok.Token != "tok_u1" || tok.ExpiresIn != 900 {
		t.Fatalf("active user token wrong: %+v, %v", tok, err)
	}
	if tok, err := svc.Authenticate("u2"); tok != nil || err != nil {
		t.Fatalf("inactive user must get nil, nil: %+v, %v", tok, err)
	}
	if tok, err := svc.Authenticate("nope"); tok != nil || err != nil {
		t.Fatalf("unknown user must get nil, nil: %+v, %v", tok, err)
	}
	if !reflect.DeepEqual(repo.calls, []string{"u1", "u2", "nope"}) {
		t.Fatalf("must look users up via the injected repository: %v", repo.calls)
	}
	boom := errors.New("db down")
	failing := &fakeRepo{users: map[string]*User{}, err: boom}
	if tok, err := NewAuthService(failing, TenantConfig{}).Authenticate("u1"); tok != nil || !errors.Is(err, boom) {
		t.Fatalf("repository error must be returned: %+v, %v", tok, err)
	}
}
