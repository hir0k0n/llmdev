import pytest
from authenticator import Authenticator

@pytest.mark.parametrize("username, password", [
    ("user", "pass")
])
def test_register(username, password):
    auth = Authenticator()
    auth.register(username, password)
    assert username in auth.users

def test_register_same_name():
    auth = Authenticator()
    auth.register("user1", "pass1")
    with pytest.raises(ValueError, match="エラー: ユーザーは既に存在します。"):
        auth.register("user1", "pass2")

@pytest.mark.parametrize("username, password", [
    ("user", "pass")
])
def test_login(username, password):
    auth = Authenticator()
    auth.register(username, password)
    assert auth.login(username, password) == "ログイン成功"

def test_login_bad_password():
    auth = Authenticator()
    auth.register("user", "pass")
    with pytest.raises(ValueError, match="エラー: ユーザー名またはパスワードが正しくありません。"):
        auth.login("user", "pass1")