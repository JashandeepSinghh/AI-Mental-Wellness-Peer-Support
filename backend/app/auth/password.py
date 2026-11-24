
"""
Password Security Module
========================

Responsibilities:
- Hash passwords using Argon2id.
- Verify passwords against stored hashes.
- Detect when an existing password hash needs upgrading.

Security:
- Plain-text passwords are never stored.
- Password hashes are generated using pwdlib.
- Argon2id is used through the recommended configuration.
- Password policy validation is handled separately.
"""

from pwdlib import PasswordHash


# ============================================================
# 1. PASSWORD HASHER INITIALIZATION
# ============================================================

# Use pwdlib's recommended secure password-hashing configuration.
# The current recommended algorithm is Argon2id.

password_hasher = PasswordHash.recommended()


# ============================================================
# 2. PASSWORD HASHING
# ============================================================

def hash_password(password: str) -> str:
    """
    Generate a secure Argon2id hash from a plain-text password.

    Args:
        password: Original user password.

    Returns:
        str: Secure password hash suitable for database storage.

    Note:
        The returned hash does not contain the original password.
    """

    if not isinstance(password, str):
        raise TypeError("Password must be a string.")

    if not password:
        raise ValueError("Password cannot be empty.")

    return password_hasher.hash(password)


# ============================================================
# 3. PASSWORD VERIFICATION
# ============================================================

def verify_password(
    plain_password: str,
    stored_hash: str,
) -> bool:
    """
    Verify a plain-text password against a stored password hash.

    Args:
        plain_password: Password supplied during login.
        stored_hash: Previously generated password hash.

    Returns:
        bool: True if the password matches, otherwise False.
    """

    if not isinstance(plain_password, str):
        return False

    if not isinstance(stored_hash, str):
        return False

    if not plain_password or not stored_hash:
        return False

    return password_hasher.verify(
        plain_password,
        stored_hash,
    )


# ============================================================
# 4. PASSWORD VERIFICATION AND HASH UPGRADING
# ============================================================

def verify_and_update_password(
    plain_password: str,
    stored_hash: str,
) -> tuple[bool, str | None]:
    """
    Verify a password and determine whether its hash needs upgrading.

    Args:
        plain_password: Password supplied during authentication.
        stored_hash: Existing password hash from the database.

    Returns:
        tuple:
            (False, None):
                Password verification failed.

            (True, None):
                Password is correct and its hash is up to date.

            (True, new_hash):
                Password is correct and the hash should be upgraded.

    Important:
        The caller must persist new_hash to the database when
        an upgrade is required.
    """

    if not isinstance(plain_password, str):
        return False, None

    if not isinstance(stored_hash, str):
        return False, None

    if not plain_password or not stored_hash:
        return False, None

    return password_hasher.verify_and_update(
        plain_password,
        stored_hash,
    )
