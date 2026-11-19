"""
Authentication and role-based access for the HTTP transport.

Stdio and the terminal are local: the caller is the OS user, with every role. Over HTTP a
caller presents `Authorization: Bearer <token>`, verified by one or more of:

  API tokens  NOVUS_AUTH_TOKENS_FILE: a TOML file of SHA-256 token hashes, each with a name
              (the audited principal), a role and an optional expiry date. Create entries
              with `novuspipeline token create`. Plaintext tokens are never stored.

                  [[token]]
                  name = "ci-bot"
                  sha256 = "9f86d0…"
                  role = "developer"
                  expires = 2027-06-30

  Shared token NOVUS_AUTH_TOKEN: one plaintext token with the admin role (simple setups).

  OIDC / JWT  NOVUS_OIDC_JWKS_URI, NOVUS_OIDC_ISSUER and NOVUS_OIDC_AUDIENCE (all required):
              tokens from your identity provider (Entra ID, Okta, Auth0, Keycloak, …),
              signature-checked against its JWKS. Roles come from the token's scopes:
              novus:read, novus:write, novus:admin. NOVUS_OIDC_ALGORITHM (default RS256).

Roles map onto the tools' MCP annotations:

  viewer     novus:read    read-only tools
  developer  novus:write   + tools that write only to the index, a sandbox or a separate branch
  admin      novus:admin   + destructive tools (working copy, checked-out branch, deleting data)
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import os
import secrets
import tomllib
from dataclasses import dataclass

from fastmcp.server.auth import AccessToken, TokenVerifier

ENV_TOKENS_FILE = "NOVUS_AUTH_TOKENS_FILE"
ENV_SHARED_TOKEN = "NOVUS_AUTH_TOKEN"
ENV_OIDC_JWKS = "NOVUS_OIDC_JWKS_URI"
ENV_OIDC_ISSUER = "NOVUS_OIDC_ISSUER"
ENV_OIDC_AUDIENCE = "NOVUS_OIDC_AUDIENCE"
ENV_OIDC_ALGORITHM = "NOVUS_OIDC_ALGORITHM"

SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN = "novus:read", "novus:write", "novus:admin"
ROLES = {
    "viewer": (SCOPE_READ,),
    "developer": (SCOPE_READ, SCOPE_WRITE),
    "admin": (SCOPE_READ, SCOPE_WRITE, SCOPE_ADMIN),
}
# Tool annotation kind -> the scope a caller needs. admin implies write implies read.
REQUIRED_SCOPE = {"read": SCOPE_READ, "write": SCOPE_WRITE, "destructive": SCOPE_ADMIN}
_IMPLIED = {SCOPE_ADMIN: {SCOPE_ADMIN, SCOPE_WRITE, SCOPE_READ}, SCOPE_WRITE: {SCOPE_WRITE, SCOPE_READ},
            SCOPE_READ: {SCOPE_READ}}


class AuthConfigError(RuntimeError):
    pass


def effective_scopes(scopes: list[str] | tuple[str, ...]) -> set[str]:
    granted: set[str] = set()
    for scope in scopes:
        granted |= _IMPLIED.get(scope, set())
    return granted


def allows(scopes: list[str] | tuple[str, ...], kind: str) -> bool:
    return REQUIRED_SCOPE[kind] in effective_scopes(scopes)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return "novus_" + secrets.token_urlsafe(32)


@dataclass(frozen=True)
class TokenEntry:
    name: str
    sha256: str
    role: str
    expires: _dt.date | None = None


def load_tokens_file(path: str) -> list[TokenEntry]:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise AuthConfigError(f"cannot read {ENV_TOKENS_FILE} `{path}`: {e}") from e
    entries, seen = [], set()
    for i, raw in enumerate(data.get("token") or [], 1):
        where = f"`{path}` token #{i}"
        if not isinstance(raw, dict):
            raise AuthConfigError(f"{where}: must be a [[token]] table")
        name, digest, role = raw.get("name"), str(raw.get("sha256", "")).lower(), raw.get("role", "viewer")
        if not isinstance(name, str) or not name.strip():
            raise AuthConfigError(f"{where}: `name` is required")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AuthConfigError(f"{where} ({name}): `sha256` must be a 64-character hex digest")
        if role not in ROLES:
            raise AuthConfigError(f"{where} ({name}): `role` must be one of {', '.join(ROLES)}")
        expires = raw.get("expires")
        if isinstance(expires, _dt.datetime):
            expires = expires.date()
        if expires is not None and not isinstance(expires, _dt.date):
            raise AuthConfigError(f"{where} ({name}): `expires` must be a date, e.g. 2027-06-30")
        if digest in seen:
            raise AuthConfigError(f"{where} ({name}): duplicate token hash")
        seen.add(digest)
        entries.append(TokenEntry(name.strip(), digest, role, expires))
    if not entries:
        raise AuthConfigError(f"{ENV_TOKENS_FILE} `{path}` defines no [[token]] entries")
    return entries


class HashedTokenVerifier(TokenVerifier):
    """Opaque bearer tokens checked against SHA-256 hashes (constant-time), with roles and expiry."""

    def __init__(self, entries: list[TokenEntry], auth_kind: str = "token") -> None:
        super().__init__()
        self.entries = entries
        self.auth_kind = auth_kind

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = hash_token(token)
        match = None
        for entry in self.entries:  # compare against every entry: no early exit timing signal
            if hmac.compare_digest(entry.sha256, digest):
                match = entry
        if match is None:
            return None
        expires_at = None
        if match.expires is not None:
            end = _dt.datetime.combine(match.expires, _dt.time.max, tzinfo=_dt.UTC)
            if _dt.datetime.now(_dt.UTC) > end:
                return None
            expires_at = int(end.timestamp())
        return AccessToken(token=token, client_id=match.name, scopes=list(ROLES[match.role]),
                           expires_at=expires_at, claims={"novus_auth": self.auth_kind, "role": match.role})


class ChainVerifier(TokenVerifier):
    """Accepts a token any of the configured verifiers accepts, tried in order."""

    def __init__(self, verifiers: list[TokenVerifier]) -> None:
        super().__init__()
        self.verifiers = verifiers

    async def verify_token(self, token: str) -> AccessToken | None:
        for verifier in self.verifiers:
            try:
                result = await verifier.verify_token(token)
            except Exception:
                result = None
            if result is not None:
                return result
        return None


def _oidc_verifier() -> TokenVerifier | None:
    jwks = os.environ.get(ENV_OIDC_JWKS, "").strip()
    issuer = os.environ.get(ENV_OIDC_ISSUER, "").strip()
    audience = os.environ.get(ENV_OIDC_AUDIENCE, "").strip()
    if not (jwks or issuer or audience):
        return None
    missing = [n for n, v in ((ENV_OIDC_JWKS, jwks), (ENV_OIDC_ISSUER, issuer), (ENV_OIDC_AUDIENCE, audience)) if not v]
    if missing:
        raise AuthConfigError(f"OIDC needs {', '.join(missing)} as well (an unchecked issuer or audience would "
                              "accept tokens issued for other applications)")
    if not jwks.startswith("https://") and not jwks.startswith("http://localhost"):
        raise AuthConfigError(f"{ENV_OIDC_JWKS} must be an https:// URL")
    from fastmcp.server.auth.providers.jwt import JWTVerifier
    return JWTVerifier(jwks_uri=jwks, issuer=issuer, audience=audience,
                       algorithm=os.environ.get(ENV_OIDC_ALGORITHM, "").strip() or "RS256")


def build_verifier() -> tuple[TokenVerifier | None, list[str]]:
    """The configured verifier (None: no authentication) and a description of each method."""
    verifiers: list[TokenVerifier] = []
    methods: list[str] = []
    path = os.environ.get(ENV_TOKENS_FILE, "").strip()
    if path:
        entries = load_tokens_file(os.path.abspath(os.path.expanduser(path)))
        verifiers.append(HashedTokenVerifier(entries))
        methods.append(f"API tokens ({len(entries)} in {path})")
    shared = os.environ.get(ENV_SHARED_TOKEN, "")
    if shared:
        if len(shared) < 24:
            raise AuthConfigError(f"{ENV_SHARED_TOKEN} must be at least 24 characters; "
                                  "generate one with `novuspipeline token create`")
        verifiers.append(HashedTokenVerifier([TokenEntry("shared-token", hash_token(shared), "admin")]))
        methods.append(f"shared token ({ENV_SHARED_TOKEN}, admin)")
    oidc = _oidc_verifier()
    if oidc is not None:
        verifiers.append(oidc)
        methods.append(f"OIDC ({os.environ[ENV_OIDC_ISSUER]})")
    if not verifiers:
        return None, []
    return (verifiers[0] if len(verifiers) == 1 else ChainVerifier(verifiers)), methods


def caller_from_token(token: AccessToken):
    """The audited identity of an authenticated request."""
    from .audit import Caller
    claims = token.claims or {}
    kind = claims.get("novus_auth") or "oidc"
    principal = token.client_id if kind != "oidc" else (claims.get("preferred_username") or claims.get("email")
                                                         or token.subject or claims.get("sub") or token.client_id)
    return Caller(str(principal), kind, tuple(sorted(effective_scopes(token.scopes))))


def token_entry_toml(name: str, token: str, role: str, expires: str = "") -> str:
    lines = ["[[token]]", f'name = "{name}"', f'sha256 = "{hash_token(token)}"', f'role = "{role}"']
    if expires:
        lines.append(f"expires = {expires}")
    return "\n".join(lines)
