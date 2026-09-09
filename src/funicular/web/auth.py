"""GitHub App OAuth (user authorisation) + server-side sessions + CSRF + rate limiting.

Threat model: the app is private. The only people who may reach any document are the GitHub
accounts whose *numeric* IDs are in FUNICULAR_ALLOWED_GITHUB_IDS. We never store the GitHub
access token — it is used once to read /user and then discarded.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import HTTPException, Request, Response

from ..config import Settings
from .store import Store

log = logging.getLogger(__name__)

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"  # noqa: S105 - URL, not a secret
GITHUB_USER = "https://api.github.com/user"


class RateLimiter:
    """Fixed-window counter per key, in memory. Good enough for a single private process."""

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            return True


@dataclass
class User:
    id: int
    login: str
    csrf: str


class Auth:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.secure = urlparse(settings.public_url).scheme == "https"
        self.cookie = "__Host-funicular_session" if self.secure else "funicular_session"
        self.login_limiter = RateLimiter(limit=10, window=60)
        self.fail_limiter = RateLimiter(limit=5, window=300)
        self._client = httpx.Client(timeout=15, headers={"User-Agent": "fuzzy-funicular"})

    # ------------------------------------------------------------- sessions
    def current_user(self, request: Request) -> User | None:
        sid = request.cookies.get(self.cookie)
        if not sid:
            return None
        s = self.settings
        row = self.store.get_session(
            sid,
            idle_seconds=s.session_idle_minutes * 60,
            absolute_seconds=s.session_absolute_hours * 3600,
        )
        if not row:
            return None
        if row["user_id"] not in s.allowed_github_ids:
            # Allowlist shrank since login: revoke immediately.
            self.store.delete_session(sid)
            return None
        return User(id=row["user_id"], login=row["login"], csrf=row["csrf"])

    def set_session_cookie(self, response: Response, sid: str) -> None:
        response.set_cookie(
            self.cookie,
            sid,
            max_age=self.settings.session_absolute_hours * 3600,
            httponly=True,
            secure=self.secure,
            samesite="lax",
            path="/",
        )

    def clear_session(self, request: Request, response: Response) -> None:
        sid = request.cookies.get(self.cookie)
        if sid:
            self.store.delete_session(sid)
        response.delete_cookie(
            self.cookie, path="/", httponly=True, secure=self.secure, samesite="lax"
        )

    # ------------------------------------------------------------- CSRF
    def check_csrf(self, request: Request, user: User, token: str | None) -> None:
        origin = request.headers.get("origin") or request.headers.get("referer")
        if origin:
            o = urlparse(origin)
            expected = urlparse(self.settings.public_url)
            same = (o.scheme, o.hostname, o.port or _default_port(o.scheme)) == (
                expected.scheme,
                expected.hostname,
                expected.port or _default_port(expected.scheme),
            )
            if not same and not (_is_loopback(o.hostname) and _is_loopback(expected.hostname)):
                raise HTTPException(403, "cross-site request blocked")
        header = request.headers.get("x-csrf-token")
        candidate = token or header or ""
        if not candidate or not hmac.compare_digest(candidate, user.csrf):
            raise HTTPException(403, "invalid CSRF token")

    # ------------------------------------------------------------- OAuth
    def login_url(self, next_path: str) -> str:
        if not next_path.startswith("/") or next_path.startswith("//"):
            next_path = "/"
        state = self.store.create_state(next_path)
        params = {
            "client_id": self.settings.github_client_id,
            "redirect_uri": f"{self.settings.public_url}/auth/callback",
            "state": state,
            # GitHub Apps ignore scope (permissions come from the App); harmless for OAuth Apps.
            "scope": "read:user",
            "allow_signup": "false",
        }
        return f"{GITHUB_AUTHORIZE}?{urlencode(params)}"

    def exchange_code(self, code: str) -> str:
        r = self._client.post(
            GITHUB_TOKEN,
            data={
                "client_id": self.settings.github_client_id,
                "client_secret": self.settings.github_client_secret,
                "code": code,
                "redirect_uri": f"{self.settings.public_url}/auth/callback",
            },
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        token = data.get("access_token")
        if not token:
            raise HTTPException(
                401, f"GitHub did not issue a token: {data.get('error', 'unknown')}"
            )
        return token

    def fetch_user(self, token: str) -> tuple[int, str]:
        r = self._client.get(
            GITHUB_USER,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        r.raise_for_status()
        data = r.json()
        return int(data["id"]), str(data.get("login", ""))

    def complete_login(self, request: Request, code: str, state: str) -> tuple[str, str]:
        """Returns (session_id, next_path). Raises HTTPException on any failure."""
        ip = client_ip(request)
        if not self.fail_limiter.check(f"cb:{ip}"):
            raise HTTPException(429, "too many sign-in attempts; wait a few minutes")
        next_path = self.store.consume_state(state)
        if next_path is None:
            self.store.audit("login.bad_state", ip)
            raise HTTPException(400, "sign-in state expired or invalid; start again")
        token = self.exchange_code(code)
        try:
            user_id, login = self.fetch_user(token)
        finally:
            token = ""  # noqa: F841 - make the intent explicit: never retained
        if user_id not in self.settings.allowed_github_ids:
            self.store.audit("login.denied", f"{login} ({user_id}) from {ip}")
            log.warning("denied GitHub user %s (%s)", login, user_id)
            raise HTTPException(403, "this GitHub account is not on the allowlist")
        sid, _csrf = self.store.create_session(user_id, login)
        self.store.audit("login.ok", f"{login} ({user_id}) from {ip}")
        return sid, next_path


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "?"


def _default_port(scheme: str | None) -> int:
    return 443 if scheme == "https" else 80


def _is_loopback(host: str | None) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def new_secret() -> str:
    return secrets.token_urlsafe(48)
