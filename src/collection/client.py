import time

import requests


class CodeforcesAPIError(Exception):
    """Raised when the Codeforces API returns an error."""


class CodeforcesClient:
    BASE_URL = "https://codeforces.com/api"
    MIN_REQUEST_INTERVAL = 2.1

    def __init__(self):
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.monotonic() - self._last_request_time

        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

    def _request(self, method: str, params: dict | None = None):
        self._wait_for_rate_limit()

        url = f"{self.BASE_URL}/{method}"

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        self._last_request_time = time.monotonic()

        response.raise_for_status()

        data = response.json()

        if data["status"] != "OK":
            comment = data.get("comment", "Unknown Codeforces API error")
            raise CodeforcesAPIError(
                f"{method} failed: {comment}"
            )

        return data["result"]

    def get_user_info(self, handle: str):
        return self._request(
            "user.info",
            {"handles": handle},
        )

    def get_rating_history(self, handle: str):
        return self._request(
            "user.rating",
            {"handle": handle},
        )

    def get_submissions(
        self,
        handle: str,
        from_: int = 1,
        count: int = 1000,
    ):
        return self._request(
            "user.status",
            {
                "handle": handle,
                "from": from_,
                "count": count,
            },
        )