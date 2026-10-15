from __future__ import annotations

import base64
import json
import sys

from .config import API_ORIGIN
from .models import Response


def manual_api_requests(messages):
    """Sanitized request accounting for the user-operated website visit."""
    requests = []
    responses = {}
    for message in messages:
        params = message.get("params", {})
        if message.get("method") == "Network.requestWillBeSent":
            url = params.get("request", {}).get("url", "")
            if url.startswith(API_ORIGIN + "/"):
                requests.append((params.get("requestId"), url[len(API_ORIGIN):].split("?",1)[0].split("#",1)[0]))
        elif message.get("method") == "Network.responseReceived":
            responses[params.get("requestId")] = params.get("response", {})
    result = []
    for request_id, path in requests:
        response = responses.get(request_id, {})
        headers = {key.lower():str(value) for key,value in response.get("headers", {}).items()}
        result.append({"path":path, "status":response.get("status"),
                       "retry_after":headers.get("retry-after")})
    return result


def matching_response(messages: list[dict], url: str):
    """Associate network responses with this navigation, never the old DOM."""
    requests = []
    responses = []
    for message in messages:
        params = message.get("params", {})
        if message.get("method") == "Network.requestWillBeSent":
            request_url = params.get("request", {}).get("url", "")
            if request_url.startswith(API_ORIGIN + "/"):
                requests.append((params.get("requestId"), request_url))
        if message.get("method") == "Network.responseReceived":
            responses.append(params)
    intended = {rid for rid, requested in requests if requested == url}
    candidates = [p for p in responses if p.get("requestId") in intended
                  and p.get("response", {}).get("url") == url]
    return (candidates[-1] if candidates else None), max(0, len(requests)-1)


class BrowserClient:
    """One plain Chrome session using the user's external network/VPN route.

    No proxy rotation, extension VPN, anti-detection switches or automatic
    challenge handling. Response bodies are read through Chrome's network API.
    """

    def __init__(self, policy, *, driver_path=None, chrome_binary=None,
                 locale="en", readiness_pause=False):
        self.policy = policy
        self.driver_path = driver_path
        self.chrome_binary = chrome_binary
        self.locale = locale
        self.readiness_pause = readiness_pause
        self.driver = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.driver is not None:
            try:
                self.driver.quit()
            finally:
                self.driver = None

    def _start(self):
        if self.driver is not None:
            return
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        options = webdriver.ChromeOptions()
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        options.add_experimental_option("prefs", {"intl.accept_languages": self.locale})
        if self.chrome_binary:
            options.binary_location = str(self.chrome_binary)
        try:
            self.driver = webdriver.Chrome(service=Service(executable_path=str(self.driver_path))
                                           if self.driver_path else Service(), options=options)
            self.driver.set_page_load_timeout(self.policy.page_load_timeout_seconds)
            self.driver.execute_cdp_cmd("Network.enable", {})
            self.driver.execute_cdp_cmd("Network.setCacheDisabled", {"cacheDisabled": True})
            if self.readiness_pause:
                input("Chrome is ready. Confirm your external network/VPN is ready, then press Enter: ")
        except BaseException:
            self.close()
            raise

    def prepare_manual_access(self, *, on_requests, confirm=None):
        """Let the user complete the site's challenge in this exact session.

        Confirmation permits an API retry; it is not proof that access works.
        No challenge is detected, clicked, solved or bypassed automatically.
        """
        from selenium.common.exceptions import TimeoutException, WebDriverException
        self._start()
        self.driver.get_log("performance")
        messages = []
        try:
            try:
                self.driver.get("https://www.sofascore.com/")
            except TimeoutException:
                # The user can still interact with a slowly loading challenge.
                pass
            (confirm or input)(
                "In the scraper's Chrome window, complete any CAPTCHA and let Sofascore load. "
                "Then return here and press Enter to continue (Interrupt Kernel to cancel): ")
        finally:
            # Stop website polling while retaining this browser's cookies. Only
            # sanitized request metadata enters the ledger, never cookie values.
            unwinding = sys.exc_info()[0] is not None
            browser_error = None
            try:
                messages.extend(json.loads(item["message"])["message"]
                                for item in self.driver.get_log("performance"))
                self.driver.get("about:blank")
                messages.extend(json.loads(item["message"])["message"]
                                for item in self.driver.get_log("performance"))
            except WebDriverException as exc:
                browser_error = exc
            on_requests(manual_api_requests(messages))
            if browser_error is not None and not unwinding:
                raise browser_error

    def fetch(self, job):
        from selenium.common.exceptions import TimeoutException, WebDriverException
        self._start()
        url = API_ORIGIN + job.path
        self.driver.get_log("performance")  # discard all prior navigation records
        navigation_error = None
        try:
            self.driver.get(url)
        except TimeoutException:
            navigation_error = "navigation_timeout"
            self.driver.execute_cdp_cmd("Page.stopLoading", {})
        except WebDriverException as exc:
            # Never retain exception text containing page contents or credentials.
            navigation_error = "blocked_by_client" if "ERR_BLOCKED_BY_CLIENT" in str(exc) else "navigation_failed"
        messages = [json.loads(item["message"])["message"] for item in self.driver.get_log("performance")]
        match, extra = matching_response(messages, url)
        if match is None:
            return Response(None, error=navigation_error or "unmatched_response", additional_api_requests=extra)
        response = match["response"]
        status = int(response["status"])
        headers = {k.lower(): str(v) for k,v in response.get("headers", {}).items()
                   if k.lower() in {"content-type", "retry-after", "date", "etag", "last-modified"}}
        if navigation_error:
            return Response(status, headers=headers, error=navigation_error, additional_api_requests=extra)
        if status != 200:
            return Response(status, headers=headers, additional_api_requests=extra)
        if "json" not in headers.get("content-type", response.get("mimeType", "")).lower():
            return Response(status, headers=headers, error="non_json_response", additional_api_requests=extra)
        try:
            body = self.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId":match["requestId"]})
            content = base64.b64decode(body["body"]).decode("utf-8") if body.get("base64Encoded") else body["body"]
            payload = json.loads(content)
        except (WebDriverException, ValueError, UnicodeError):
            return Response(status, headers=headers, error="invalid_response_body", additional_api_requests=extra)
        return Response(status, payload, headers, additional_api_requests=extra)
