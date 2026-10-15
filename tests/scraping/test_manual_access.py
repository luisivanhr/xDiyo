"""Offline tests: manual confirmation and browser-session continuity."""
import json

import pytest
from selenium.common.exceptions import TimeoutException, WebDriverException

from scraping.sofascore.client import BrowserClient
from scraping.sofascore.config import API_ORIGIN, RatePolicy
from scraping.sofascore.models import Job


def network_messages(url, request_id="1", status=200):
    return [
        {"method":"Network.requestWillBeSent", "params":{
            "requestId":request_id,"request":{"url":url}}},
        {"method":"Network.responseReceived", "params":{
            "requestId":request_id,"response":{"url":url,"status":status,
                "headers":{"content-type":"application/json"}}}},
    ]


class Browser:
    def __init__(self, timeout=False):
        self.visits=[]
        self.pending=[]
        self.session_confirmed=False
        self.timeout=timeout

    def get(self,url):
        self.visits.append(url)
        if url == "https://www.sofascore.com/":
            self.pending += network_messages(API_ORIGIN+"/sport/football/events?private=omit")
            if self.timeout:
                raise TimeoutException("slow homepage")
        elif url.startswith(API_ORIGIN):
            assert self.session_confirmed
            self.pending += network_messages(url,"2")

    def get_log(self,kind):
        result=[{"message":json.dumps({"message":message})} for message in self.pending]
        self.pending=[]
        return result

    def execute_cdp_cmd(self,command,arguments):
        assert command=="Network.getResponseBody"
        return {"body":'{"shotmap":[]}',"base64Encoded":False}


@pytest.mark.parametrize("timeout",[False,True])
def test_manual_confirmation_keeps_session_and_separates_website_traffic(timeout):
    client=BrowserClient(RatePolicy())
    browser=client.driver=Browser(timeout)
    observed=[]
    def confirm(prompt):
        assert browser.visits==["https://www.sofascore.com/"]
        assert "press Enter" in prompt
        browser.session_confirmed=True
        return ""
    client.prepare_manual_access(on_requests=observed.extend,confirm=confirm)
    assert observed==[{"path":"/sport/football/events","status":200,"retry_after":None}]
    assert browser.visits[-1]=="about:blank"
    response=client.fetch(Job("/event/100/shotmap","shotmap",100))
    assert client.driver is browser and response.status==200
    assert response.payload=={"shotmap":[]} and response.additional_api_requests==0


def test_manual_cancel_stops_page_and_accounts_observed_requests():
    client=BrowserClient(RatePolicy())
    client.driver=Browser()
    observed=[]
    def cancel(prompt):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        client.prepare_manual_access(on_requests=observed.extend,confirm=cancel)
    assert len(observed)==1
    assert client.driver.visits==["https://www.sofascore.com/","about:blank"]


def test_closed_manual_browser_does_not_report_success():
    client=BrowserClient(RatePolicy())
    client.driver=Browser()
    observed=[]
    def confirm(prompt):
        def closed(kind):
            raise WebDriverException("User closed browser")
        client.driver.get_log=closed
    with pytest.raises(WebDriverException):
        client.prepare_manual_access(on_requests=observed.extend,confirm=confirm)
