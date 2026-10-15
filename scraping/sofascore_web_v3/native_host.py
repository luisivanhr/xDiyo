"""Chrome Native Messaging transport. Only local campaign operations are exposed."""
from contextlib import ExitStack
import json
import os
import struct
import sys
import time

from .campaign import BASE, ROOT, PROTOCOL_VERSION, Campaign, campaign_path, collector_lock, read_json
from .discovery import DISCOVERY_BASE, open_discovery, league_catalog, prepare_discovery

HOST_NAME = "com.xdiyo.sofascore"
HOST_DIRECTORY = ROOT / "data/website_capture_v3/native_host"
MAX_INPUT = 12 * 1024 * 1024
MAX_OUTPUT = 1024 * 1024


def read_exact(stream, length):
    chunks = bytearray()
    while len(chunks) < length:
        part = stream.read(length-len(chunks))
        if not part:
            if not chunks: return None
            raise ValueError("Truncated native message")
        chunks.extend(part)
    return bytes(chunks)


def read_message(stream):
    header = read_exact(stream,4)
    if header is None: return None
    size = struct.unpack("=I",header)[0]
    if not 0 < size <= MAX_INPUT: raise ValueError("Native message exceeds the input limit")
    body = read_exact(stream,size)
    if body is None: raise ValueError("Truncated native message")
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value,dict): raise ValueError("Native message must be an object")
    return value


def write_message(stream, value):
    body = json.dumps(value,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")
    if len(body) > MAX_OUTPUT: raise ValueError("Native response exceeds the output limit")
    stream.write(struct.pack("=I",len(body))+body)
    stream.flush()


class Host:
    def __init__(self, *, base=BASE, discovery_base=DISCOVERY_BASE, lock_path=None, monotonic=time.perf_counter, sleep=time.sleep):
        self.base = base
        self.discovery_base = discovery_base
        self.lock_path = lock_path or ROOT/"data/collection_state/collector.lock"
        self.stack = ExitStack()
        self.campaign = None
        self.discovery = None
        self.monotonic, self.sleep = monotonic, sleep

    def delay(self, event_id, lease, milliseconds):
        """Bounded local pacing; keep pause checks responsive without provider traffic."""
        # perf_counter is monotonic and avoids coarse GetTickCount64 resolution
        # used by time.monotonic in some Windows Python runtimes.
        self.campaign.verify_lease(event_id, lease)
        if type(milliseconds) is not int or not 1 <= milliseconds <= 1000:
            raise ValueError("Local pacing delay must be an integer from 1 to 1000 ms")
        started = self.monotonic()
        deadline = started + milliseconds / 1000
        while True:
            pause = (self.campaign.directory / "PAUSE").exists()
            limit = self.campaign.run_limit_reached()
            remaining = deadline - self.monotonic()
            if pause or limit or remaining <= 0:
                break
            self.sleep(min(0.1, remaining))
        self.campaign.verify_lease(event_id, lease)
        return {"waited_ms": (self.monotonic() - started) * 1000,
                "pause_requested": pause or limit, "run_limit_reached": limit}

    def close(self):
        if self.campaign and self.campaign.current: self.campaign.pause("browser_connection_closed")
        if self.discovery and self.discovery.current: self.discovery.pause("browser_connection_closed")
        self.stack.close()

    def dispatch(self, message):
        operation = message.get("operation")
        if not isinstance(operation, str): raise ValueError("Missing native operation")
        if operation == "hello":
            if message.get("protocol") != PROTOCOL_VERSION: raise ValueError("Extension/Python protocol version mismatch; reload the extension")
            return {"protocol":PROTOCOL_VERSION,"version":"0.4.1","host":HOST_NAME}
        if operation == "discovery_catalog":
            return {"leagues": league_catalog()}
        if operation in {"discovery_create", "discovery_open"}:
            if self.campaign or self.discovery:
                raise ValueError("Disconnect the current campaign or discovery first")
            self.stack.enter_context(collector_lock(self.lock_path))
            if operation == "discovery_create":
                result = prepare_discovery(season=message.get("season"), leagues=message.get("leagues"),
                    include_lineups=message.get("include_lineups", True),
                    random_delay_max_seconds=message.get("random_delay_max_seconds"),
                    mode=message.get("mode", "season"), base=self.discovery_base)
                directory = campaign_path(result["discovery_id"], self.discovery_base)
            else:
                directory = campaign_path(message.get("discovery_id"), self.discovery_base)
                if not (directory/"config.json").is_file(): raise ValueError("Unknown discovery ID")
            self.discovery = self.stack.enter_context(open_discovery(directory))
            return self.discovery.status()
        if operation.startswith("discovery_"):
            if not self.discovery: raise ValueError("Open or create a discovery first")
            if operation == "discovery_next": return self.discovery.next_job(resume=message.get("resume") is True)
            if operation == "discovery_observe": return self.discovery.observe(message.get("path"), message.get("lease"), message["entry"])
            if operation == "discovery_status": return self.discovery.status()
            if operation == "discovery_pause": return self.discovery.pause(message.get("reason", "Paused by user"))
            if operation == "discovery_prepare":
                if self.discovery.current: raise ValueError("Finish discovery before preparing collection")
                return self.discovery.prepare_collection(campaign_base=self.base)
            if operation == "discovery_delay":
                milliseconds = message.get("milliseconds")
                if type(milliseconds) is not int or not 1 <= milliseconds <= 1000:
                    raise ValueError("Discovery pacing delay must be 1 to 1000 ms")
                self.sleep(milliseconds / 1000)
                return {"waited_ms": milliseconds}
            raise ValueError("Unknown discovery operation")
        if operation == "open":
            if self.campaign or self.discovery: raise ValueError("A campaign or discovery is already attached to this connection")
            directory = campaign_path(message.get("campaign_id"),self.base)
            if not (directory/"config.json").is_file(): raise ValueError("Prepare this campaign in the notebook first")
            self.stack.enter_context(collector_lock(self.lock_path))
            self.campaign = self.stack.enter_context(Campaign(directory))
            self.campaign.write_status("connected")
            return {"campaign_id":self.campaign.config["campaign_id"],"policy":self.campaign.policy}
        if not self.campaign: raise ValueError("Open a prepared campaign first")
        if operation == "next": return self.campaign.next_match(resume=message.get("resume") is True)
        if operation == "pause": return self.campaign.pause(str(message.get("reason","user_paused"))[:200])
        if operation == "status": return self.campaign.write_status()
        eid, lease = message.get("event_id"), message.get("lease")
        if type(eid) is not int: raise ValueError("Invalid event ID")
        if operation == "observe": return self.campaign.observe(eid,lease,message["entry"])
        if operation == "progress": return self.campaign.progress(eid,lease)
        if operation == "delay": return self.delay(eid,lease,message.get("milliseconds"))
        if operation == "finish":
            return self.campaign.finish(eid,lease,page_event_id=message.get("page_event_id"),reason=str(message.get("reason","views_exhausted"))[:200],
                                       page_evidence=message.get("page_evidence"))
        raise ValueError("Unknown native operation")


def serve(input_stream, output_stream, *, base=BASE, lock_path=None):
    host = Host(base=base,lock_path=lock_path)
    try:
        while True:
            message = read_message(input_stream)
            if message is None: break
            try:
                result = host.dispatch(message)
                response = {"id":message.get("id"),"ok":True,"result":result}
            except (ValueError,KeyError,TypeError,RuntimeError,OSError) as error:
                response = {"id":message.get("id"),"ok":False,"error":str(error)}
            write_message(output_stream,response)
    finally:
        host.close()


def main():
    manifest = read_json(HOST_DIRECTORY/f"{HOST_NAME}.json")
    if len(sys.argv) < 2 or sys.argv[1] not in manifest["allowed_origins"]:
        raise SystemExit("This writer only accepts the configured Chrome extension")
    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(),os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(),os.O_BINARY)
    serve(sys.stdin.buffer,sys.stdout.buffer)


if __name__ == "__main__": main()
