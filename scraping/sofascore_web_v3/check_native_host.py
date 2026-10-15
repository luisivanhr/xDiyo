"""Verify the compiled local writer responds while its input pipe stays open."""
import json
import os
import queue
import subprocess
import threading

from .native_host import HOST_DIRECTORY, HOST_NAME, read_message, write_message
from .campaign import read_json


def check():
    manifest=read_json(HOST_DIRECTORY/f"{HOST_NAME}.json")
    process=subprocess.Popen([manifest["path"],manifest["allowed_origins"][0]],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    results=queue.Queue()
    def receive():
        try:
            while True:
                message=read_message(process.stdout)
                if message is None: break
                results.put(message)
        except Exception as error: results.put(error)
    reader=threading.Thread(target=receive,daemon=True)
    reader.start()
    try:
        for message_id in (1,2):
            write_message(process.stdin,{"id":message_id,"operation":"hello","protocol":1})
            try: response=results.get(timeout=5)
            except queue.Empty as error: raise RuntimeError("Writer did not respond while the native port remained open") from error
            expected={"id":message_id,"ok":True,"result":{"protocol":1,"version":"0.2.21","host":HOST_NAME}}
            if response != expected: raise RuntimeError(f"Unexpected native response: {response}")
        process.stdin.close()
        code=process.wait(timeout=5)
        if code or process.stderr.read(): raise RuntimeError("Writer did not close cleanly")
    finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdout.close(); process.stderr.close()
        if not process.stdin.closed: process.stdin.close()
    return {"compiled_host":"passed","persistent_port_roundtrips":2,"version":"0.2.21",
            "scope":"Local process only; Chrome's installed extension must also be connected separately."}


if __name__ == "__main__": print(json.dumps(check(),indent=2))
