"""Interactive, bounded experiment. Commands operate on observed UI elements."""
import argparse
import json
from pathlib import Path
import queue
import threading

from .capture import WebsiteCapture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--driver-path",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--max-api-requests",type=int,default=80)
    parser.add_argument("--max-seconds",type=int,default=600)
    parser.add_argument("--manual-challenge-seconds",type=int,default=0,
                        help="Bounded homepage observation while you solve a challenge manually")
    args=parser.parse_args()
    reference=json.loads(args.reference.read_text(encoding="utf-8"))
    capture=WebsiteCapture(reference,output_root=args.output,driver_path=args.driver_path,
                           max_api_requests=args.max_api_requests,max_seconds=args.max_seconds)
    commands=queue.Queue()

    def read_commands():
        while True:
            try:
                line=input()
            except EOFError:
                commands.put({"command":"finish"})
                return
            try:
                commands.put(json.loads(line))
            except ValueError:
                print('Use JSON: {"command":"snapshot"}, {"command":"click","index":0}, or {"command":"finish"}',flush=True)

    try:
        capture.start()
        if args.manual_challenge_seconds:
            capture.open_homepage()
            print("Complete any challenge manually in Chrome during the warmup window.",flush=True)
            capture.observe(args.manual_challenge_seconds)
        if capture.stop_reason:
            print(json.dumps({"final":capture.finish()}),flush=True)
            return
        capture.open_match()
        capture.observe(6)
        print(json.dumps({"snapshot":capture.snapshot(),"report":capture.report()}),flush=True)
        print("READY: snapshot / click / report / finish. CAPTCHA is completed manually in the browser.",flush=True)
        threading.Thread(target=read_commands,daemon=True).start()
        while not capture.stop_reason:
            capture.poll()
            try:
                command=commands.get(timeout=.25)
            except queue.Empty:
                continue
            action=command.get("command")
            if action=="finish":
                break
            if action=="snapshot":
                print(json.dumps(capture.snapshot()),flush=True)
            elif action=="click":
                capture.click(int(command["index"]))
                capture.observe(4)
                print(json.dumps({"snapshot":capture.snapshot(),"report":capture.report()}),flush=True)
            elif action=="report":
                print(json.dumps(capture.report()),flush=True)
        print(json.dumps({"final":capture.finish()}),flush=True)
    finally:
        capture.close()


if __name__=="__main__":
    main()
