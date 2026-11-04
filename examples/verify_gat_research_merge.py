"""Run isolated pinned-main/candidate comparisons without modifying either source."""
from pathlib import Path
import argparse, os, subprocess, sys, json
import torch


def exact(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=path)
    elif isinstance(a, dict):
        assert list(a) == list(b), (path, "keys", list(a), list(b))
        for k in a: exact(a[k], b[k], path+"."+str(k))
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x,y) in enumerate(zip(a,b)): exact(x,y,path+"."+str(i))
    else: assert a == b, (path,a,b)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--kinds", nargs="+", default=["default","released_statistics","gat_memory","gat_statistics","gat_combined","cnn"])
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1",
             XDIYO_PARITY_DEVICE=args.device)
    receipt=json.loads((args.output/"receipt.json").read_text()) if (args.output/"receipt.json").exists() else {}
    for kind in args.kinds:
        paths=[]
        for label,source in (("main",args.reference),("candidate",root)):
            out=args.output/kind/label
            env["PYTHONPATH"]=str(source/"src")
            cmd=[sys.executable,str(root/"tests/analytics/gat_reconciliation_worker.py"),str(source),str(out),kind]
            result=subprocess.run(cmd,env=env,cwd=root,capture_output=True,text=True)
            out.mkdir(parents=True,exist_ok=True)
            (out/"worker.log").write_text(result.stdout+result.stderr,encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{label} {kind} worker failed: {out / 'worker.log'}")
            paths.append(out)
        a,b=paths
        exact(torch.load(a/"trajectory.pt",weights_only=False),torch.load(b/"trajectory.pt",weights_only=False))
        exact(json.loads((a/"outputs.json").read_text()),json.loads((b/"outputs.json").read_text()))
        receipt[kind]={"passed":True,"device":args.device,"updates":3,"tolerance":"exact"}
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"receipt.json").write_text(json.dumps(receipt,indent=2))
        print(kind+": exact three-update and output parity",flush=True)


if __name__=="__main__": main()
