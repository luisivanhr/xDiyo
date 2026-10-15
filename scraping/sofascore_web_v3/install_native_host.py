"""Build/review the local writer; registration is a separate explicit command."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .campaign import ROOT, read_json
from .native_host import HOST_NAME, HOST_DIRECTORY
from scraping.sofascore.store import atomic_write, json_bytes


def prepare(extension_id, *, directory=HOST_DIRECTORY):
    if not re.fullmatch(r"[a-p]{32}",extension_id): raise ValueError("Invalid Chrome extension ID")
    if os.name != "nt": raise ValueError("This installer currently targets Windows")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    python = ROOT/".venv/Scripts/python.exe"
    if not python.is_file(): raise ValueError("The project Python environment is missing")
    # The wrapper has no shell and starts Python with a hidden child process.
    source = '''using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
class NativeHost {
  static void Pump(Stream input, Stream output) {
    var buffer = new byte[65536];
    int count;
    while ((count = input.Read(buffer, 0, buffer.Length)) > 0) {
      output.Write(buffer, 0, count);
      output.Flush();
    }
  }
  static int Main(string[] args) {
    try { return Run(args); }
    catch (Exception error) { Console.Error.WriteLine(error.Message); return 1; }
  }
  static int Run(string[] args) {
    if (args.Length < 1 || args[0] != @@ORIGIN@@) return 2;
    var start = new ProcessStartInfo(@@PYTHON@@, "-B -u -m scraping.sofascore_web_v3.native_host " + @@ORIGIN@@);
    start.WorkingDirectory = @@PROJECT@@;
    start.UseShellExecute = false;
    start.CreateNoWindow = true;
    start.RedirectStandardInput = true;
    start.RedirectStandardOutput = true;
    start.RedirectStandardError = true;
    using (var process = Process.Start(start)) {
      Task.Factory.StartNew(() => {
        try { Pump(Console.OpenStandardInput(), process.StandardInput.BaseStream); }
        finally { process.StandardInput.Close(); }
      });
      Task.Factory.StartNew(() => Pump(process.StandardError.BaseStream, Console.OpenStandardError()));
      Pump(process.StandardOutput.BaseStream, Console.OpenStandardOutput());
      process.WaitForExit();
      return process.ExitCode;
    }
  }
}
'''.replace("@@ORIGIN@@",json.dumps(f"chrome-extension://{extension_id}/")).replace("@@PYTHON@@",json.dumps(str(python))).replace("@@PROJECT@@",json.dumps(str(ROOT)))
    source_path = directory/"NativeHost.cs"
    atomic_write(source_path,source.encode("utf-8"))
    compiler = Path(os.environ.get("WINDIR",r"C:\Windows"))/"Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not compiler.is_file(): raise ValueError("Windows .NET Framework compiler was not found")
    executable = directory/"xdiyo-native-host.exe"
    subprocess.run([str(compiler),"/nologo","/target:exe",f"/out:{executable}",str(source_path)],check=True,capture_output=True,text=True)
    manifest = {"name":HOST_NAME,"description":"xDiyo local Sofascore capture writer","path":str(executable),
                "type":"stdio","allowed_origins":[f"chrome-extension://{extension_id}/"]}
    path = directory/f"{HOST_NAME}.json"
    atomic_write(path,json_bytes(manifest))
    return {"manifest":str(path),"executable":str(executable),"source":str(source_path),
            "registry_key":f"HKEY_CURRENT_USER\\Software\\Google\\Chrome\\NativeMessagingHosts\\{HOST_NAME}","registered":False}


def register():
    if os.name != "nt": raise ValueError("Windows registration is required")
    import winreg
    path = HOST_DIRECTORY/f"{HOST_NAME}.json"
    manifest = read_json(path)
    if not Path(manifest["path"]).is_file(): raise ValueError("Prepare the native host first")
    key_path = rf"Software\Google\Chrome\NativeMessagingHosts\{HOST_NAME}"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key_path) as key:
            previous = winreg.QueryValueEx(key,"")[0]
        if Path(previous).resolve() != path.resolve():
            raise ValueError("This native host name is already registered to another location")
    except FileNotFoundError:
        pass
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER,key_path) as key:
        winreg.SetValueEx(key,"",0,winreg.REG_SZ,str(path.resolve()))
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key_path) as key:
        actual = winreg.QueryValueEx(key,"")[0]
    if actual != str(path.resolve()): raise ValueError("Native host registration readback mismatch")
    return {"registered":True,"manifest":actual,"allowed_origins":manifest["allowed_origins"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-id")
    parser.add_argument("--register",action="store_true")
    args = parser.parse_args()
    result = register() if args.register else prepare(args.extension_id or "")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
