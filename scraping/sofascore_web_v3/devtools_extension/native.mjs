export const VERSION = "0.4.1";
export const PROTOCOL = 1;
export const HOST = "com.xdiyo.sofascore";

// One port keeps one local Python writer alive for the entire campaign.
export class NativeClient {
  constructor(runtime, {timeoutMs = 60000, onDisconnect = () => {}} = {}) {
    this.runtime = runtime;
    this.timeoutMs = timeoutMs;
    this.onDisconnect = onDisconnect;
    this.pending = new Map();
    this.serial = 0;
  }
  async connect() {
    if (this.port) throw new Error("Already connected to Python");
    this.port = this.runtime.connectNative(HOST);
    const port=this.port;
    port.onMessage.addListener(message => {
      if(this.port!==port) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      clearTimeout(pending.timer);
      this.pending.delete(message.id);
      if (message.ok) pending.resolve(message.result);
      else pending.reject(new Error(message.error || "Python rejected the operation"));
    });
    port.onDisconnect.addListener(() => {
      if(this.port!==port) return;
      const reason = this.runtime.lastError?.message || "Python connection closed";
      this.port = null;
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer); pending.reject(new Error(reason));
      }
      this.pending.clear();
      this.onDisconnect(reason);
    });
    const hello = await this.call("hello", {protocol:PROTOCOL});
    if (hello.protocol !== PROTOCOL || hello.version !== VERSION) {
      this.close(); throw new Error("Extension/Python version mismatch. Reload the extension and reopen DevTools.");
    }
    return hello;
  }
  call(operation, args = {}) {
    if (!this.port) return Promise.reject(new Error("Python is disconnected"));
    const id = ++this.serial;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Python did not acknowledge ${operation}; capture paused. Reconnect to resume.`));
        this.close();
      }, this.timeoutMs);
      this.pending.set(id, {resolve, reject, timer});
      try { this.port.postMessage({id, operation, ...args}); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  close() {
    const port=this.port;
    if(!port) return;
    // Chrome need not notify the end that called disconnect(). Clear our local
    // connection immediately so Disconnect -> reconnect works in the same panel.
    this.port=null;
    const reason="Python connection closed";
    for(const pending of this.pending.values()) {
      clearTimeout(pending.timer);pending.reject(new Error(reason));
    }
    this.pending.clear();
    port.disconnect();
    this.onDisconnect(reason);
  }
}
