// Local extension-to-extension handshake. No website or native requests here.
try {
  const tab=await chrome.tabs.getCurrent();
  const nonce=location.hash.slice(1);
  if (!Number.isInteger(tab?.id) || !nonce) throw new Error("No active reset session");
  const reply=await chrome.runtime.sendMessage({type:"xdiyo_memory_reset_ready",tabId:tab.id,nonce});
  if (reply?.ok !== true) throw new Error("Collector did not acknowledge the reset page");
} catch (error) {
  document.getElementById("status").textContent=`Saved matches are safe. ${error.message}. Follow the resume instructions below.`;
}
