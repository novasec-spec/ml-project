"""One-shot remote Ollama prompt utility using the same env config as client.py."""
import json, os, urllib.error, urllib.request
from pathlib import Path
CONFIG=Path.home()/".config/remote-ollama/env"
def chat(prompt,model=None):
    if CONFIG.exists():
        for line in CONFIG.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip().strip("'\""))
    url=os.getenv("OLLAMA_REMOTE_URL","").rstrip("/"); key=os.getenv("OLLAMA_REMOTE_API_KEY","")
    if not url or not key: raise RuntimeError(f"Missing remote Ollama settings in {CONFIG}")
    payload={"model":model or os.getenv("OLLAMA_MODEL","qwen3:8b"),"messages":[{"role":"user","content":prompt}],"stream":False,"options":{"num_ctx":int(os.getenv("OLLAMA_NUM_CTX","8192")),"num_predict":int(os.getenv("OLLAMA_NUM_PREDICT","1500")),"temperature":0.2}}
    req=urllib.request.Request(url+"/api/chat",data=json.dumps(payload).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},method="POST")
    try:
        with urllib.request.urlopen(req,timeout=int(os.getenv("OLLAMA_TIMEOUT","500"))) as r: data=json.load(r)
    except urllib.error.HTTPError as e: raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode(errors='replace')[:1500]}") from e
    except (urllib.error.URLError,TimeoutError,OSError) as e: raise RuntimeError(f"Remote Ollama request failed: {e}") from e
    return (data.get("message") or {}).get("content","")
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("prompt",nargs="?",default="Reply exactly: remote Ollama is reachable."); p.add_argument("--model"); a=p.parse_args()
    try: print(chat(a.prompt,a.model))
    except Exception as e: raise SystemExit("ERROR: "+str(e))
