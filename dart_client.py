# dart_client.py
from __future__ import annotations
import os, io, zipfile, time
import xmltodict, requests

DART_BASE = "https://opendart.fss.or.kr/api"
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _api_key() -> str | None:
    return os.environ.get("DART_API_KEY") or os.environ.get("OPENDART_API_KEY")

def _download_corpcode_zip(api_key: str) -> str | None:
    url = f"{DART_BASE}/corpCode.xml?crtfc_key={api_key}"
    path = os.path.join(CACHE_DIR, "corpCode.xml")
    # OpenDART returns zip stream but name is .xml in doc; handle both
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        # Try unzip first
        try:
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            with z.open(z.namelist()[0]) as f:
                data = f.read()
            with open(path, "wb") as w:
                w.write(data)
        except zipfile.BadZipFile:
            # Maybe it's plain xml (rare)
            with open(path, "wb") as w:
                w.write(resp.content)
        return path
    except Exception:
        return None

def _ensure_corpcode_xml(api_key: str) -> str | None:
    path = os.path.join(CACHE_DIR, "corpCode.xml")
    if not os.path.exists(path):
        return _download_corpcode_zip(api_key)
    # refresh monthly
    if time.time() - os.path.getmtime(path) > 30*24*3600:
        return _download_corpcode_zip(api_key)
    return path

def _load_corp_map(xml_path: str) -> dict[str, str]:
    with open(xml_path, "rb") as f:
        doc = xmltodict.parse(f.read())
    rows = doc.get("result", {}).get("list", [])
    out = {}
    for r in rows:
        name = str(r.get("corp_name", "")).strip()
        code = str(r.get("corp_code", "")).strip()
        if name and code:
            out[name] = code
    return out

def resolve_corp_code(corp_name: str) -> str | None:
    api_key = _api_key()
    if not api_key:
        return None
    xml_path = _ensure_corpcode_xml(api_key)
    if not xml_path:
        return None
    cmap = _load_corp_map(xml_path)
    return cmap.get(corp_name)

def latest_filings(corp_name: str, max_items: int = 3) -> list[dict]:
    """
    corp_name 예: '삼성전자', 'SG글로벌'
    """
    api_key = _api_key()
    if not api_key:
        return []
    code = resolve_corp_code(corp_name)
    if not code:
        return []
    try:
        url = f"{DART_BASE}/list.json?crtfc_key={api_key}&corp_code={code}&page_no=1&page_count={max_items}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        items = js.get("list", []) if isinstance(js, dict) else []
        out = []
        for it in items[:max_items]:
            out.append({
                "rpt": it.get("report_nm", ""),
                "rcpno": it.get("rcept_no", ""),
                "rcpdt": it.get("rcept_dt", ""),
                "link": f"https://dart.fss.go.kr/dsaf001/main.do?rcpNo={it.get('rcept_no','')}"
            })
        return out
    except Exception:
        return []
