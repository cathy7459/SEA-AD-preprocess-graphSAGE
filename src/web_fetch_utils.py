from __future__ import annotations

import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


def build_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def fetch_html(session: requests.Session, url: str, timeout: int = 60) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_links(html: str, source_page: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    records: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if not href:
            continue
        if href.startswith("/"):
            href = requests.compat.urljoin(source_page, href)
        records.append({"title": text, "url": href, "source_page": source_page})
    return records


def stream_download(
    session: requests.Session,
    url: str,
    destination: str | Path,
    retries: int = 4,
    timeout: int = 60,
    chunk_bytes: int = 1024 * 1024,
) -> dict[str, str]:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".part")
    resume_pos = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout, headers=headers) as response:
                if response.status_code not in {200, 206}:
                    response.raise_for_status()
                mode = "ab" if resume_pos and response.status_code == 206 else "wb"
                with temp_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_bytes):
                        if chunk:
                            handle.write(chunk)
            temp_path.replace(dest)
            return {
                "status": "downloaded",
                "local_path": str(dest),
                "sha256": sha256sum(dest),
            }
        except Exception as exc:
            error = exc
            time.sleep(min(2**attempt, 10))
    return {"status": "failed", "local_path": str(dest), "error": str(error)}


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def head_metadata(session: requests.Session, url: str, timeout: int = 60) -> dict[str, str]:
    response = session.head(url, allow_redirects=True, timeout=timeout)
    return {
        "final_url": response.url,
        "content_type": response.headers.get("Content-Type", ""),
        "content_length": response.headers.get("Content-Length", ""),
        "etag": response.headers.get("ETag", ""),
    }


def choose_downloadable_links(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    downloadable_domains = ("cdn.prod.website-files.com", ".s3.amazonaws.com", "s3.amazonaws.com")
    chosen: list[dict[str, str]] = []
    for row in records:
        url = row["url"]
        lowered = url.lower()
        base_without_fragment = lowered.split("#", 1)[0]
        if ("index.html" in lowered) or lowered.endswith(".html") or base_without_fragment.endswith(".html"):
            continue
        if any(domain in url for domain in downloadable_domains) or lowered.endswith((".csv", ".xlsx", ".h5ad", ".zip", ".tsv", ".parquet")):
            chosen.append(row)
    return chosen


def list_public_s3_bucket(session: requests.Session, bucket_name: str, prefix: str | None = None, timeout: int = 60) -> list[dict[str, str]]:
    """List a public S3 bucket anonymously via ListObjectsV2 XML.

    The official SEA-AD AWS registry page exposes public bucket names and browse URLs.
    We use the bucket endpoint directly so the pipeline can build file manifests without
    requiring the AWS CLI. This is critical for restartable, web-first provenance.
    """
    continuation_token: str | None = None
    rows: list[dict[str, str]] = []
    base_url = f"https://{bucket_name}.s3.amazonaws.com"
    while True:
        params = {"list-type": "2"}
        if prefix:
            params["prefix"] = prefix
        if continuation_token:
            params["continuation-token"] = continuation_token
        response = session.get(base_url, params=params, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for item in root.findall("s3:Contents", namespace):
            key = item.findtext("s3:Key", default="", namespaces=namespace)
            size = item.findtext("s3:Size", default="0", namespaces=namespace)
            last_modified = item.findtext("s3:LastModified", default="", namespaces=namespace)
            if not key or key.endswith("/"):
                continue
            rows.append(
                {
                    "bucket_name": bucket_name,
                    "key": key,
                    "size_bytes": int(size),
                    "last_modified": last_modified,
                    "remote_url": f"{base_url}/{key}",
                }
            )
        is_truncated = root.findtext("s3:IsTruncated", default="false", namespaces=namespace).lower() == "true"
        continuation_token = root.findtext("s3:NextContinuationToken", default="", namespaces=namespace) or None
        if not is_truncated:
            break
    return rows
