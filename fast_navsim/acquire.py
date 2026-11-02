"""Bounded public-source downloads; compressed response bytes are budgeted.

Archives are consumed as streams, never downloaded in full. A retained member
hash verifies that local file only, not the rest of its parent archive.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


class ByteBudget:
    def __init__(self, limit, used=0):
        self.limit = int(limit)
        self.used = int(used)

    @property
    def remaining(self):
        return self.limit - self.used


class BudgetReader:
    def __init__(self, response, budget, archive_limit):
        self.response = response
        self.budget = budget
        self.archive_limit = int(archive_limit)
        self.used = 0
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        remaining = min(self.budget.remaining, self.archive_limit - self.used)
        if remaining <= 0:
            raise RuntimeError("response body byte budget exhausted")
        count = remaining if size < 0 else min(size, remaining)
        data = self.response.read(count)
        self.used += len(data)
        self.budget.used += len(data)
        self.sha.update(data)
        return data


class HTTPRangeReader:
    """Sequential bounded HTTP transport, tolerating short CDN connections."""
    def __init__(self,url,budget,archive_limit,chunk_size=1_000_000,
                 opener=urllib.request.urlopen):
        self.url,self.budget,self.archive_limit=url,budget,int(archive_limit)
        self.chunk_size,self.opener=chunk_size,opener
        self.buffer=b""
        self.response_body_bytes=0
        self.requests=0
        self.sha=hashlib.sha256()

    def read(self,size=-1):
        if size < 0:
            size=self.archive_limit-self.response_body_bytes+len(self.buffer)
        result=[]
        while size:
            if self.buffer:
                take=min(size,len(self.buffer))
                result.append(self.buffer[:take]);self.buffer=self.buffer[take:];size-=take
                continue
            remaining=min(self.archive_limit-self.response_body_bytes,self.budget.remaining)
            if remaining<=0:
                raise RuntimeError("response body byte budget exhausted")
            start=self.response_body_bytes
            end=start+min(self.chunk_size,remaining)-1
            request=urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{end}"})
            with self.opener(request,timeout=90) as response:
                self.requests+=1
                if response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {start}-"):
                    raise ValueError("Server did not honor the bounded Range request")
                reader=BudgetReader(response,self.budget,end-start+1)
                chunks=[]
                while reader.used<end-start+1:
                    chunk=reader.read(min(65536,end-start+1-reader.used))
                    if not chunk:break
                    chunks.append(chunk)
                self.buffer=b"".join(chunks)
            if not self.buffer:
                raise RuntimeError("Empty ranged response before archive bound")
            self.response_body_bytes+=len(self.buffer)
            self.sha.update(self.buffer)
        return b"".join(result)


def fetch_bytes(url, budget, limit=2_000_000):
    with urllib.request.urlopen(url, timeout=90) as response:
        reader = BudgetReader(response, budget, limit)
        chunks = []
        while True:
            chunk = reader.read(min(65536, limit - reader.used))
            if not chunk:
                break
            chunks.append(chunk)
            if reader.used >= limit:
                raise RuntimeError("small-response byte budget exhausted")
        return b"".join(chunks)


def extract_tar_prefix(response, destination, budget, select, max_members,
                       archive_limit, stop=None):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reader = BudgetReader(response, budget, archive_limit)
    retained = []
    scanned = 0
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            scanned += 1
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not member.isfile() or not select(member.name):
                continue
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            digest = hashlib.sha256()
            temporary=target.with_name(target.name+".partial")
            try:
                with temporary.open("wb") as output:
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                if temporary.stat().st_size!=member.size:
                    raise tarfile.ReadError("Incomplete selected archive member")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            retained.append({"member": member.name, "path": str(target),
                             "bytes": member.size, "sha256": digest.hexdigest()})
            if len(retained) >= max_members or (stop and stop(retained)):
                break
    return {"files": retained, "members_scanned": scanned,
            "response_body_bytes": reader.used,
            "response_prefix_sha256": reader.sha.hexdigest(),
            "full_archive_verified": False}


def stream_archive(url, destination, budget, select, max_members,
                   archive_limit, stop=None):
    with urllib.request.urlopen(url, timeout=90) as response:
        result = extract_tar_prefix(response, destination, budget, select,
                                    max_members, archive_limit, stop)
    result["url"] = url
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def download_prefix_ranges(url, destination, budget, prefix_bytes,
                           chunk_size=1_000_000, opener=urllib.request.urlopen,
                           resume=False):
    """Download only a fixed compressed prefix, resuming short CDN bodies.

    Range offsets and byte counts are checked. Every received body is budgeted,
    including rejected responses. No full-archive integrity claim is made.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    requests = 0
    digest = hashlib.sha256()
    offset = destination.stat().st_size if resume and destination.exists() else 0
    initial_offset = offset
    if offset > prefix_bytes:
        raise ValueError("Existing prefix is larger than requested bound")
    if offset:
        with destination.open("rb") as previous:
            for data in iter(lambda:previous.read(65536),b""):
                digest.update(data)
    with destination.open("ab" if offset else "wb") as output:
        while offset < prefix_bytes:
            end = min(offset+chunk_size, prefix_bytes)-1
            request = urllib.request.Request(url, headers={"Range":f"bytes={offset}-{end}"})
            with opener(request, timeout=90) as response:
                requests += 1
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server did not honor the bounded Range request")
                reader = BudgetReader(response,budget,end-offset+1)
                received = 0
                while received < end-offset+1:
                    data = reader.read(min(65536,end-offset+1-received))
                    if not data:
                        break
                    output.write(data)
                    digest.update(data)
                    received += len(data)
            if not received:
                raise RuntimeError("Empty ranged response before requested prefix end")
            offset += received
    return {"url":url, "path":str(destination.resolve()), "response_body_bytes":offset-initial_offset,
            "prefix_bytes":offset,
            "response_prefix_sha256":digest.hexdigest(),"requests":requests,
            "full_archive_verified":False}
