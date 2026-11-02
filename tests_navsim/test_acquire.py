import io
import tarfile

import pytest

from fast_navsim.acquire import ByteBudget, BudgetReader, extract_tar_prefix, download_prefix_ranges, HTTPRangeReader


def archive():
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode="w:gz") as tf:
        for name,payload in [("root/a.txt",b"alpha"),("root/b.txt",b"beta")]:
            member=tarfile.TarInfo(name);member.size=len(payload)
            tf.addfile(member,io.BytesIO(payload))
    return out.getvalue()


def test_budget_reader_hard_limits_all_response_bytes():
    budget=ByteBudget(5)
    reader=BudgetReader(io.BytesIO(b"abcdefgh"),budget,archive_limit=8)
    assert reader.read(3)==b"abc"
    assert reader.read(3)==b"de"
    with pytest.raises(RuntimeError, match="budget"):
        reader.read(1)
    assert budget.used==5


def test_extract_only_selected_member_from_stream_and_hash(tmp_path):
    payload=archive();budget=ByteBudget(10000)
    retained=extract_tar_prefix(io.BytesIO(payload),tmp_path,budget,
                                lambda name: name.endswith("a.txt"),max_members=1,
                                archive_limit=10000)
    assert len(retained["files"])==1
    assert (tmp_path/"root/a.txt").read_bytes()==b"alpha"
    assert not (tmp_path/"root/b.txt").exists()
    assert len(retained["files"][0]["sha256"])==64
    assert retained["full_archive_verified"] is False
    assert budget.used==len(payload)


def test_stream_rejects_unsafe_paths(tmp_path):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode="w:gz") as tf:
        item=tarfile.TarInfo("../outside");item.size=1;tf.addfile(item,io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_tar_prefix(io.BytesIO(out.getvalue()),tmp_path,ByteBudget(10000),lambda _:True,1,10000)


def test_ranges_resume_short_http_bodies_without_counting_local_reads(tmp_path):
    payload=b"abcdefghij"
    class Response(io.BytesIO):
        status=206
        def __init__(self,start,end):
            super().__init__(payload[start:min(end+1,start+2)])
            self.headers={"Content-Range":f"bytes {start}-{end}/{len(payload)}"}
    def opener(request,timeout):
        start,end=map(int,request.headers["Range"].split("=")[1].split("-"))
        return Response(start,end)
    budget=ByteBudget(20)
    report=download_prefix_ranges("https://example.test/pinned",tmp_path/"prefix",budget,8,chunk_size=4,opener=opener)
    assert (tmp_path/"prefix").read_bytes()==payload[:8]
    assert budget.used==8
    assert report["requests"]==4
    assert report["full_archive_verified"] is False


def test_ranges_can_extend_existing_prefix_without_refetch(tmp_path):
    payload=b"abcdefghij";path=tmp_path/"prefix";path.write_bytes(payload[:3])
    class Response(io.BytesIO):
        status=206
        def __init__(self,start,end):
            super().__init__(payload[start:end+1]);self.headers={"Content-Range":f"bytes {start}-{end}/10"}
    def opener(request,timeout):
        a,b=map(int,request.headers["Range"].split("=")[1].split("-"));assert a>=3
        return Response(a,b)
    budget=ByteBudget(10)
    report=download_prefix_ranges("https://example.test/pinned",path,budget,8,opener=opener,resume=True)
    assert path.read_bytes()==payload[:8]
    assert budget.used==5
    assert report["response_body_bytes"]==5
    assert report["prefix_bytes"]==8


def test_range_stream_buffers_and_resumes_short_bodies():
    payload=b"abcdefghij"
    class Response(io.BytesIO):
        status=206
        def __init__(self,start,end):
            super().__init__(payload[start:min(start+3,end+1)]);self.headers={"Content-Range":f"bytes {start}-{end}/10"}
    def opener(request,timeout):
        a,b=map(int,request.headers["Range"].split("=")[1].split("-"));return Response(a,b)
    budget=ByteBudget(10)
    stream=HTTPRangeReader("https://example.test/pinned",budget,8,chunk_size=4,opener=opener)
    assert stream.read(2)==b"ab"
    assert stream.read(4)==b"cdef"
    assert stream.read(2)==b"gh"
    assert budget.used==8
    assert stream.response_body_bytes==8


def test_incomplete_selected_member_is_never_published(tmp_path):
    import random
    out=io.BytesIO();data=random.Random(4).randbytes(100_000)
    with tarfile.open(fileobj=out,mode="w:gz") as tf:
        member=tarfile.TarInfo("root/a.txt");member.size=len(data);tf.addfile(member,io.BytesIO(data))
    payload=out.getvalue()[:2000]
    with pytest.raises((EOFError,tarfile.ReadError,RuntimeError)):
        extract_tar_prefix(io.BytesIO(payload),tmp_path,ByteBudget(10000),lambda _:True,1,10000)
    assert not (tmp_path/"root/a.txt").exists()
