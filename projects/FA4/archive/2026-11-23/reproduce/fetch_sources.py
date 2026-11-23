"""Fetch only the pinned public flash_attn source, never install host packages."""
import argparse,hashlib,json,tarfile,tempfile,urllib.request
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--revision',choices=['pr3109','pr3132'],required=True);p.add_argument('--destination',type=Path);a=p.parse_args();pins=json.loads((ROOT/'provenance/source-pins.json').read_text());sha=pins[a.revision.upper()];target=a.destination or ROOT/'_sources'/a.revision
 if target.exists():raise SystemExit('Refusing to overwrite existing destination: '+str(target))
 target.parent.mkdir(parents=True,exist_ok=True);url='https://codeload.github.com/Dao-AILab/flash-attention/tar.gz/'+sha;h=hashlib.sha256()
 with tempfile.TemporaryFile() as stream:
  with urllib.request.urlopen(url,timeout=120) as response:
   while data:=response.read(1024*1024):h.update(data);stream.write(data)
  stream.seek(0)
  with tarfile.open(fileobj=stream,mode='r:gz') as tar:
   for m in tar:
    parts=PurePosixPath(m.name).parts[1:]
    if not parts or '..' in parts or not m.isfile():continue
    if parts[0]!='flash_attn' and parts[0] not in ['LICENSE','LICENSE.txt']:continue
    dst=target.joinpath(*parts);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(tar.extractfile(m).read())
 if a.revision=='pr3109':
  manifest=json.loads((ROOT/'provenance/pr3109-source-manifest.json').read_text())
  for name,want in manifest['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==want,name
 (target/'source-provenance.json').write_text(json.dumps(dict(commit=sha,url=url,download_sha256=h.hexdigest()),indent=2));print(target/'flash_attn')
if __name__=='__main__':main()
