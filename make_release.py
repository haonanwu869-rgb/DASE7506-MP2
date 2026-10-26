"""Refresh file checksums and create a portable zip beside this directory."""
from pathlib import Path
import hashlib, json, zipfile

root = Path(__file__).resolve().parent
files = sorted(p for p in root.rglob('*') if p.is_file()
               and '.git' not in p.parts and '__pycache__' not in p.parts and '.venv' not in p.parts
               and p.name != 'FILES_SHA256.json')
hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest = root/'FILES_SHA256.json'
manifest.write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
archive = root.parent/'final_submission.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in files+[manifest]:
        z.write(p, (Path(root.name)/p.relative_to(root)).as_posix())
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files)+1}, indent=2))
