"""Explicitly download the official BEIR SciFact corpus and test judgments.

Run on the HOST from repository root (not from a hot-reload request handler).
"""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip'
ROOT = Path(os.getenv('AIRA_DATA_DIR', '.data')) / 'retrieval_eval' / 'datasets' / 'scifact'
ALLOW = ('corpus.jsonl', 'queries.jsonl', 'qrels/test.tsv')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    print('Downloading official BEIR SciFact archive:', URL)
    with urllib.request.urlopen(URL, timeout=120) as response:
        length = int(response.headers.get('Content-Length', 0))
        if length > 100_000_000:
            raise RuntimeError('Unexpectedly large archive')
        archive = response.read(100_000_001)
    if len(archive) > 100_000_000:
        raise RuntimeError('Archive exceeds safety limit')
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in ALLOW:
            candidates = [file for file in zipped.namelist()
                          if file == name or file == 'scifact/' + name]
            if len(candidates) != 1:
                raise ValueError(f'Missing or ambiguous {name} in official archive')
            member = zipped.getinfo(candidates[0])
            if member.file_size > 80_000_000:
                raise ValueError(f'Unexpected file size: {name}')
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            with zipped.open(member) as src, temporary.open('wb') as dst:
                while chunk := src.read(65536):
                    dst.write(chunk)
            os.replace(temporary, target)
            print(f'Installed {target}')
    print('Done. BEIR SciFact test dataset is available to AIRA.')


if __name__ == '__main__':
    main()
