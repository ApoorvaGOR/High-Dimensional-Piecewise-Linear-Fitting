"""Build a clean GitHub source ZIP; optionally include pinned offline data."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datasets import ROOT, SOURCES, acquire


def build(include_data=False):
    target=ROOT/'dist'/('slope-change-benchmark-offline.zip' if include_data else 'slope-change-benchmark.zip')
    target.parent.mkdir(parents=True,exist_ok=True)
    selected=set()
    for pattern in ['*.py','*.md','LICENSE','requirements*.txt','.gitignore']:
        selected.update(ROOT.glob(pattern))
    for folder in ['docs','figures','tests','.github']:
        selected.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    selected.update(p for p in (ROOT/'data').rglob('*') if p.is_file() and 'raw' not in p.relative_to(ROOT/'data').parts)
    selected.update(p for p in (ROOT/'validation').glob('*') if p.is_file())
    if include_data:
        selected.update(acquire(name) for name in SOURCES)
    selected={path for path in selected if path.name != '.DS_Store'}
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(selected):
            archive.write(path,Path('slope-change-benchmark')/path.relative_to(ROOT))
    print(f'{target} ({target.stat().st_size/1024**2:.2f} MiB)')
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-data',action='store_true')
    args=parser.parse_args(); build(args.include_data)
