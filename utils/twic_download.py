import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import requests

TWIC_ZIP_URL = "https://theweekinchess.com/zips/twic{n}g.zip"  # url pattern  [oai_citation:1‡Gist](https://gist.github.com/roryk/ba72531b3b0f423dc8b9bd69beb63f8e?utm_source=chatgpt.com)


def download_file(url: str, dst_path: str, chunk_size: int = 1 << 20) -> None:
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

def download_twic_zips(twic_numbers: Iterable[int], out_dir: str) -> List[str]:
    """
    Downloads twicXXXXg.zip files into out_dir. Returns list of zip paths.
    """
    out = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for n in twic_numbers:
        url = TWIC_ZIP_URL.format(n=n)
        zip_path = os.path.join(out_dir, f"twic{n}g.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading {url}")
            download_file(url, zip_path)
        else:
            print(f"Already have {zip_path}")
        out.append(zip_path)
    return out

def build_merged_pgn_from_zips(zip_paths: List[str], out_pgn_path: str) -> None:
    """
    Extracts all .pgn files from the zips and concatenates into one .pgn.
    """
    Path(out_pgn_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pgn_path, "wb") as out_f:
        for zp in zip_paths:
            with zipfile.ZipFile(zp, "r") as z:
                pgn_names = [n for n in z.namelist() if n.lower().endswith(".pgn")]
                if not pgn_names:
                    raise RuntimeError(f"No PGN inside {zp}")

                # TWIC zips usually contain a single pgn; if multiple, concatenate all
                for name in sorted(pgn_names):
                    data = z.read(name)
                    # Ensure separation between files
                    if not data.endswith(b"\n"):
                        data += b"\n"
                    out_f.write(data)
    print(f"Wrote merged PGN to {out_pgn_path}")

def twic_range(start: int, end: int) -> List[int]:
    return list(range(start, end + 1))