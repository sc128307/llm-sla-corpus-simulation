import os
import re
import zipfile
import pandas as pd
from .paths import HUMAN_PATH

# Canonical human baseline. Everything that needs the human essays should load
# through load_human_baseline() or this constant.
HUMAN_BASELINE_PATH = str(HUMAN_PATH)


def load_human_baseline(path: str = None) -> pd.DataFrame:
    """Load the canonical human baseline with a stable schema.

    Guarantees the columns ``id`` and ``region`` exist (accepting the
    alternate input names ``text_id`` and ``nationality`` when needed),
    so callers never have to branch on column names again.

    Args:
        path: optional override; defaults to corpus/human/human_data_full.csv
    """
    csv_path = path or HUMAN_BASELINE_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Human baseline not found at {csv_path}. Rebuild it with "
            f"`python -m src.data_loader --save` (reads the ICNALE zip)."
        )
    df = pd.read_csv(csv_path)
    if "id" not in df.columns and "text_id" in df.columns:
        df = df.rename(columns={"text_id": "id"})
    if "region" not in df.columns and "nationality" in df.columns:
        df = df.rename(columns={"nationality": "region"})
    # The canonical file carries a leading BOM (\ufeff) inside every text
    # field (utf_8_sig write artifact); strip it so string comparisons and
    # downstream cleaning are exact.
    if "text" in df.columns:
        df["text"] = df["text"].astype(str).str.replace("\ufeff", "",
                                                        regex=False)
    return df


class ICNALELoader:
    """
    Loader for ICNALE_WE (Written Essays), covering the 11 target
    regions/countries. Reads either an extracted directory or — if the
    extracted folder is absent — directly from the zip archive
    (``corpus/raw/ICNALE_WE.zip``), so the raw data never needs to be
    unpacked on disk.
    """

    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        # Regex parse: WE_CHN_PTJ0_001_B1_1.txt
        self.filename_pattern = re.compile(
            r"WE_([A-Z]{3})_([A-Z0-9]+)_(\d+)_([A-Z0-9_]+)\.txt"
        )

    # --- ICNALE region codes (canonical 'region' column values) ---
    @property
    def target_countries(self):
        return [
            "CHN",
            "TWN",
            "HKG",  # Greater China
            "JPN",
            "KOR",  # East Asian neighbours
            "ENS",  # native-speaker baseline
            "PAK",  # South Asia
            "THA",
            "IDN",  # Southeast Asia (EFL)
            "SIN",
            "PHL",  # Southeast Asia (ESL/Outer Circle)
        ]

    def load_data(self) -> pd.DataFrame:
        print(f"🚀 Scanning: {self.raw_path}")
        print(f"🎯 Target coverage: {len(self.target_countries)} regions")

        data_list = []

        # --- Prefer an extracted directory; else read from the zip ---
        if os.path.isdir(self.raw_path):
            for root, dirs, files in os.walk(self.raw_path):
                for file in files:
                    if not file.endswith(".txt"):
                        continue
                    match = self.filename_pattern.match(file)
                    if match:
                        full_path = os.path.join(root, file)
                        with open(full_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                        if text:
                            data_list.append(self._row_from_match(match, text))
        else:
            zip_candidates = [
                self.raw_path + ".zip",
                os.path.join(
                    os.path.dirname(self.raw_path), "ICNALE_WE.zip"
                ),
            ]
            zip_file = next((z for z in zip_candidates if os.path.exists(z)),
                            None)
            if zip_file is None:
                raise FileNotFoundError(
                    f"No extracted dir ({self.raw_path}) and no zip archive "
                    f"found among: {zip_candidates}"
                )
            print(f"📦 Reading from zip: {zip_file}")
            with zipfile.ZipFile(zip_file) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = os.path.basename(info.filename)
                    if not name.endswith(".txt"):
                        continue
                    match = self.filename_pattern.match(name)
                    if not match:
                        continue
                    text = zf.read(info).decode("utf-8-sig", errors="replace")
                    text = text.strip()
                    if text:
                        data_list.append(self._row_from_match(match, text))

        df = pd.DataFrame(data_list)

        # --- Summary report ---
        print("\n" + "=" * 40)
        print(f"✅ Data loaded! Total samples: {len(df)}")
        print("=" * 40)
        print(f"{'Region':<15} | {'Count':<10}")
        print("-" * 30)
        for region, count in df["region"].value_counts().items():
            print(f"{region:<15} | {count:<10}")
        print("=" * 40)

        return df

    def _row_from_match(self, match, text: str) -> dict:
        country, topic_code, student_id, proficiency = match.groups()
        return {
            "id": f"{country}_{student_id}",
            "region": country,          # canonical column name (was 'nationality')
            "topic": self._map_topic(topic_code),
            "proficiency": proficiency,  # e.g. A2_0, B1_1, B2_2
            "text": self._clean_text(text),
            "file_name": match.group(0),
        }

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _map_topic(self, code: str) -> str:
        if "PTJ" in code:
            return "Part-time Job"
        if "SMK" in code:
            return "Smoking Ban"
        return code


if __name__ == "__main__":
    import sys

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "corpus", "raw", "ICNALE_WE")
    canonical_path = str(HUMAN_PATH)

    loader = ICNALELoader(raw_path)
    df = loader.load_data()
    print(f"\nLoaded {len(df)} essays from the ICNALE_WE archive.")

    if "--save" in sys.argv:
        os.makedirs(os.path.dirname(canonical_path), exist_ok=True)
        df.to_csv(canonical_path, index=False, encoding="utf_8_sig")
        print(f"💾 Saved canonical baseline -> {canonical_path}")
    else:
        # Default: verify-only. NEVER overwrites the canonical baseline.
        if os.path.exists(canonical_path):
            current = load_human_baseline(canonical_path)
            n_cur = len(current)
            same_regions = (
                current["region"].value_counts().sort_index().to_dict()
                == df["region"].value_counts().sort_index().to_dict()
            )
            print(f"Canonical file rows      : {n_cur}")
            print(f"Archive rows             : {len(df)}")
            print(f"Per-region counts match  : {same_regions}")
            status = "OK" if (len(df) == n_cur and same_regions) else "MISMATCH"
            print(f"Verify result            : {status}")
            if status != "OK":
                print("Run `python -m src.data_loader --save` after inspecting "
                      "the difference.")
        else:
            print(f"No canonical file at {canonical_path} — run with --save "
                  f"to create it.")
