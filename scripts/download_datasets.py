#!/usr/bin/env python3
"""
データセット自動ダウンロードスクリプト

Hugging Face Hubから必要なコーパスをダウンロードします。
- LibriTTS (train-other-500サブセット)
- VCTK
- JVS
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import load_dataset
import shutil


def download_libritts(data_dir: Path, subset: str = "train-other-500"):
    """
    LibriTTSをOpenSLRからダウンロード

    Args:
        data_dir: 保存先ディレクトリ (data/libritts/)
        subset: ダウンロードするサブセット名
    """
    print(f"\n=== Downloading LibriTTS ({subset}) ===")

    import urllib.request
    import tarfile

    # OpenSLRのURL
    base_url = "https://www.openslr.org/resources/60"
    tar_filename = f"{subset}.tar.gz"
    tar_url = f"{base_url}/{tar_filename}"

    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / tar_filename
    output_dir = data_dir / subset

    # 既に展開済みの場合はスキップ
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Already exists: {output_dir}")
        return

    # tarファイルをダウンロード
    if not tar_path.exists():
        print(f"Downloading from {tar_url}...")
        print("(This may take a while, the file is several GB)")
        urllib.request.urlretrieve(tar_url, tar_path)
        print(f"✓ Downloaded to {tar_path}")
    else:
        print(f"Using cached file: {tar_path}")

    # 展開
    print(f"Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    print(f"✓ LibriTTS extracted to {output_dir}")

    # tarファイルを削除してスペースを節約(オプション)
    # tar_path.unlink()
    # print(f"Removed temporary file: {tar_filename}")


def download_vctk(data_dir: Path):
    """
    VCTKを公式ソースからダウンロード

    Args:
        data_dir: 保存先ディレクトリ (data/vctk/)
    """
    print("\n=== Downloading VCTK ===")

    import urllib.request
    import zipfile

    # 公式のダウンロードURL (DataShare)
    # Note: DataShareからは直接ダウンロードできない場合があるため、代替URLを使用
    vctk_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    zip_filename = "VCTK-Corpus-0.92.zip"

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / zip_filename

    # 既に展開済みの場合はスキップ
    if (data_dir / "wav48_silence_trimmed").exists():
        print(f"✓ Already exists: {data_dir / 'wav48_silence_trimmed'}")
        return

    # zipファイルをダウンロード
    if not zip_path.exists():
        print(f"Downloading from {vctk_url}...")
        print("(This may take a while, the file is ~10GB)")
        try:
            urllib.request.urlretrieve(vctk_url, zip_path)
            print(f"✓ Downloaded to {zip_path}")
        except Exception as e:
            print(f"⚠ Download failed: {e}")
            print("  代替: 以下から手動ダウンロードしてください:")
            print("  https://datashare.ed.ac.uk/handle/10283/3443")
            print(f"  ダウンロード後、{data_dir}に配置してください")
            return
    else:
        print(f"Using cached file: {zip_path}")

    # 展開
    print(f"Extracting to {data_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"✓ VCTK extracted to {data_dir}")

    # zipファイルを削除してスペースを節約(オプション)
    # zip_path.unlink()
    # print(f"Removed temporary file: {zip_filename}")


def download_jvs(data_dir: Path):
    """
    JVS Corpusのダウンロード情報を表示

    Args:
        data_dir: 保存先ディレクトリ (data/jvs/)
    """
    print("\n=== JVS Corpus ===")

    data_dir.mkdir(parents=True, exist_ok=True)

    # JVS corpusが既に存在するかチェック
    if (data_dir / "jvs_ver1").exists() or any(data_dir.glob("jvs*")):
        print(f"✓ JVS Corpus already exists in {data_dir}")
        return

    print("⚠ JVS Corpusは自動ダウンロードできません")
    print("\n【手動ダウンロード手順】")
    print("1. 以下のURLにアクセスしてください:")
    print("   https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus")
    print("\n2. Google Driveから jvs_ver1.zip (3.5GB) をダウンロード")
    print(f"\n3. 以下のディレクトリに展開してください:")
    print(f"   {data_dir.absolute()}")
    print("\n4. 展開後のディレクトリ構造:")
    print(f"   {data_dir.absolute()}/jvs_ver1/")
    print("   ├── jvs001/")
    print("   ├── jvs002/")
    print("   └── ...")
    print("\nコーパス情報:")
    print("- 100話者 (男性50名、女性50名)")
    print("- 30時間の音声データ (22時間は並列発話)")
    print("- 24 kHz サンプリングレート")
    print("- 通常発話、囁き声、裏声の3スタイル")


def download_hubert_models(cache_dir: Path):
    """
    HuBERT-BASEモデルとK-meansモデルをダウンロード

    Args:
        cache_dir: キャッシュディレクトリ (data/cache/hubert/)
    """
    print("\n=== Downloading HuBERT models ===")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # HuBERT-BASE checkpoint from fairseq
    print("Downloading hubert_base_ls960.pt...")
    import urllib.request
    hubert_url = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
    hubert_path = cache_dir / "hubert_base_ls960.pt"

    if not hubert_path.exists():
        urllib.request.urlretrieve(hubert_url, hubert_path)
        print(f"✓ Saved to {hubert_path}")
    else:
        print(f"✓ Already exists: {hubert_path}")

    # K-means model (km100)
    print("Downloading km.bin (km100 quantizer)...")
    km_url = "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin"
    km_path = cache_dir / "km100.bin"

    if not km_path.exists():
        urllib.request.urlretrieve(km_url, km_path)
        print(f"✓ Saved to {km_path}")
    else:
        print(f"✓ Already exists: {km_path}")

    print(f"\n✓ HuBERT models saved to {cache_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for StreamVC")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for datasets (default: data/)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["libritts", "vctk", "jvs", "hubert", "all"],
        default=["all"],
        help="Which datasets to download (default: all)"
    )
    parser.add_argument(
        "--libritts-subset",
        type=str,
        default="train-other-500",
        help="LibriTTS subset to download (default: train-other-500)"
    )

    args = parser.parse_args()

    # Create base data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)

    download_all = "all" in args.datasets

    # Download HuBERT models first (needed for preprocessing)
    if download_all or "hubert" in args.datasets:
        download_hubert_models(args.data_dir / "cache" / "hubert")

    # Download datasets
    if download_all or "libritts" in args.datasets:
        download_libritts(args.data_dir / "libritts", args.libritts_subset)

    if download_all or "vctk" in args.datasets:
        download_vctk(args.data_dir / "vctk")

    if download_all or "jvs" in args.datasets:
        download_jvs(args.data_dir / "jvs")

    print("\n" + "="*50)
    print("✓ Download complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Generate metadata.jsonl for each dataset")
    print("2. Run preprocessing to generate feature caches")
    print("3. Start training with: python scripts/train.py")


if __name__ == "__main__":
    main()
