"""
全環境・全手法で bandwidth distribution を生成するスクリプト

出力先は results/bandwidth_distribution_new/ に統一
（既存のファイルを上書きしない）
"""

import subprocess
import sys
from pathlib import Path

# 実行対象の組み合わせ（画像の形式に合わせて1つの手法のみ）
METHODS = [
    "proposed",
]

ENVIRONMENTS = [
    "static",
    "node_switching",
    "bandwidth_fluctuation",
]

OPT_TYPE = "bandwidth_only"

# 実行パラメータ
GENERATIONS = 1000
ANTS = 10
SIMULATIONS = 100

# 出力先ディレクトリ
OUTPUT_BASE = Path("aco_moo_routing/results/bandwidth_distribution_new")
RESULTS_DIR = Path("aco_moo_routing/results")

# スクリプトパス
SCRIPT_PATH = Path("aco_moo_routing/analysis/plot_bandwidth_distribution.py")


def run_plot(method: str, environment: str, bin_mode: str = "relative"):
    """bandwidth distribution を生成"""
    # CSVファイルのパスを取得（存在確認用）
    csv_path = RESULTS_DIR / method / environment / OPT_TYPE / "ant_solution_log.csv"

    if not csv_path.exists():
        print(f"⚠️ CSVファイルが見つかりません: {csv_path}")
        return False

    # 出力先ディレクトリ（全環境同じ場所）
    output_dir = OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    # ファイル名に環境名を含める（手法名も含める）
    base_filename = (
        "relative_bandwidth_distribution"
        if bin_mode == "relative"
        else "bandwidth_distribution"
    )
    output_filename = f"{base_filename}_{environment}"
    output_path = output_dir / output_filename

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--method",
        method,
        "--environment",
        environment,
        "--opt-type",
        OPT_TYPE,
        "--generations",
        str(GENERATIONS),
        "--ants",
        str(ANTS),
        "--simulations",
        str(SIMULATIONS),
        "--bin-mode",
        bin_mode,
        "--results-dir",
        str(RESULTS_DIR),
        "--output",
        str(output_path),
    ]

    print(f"\n{'='*80}")
    print(f"実行中: {method} / {environment} ({bin_mode} mode)")
    print(f"出力先: {output_path}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("⚠️ 警告:", result.stderr)
        print(f"✅ 完了: {method} / {environment} ({bin_mode} mode)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ エラー: {method} / {environment} ({bin_mode} mode)")
        print(f"   コマンド: {' '.join(cmd)}")
        print(f"   エラー出力: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ CSVファイルが見つかりません: {method} / {environment}")
        return False


def main():
    """全組み合わせを実行"""
    print(f"\n{'='*80}")
    print("Bandwidth Distribution 生成スクリプト")
    print(f"{'='*80}")
    print(f"出力先: {OUTPUT_BASE}")
    print(f"世代数: {GENERATIONS}, アリ数: {ANTS}, シミュレーション数: {SIMULATIONS}")
    print(f"{'='*80}\n")

    total = len(METHODS) * len(ENVIRONMENTS)  # relative mode のみ
    success = 0
    failed = 0

    for method in METHODS:
        for environment in ENVIRONMENTS:
            # relative mode のみ実行
            if run_plot(method, environment, bin_mode="relative"):
                success += 1
            else:
                failed += 1

    print(f"\n{'='*80}")
    print("実行結果サマリー")
    print(f"{'='*80}")
    print(f"成功: {success} / {total}")
    print(f"失敗: {failed} / {total}")
    print(f"出力先: {OUTPUT_BASE}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
