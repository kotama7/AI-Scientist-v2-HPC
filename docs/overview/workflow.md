# ワークフロー概要 (Workflow Overview)

このドキュメントでは、HPC-AutoResearchの全体ワークフローを説明します。
アイデア生成から論文レビューまでの完全なフローを理解することができます。

## 全体フロー図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HPC-AutoResearch ワークフロー                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                        │
│  │ 1. アイデア準備   │  perform_ideation_temp_free.py (オプション)            │
│  │    (Ideation)    │  ワークショップ記述 → アイデアJSON                      │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 2. 実験ランチャー │  launch_scientist_bfts.py                             │
│  │    (Launch)      │  アイデアJSON読込 → 実験ディレクトリ作成               │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ 3. BFTS実験 (Tree Search)                                            │   │
│  │                                                                       │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │   │
│  │  │ Stage 1 │──▶│ Stage 2 │──▶│ Stage 3 │──▶│ Stage 4 │              │   │
│  │  │ Draft   │   │ Hyparam │   │ Improve │   │Ablation │              │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘              │   │
│  │       │                                                               │   │
│  │       ▼ 各ノードで実行                                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ Phase 0: プランニング                                        │    │   │
│  │  │ Phase 1: ダウンロード/インストール (Singularity内)            │    │   │
│  │  │ Phase 2: コーディング                                        │    │   │
│  │  │ Phase 3: コンパイル                                          │    │   │
│  │  │ Phase 4: 実行                                                │    │   │
│  │  │ → メトリクス抽出 → プロット生成 → VLM分析 → サマリー         │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                                                                       │   │
│  └────────┬─────────────────────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 4. プロット集約   │  perform_plotting.py                                  │
│  │ (Plot Aggregation)│  ベストノードからプロット選択                         │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 5. 論文生成      │  perform_writeup.py                                   │
│  │    (Writeup)     │  LaTeX生成 + 引用収集                                  │
│  └────────┬─────────┘                                                        │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ 6. レビュー      │  perform_llm_review.py + perform_vlm_review.py        │
│  │    (Review)      │  NeurIPS形式評価 + 図版レビュー                        │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 詳細フロー

### ステップ1: アイデア準備（オプション）

既存のアイデアJSONを使用する場合はスキップ可能です。

```bash
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/himeno_benchmark_challenge.md \
  --model gpt-4o-2024-05-13 \
  --max-num-generations 3 \
  --num-reflections 5
```

**入力**: ワークショップ記述マークダウン（`*.md`）
**出力**: アイデアJSON（同じディレクトリに`*.json`として生成）

### ステップ2: 実験ランチャー

メインのエントリーポイントです。

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 4 \
  --enable_memgpt
```

**処理内容**:
1. アイデアJSONからアイデアを読み込み
2. 実験ディレクトリ `experiments/<timestamp>_<idea>_attempt_<id>/` を作成
3. `idea.md`, `idea.json`, `bfts_config.yaml` を書き込み
4. BFTS実験を開始

### ステップ3: BFTS実験

木探索による実験の自動実行です。

**ステージ構成**:
- **Stage 1 (Draft)**: 初期実装のドラフトを生成
- **Stage 2 (Hyperparameter)**: ハイパーパラメータ調整
- **Stage 3 (Improve)**: デバッグと改善
- **Stage 4 (Ablation)**: アブレーション研究

**各ノードで実行されるPhase**:

```
┌─────────────────────────────────────────────────────────────────┐
│ ノード実行フロー                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 0: プランニング                                           │
│  ├── 環境情報収集 (OS, CPU, GPU, コンパイラ, ライブラリ)         │
│  ├── 過去の実行履歴を参照                                        │
│  └── Phase 1-4の実行計画をJSON出力                               │
│           │                                                      │
│           ▼                                                      │
│  Phase 1: ダウンロード/インストール (Singularityコンテナ内)       │
│  ├── apt-get, pip install                                        │
│  ├── ソースからビルド                                            │
│  └── 反復的なインストーラー (最大12ステップ)                     │
│           │                                                      │
│           ▼                                                      │
│  Phase 2: コーディング                                           │
│  ├── LLMがソースコードを生成                                     │
│  └── ワークスペースにファイル書き込み                            │
│           │                                                      │
│           ▼                                                      │
│  Phase 3: コンパイル                                             │
│  ├── gcc/g++/nvcc等でビルド                                      │
│  └── エラー時はデバッグノードを生成                              │
│           │                                                      │
│           ▼                                                      │
│  Phase 4: 実行                                                   │
│  ├── プログラム実行                                              │
│  └── 出力ファイル (.npy) 収集                                    │
│           │                                                      │
│           ▼                                                      │
│  ポスト処理                                                      │
│  ├── メトリクス抽出 (スピードアップ、精度など)                   │
│  ├── プロットコード生成・実行                                    │
│  ├── VLM分析 (プロット品質評価)                                  │
│  └── ノードサマリー生成                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ステップ4: プロット集約

複数ノードから生成されたプロットを選択・集約します。

```bash
# ランチャー経由で自動実行、または個別実行:
python ai_scientist/perform_plotting.py \
  --experiment_dir experiments/<run>
```

**出力**: `experiments/<run>/figures/` および `auto_plot_aggregator.py`

### ステップ5: 論文生成

実験結果からLaTeX論文を生成します。

```bash
# ランチャー経由で自動実行、または:
python generate_paper.py \
  --experiment-dir experiments/<run> \
  --writeup-type normal
```

**出力**: `experiments/<run>/<run>.pdf`

### ステップ6: レビュー

生成された論文をNeurIPS形式でレビューします。

**出力**:
- `review_text.txt`: テキストレビュー
- `review_img_cap_ref.json`: 図版・キャプションレビュー

## スキップオプション

各ステージはスキップ可能です：

```bash
python launch_scientist_bfts.py \
  --skip_plot \       # プロット集約をスキップ
  --skip_writeup \    # 論文生成をスキップ
  --skip_review \     # レビューをスキップ
  ...
```

## ミニマル検証ラン

最小構成での動作確認：

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/himeno_benchmark_challenge.json \
  --idea_idx 0 \
  --singularity_image template/base.sif \
  --num_workers 2 \
  --skip_plot --skip_writeup --skip_review
```

## 既存実験からの再実行

既存の実験ディレクトリからプロット/論文を再生成：

```bash
python generate_paper.py \
  --experiment-dir experiments/<run> \
  --writeup-type normal \
  --model-agg-plots o3-mini-2025-01-31 \
  --model-writeup o1-preview-2024-09-12
```

## メモリ有効時のフロー

`--enable_memgpt` を指定すると、各フェーズでメモリ管理が有効になります：

```
┌─────────────────────────────────────────────────────────────────┐
│ メモリ有効時の追加処理                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 各Phase完了時                                                │
│     └── イベントをリコールメモリに自動記録                       │
│     └── エラー/成功詳細をアーカイブメモリに自動保存              │
│                                                                  │
│  2. LLMプロンプト生成時                                          │
│     └── コア/リコール/アーカイブからコンテキスト注入             │
│     └── LLMは <memory_update> ブロックでメモリを操作可能         │
│                                                                  │
│  3. LLMによるメモリ管理                                          │
│     └── idea_md_summary: LLMが必要に応じてコアに保存             │
│     └── phase0_summary: LLMが必要に応じてコアに保存              │
│     └── その他の重要情報: LLMが判断して保存                      │
│                                                                  │
│  4. 実験終了時                                                   │
│     └── final_memory-for-paper.md/json 生成                      │
│                                                                  │
│  注: idea_md_summary, phase0_summary は自動注入されません。       │
│      LLMが自律的に <memory_update> を使用して管理します。         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 関連ドキュメント

- [quickstart.md](getting-started/quickstart.md) - クイックスタートガイド
- [execution-modes.md](configuration/execution-modes.md) - Split/Single モードの詳細
- [outputs.md](configuration/outputs.md) - 出力ファイルの詳細
- [memory/memory.md](memory/memory.md) - メモリシステムの詳細
