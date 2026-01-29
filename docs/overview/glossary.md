# 用語集 (Glossary)

このドキュメントでは、HPC-AutoResearchで使用される主要な用語を説明します。

## 全体構成関連

### BFTS (Best-First Tree Search)
エージェントが探索木を構築しながら最適解を探す木探索アルゴリズム。各ノードは1つのコード実装を表し、成功・失敗に応じて次のノードが生成されます。

### Tree Search
実験コードの探索を木構造として行う手法。ルートノードから始まり、ドラフト→デバッグ→改善のサイクルで枝を伸ばしていきます。

### Node (ノード)
探索木における1つの実験コード実装。ノードはPhase 0-4を経て実行され、成功すると子ノードを生成します。

### Stage (ステージ)
探索の進行段階を表します：
- **Stage 1** (`initial_implementation`): 初期実装の生成と動作検証
- **Stage 2** (`baseline_tuning`): ベースラインチューニング、追加データセットでの評価
- **Stage 3** (`creative_research`): 創造的改善、実験計画の実行
- **Stage 4** (`ablation_studies`): アブレーション研究、リスク要因の検証

### Worker (ワーカー)
並列実行されるエージェントプロセス。`num_workers`で並列数を指定し、各ワーカーはGPUまたはCPUにマップされます。

## Phase（フェーズ）関連

### Phase 0 (Planning)
プランニングフェーズ。環境情報を収集し、Phase 1-4の実行計画を生成します。

### Phase 1 (Download/Install)
依存関係のダウンロードとインストールを行うフェーズ。Singularityコンテナ内で`apt-get`、`pip install`、ソースビルドなどを実行します。反復的インストーラーの最大ステップ数は`phase1_max_steps`で設定（デフォルト: 100）。

### Phase 2 (Coding)
コード生成フェーズ。LLMがワークスペースにファイルを生成します。

### Phase 3 (Compile)
コンパイルフェーズ。`gcc`、`make`などでビルドを実行します。

### Phase 4 (Run)
実行フェーズ。ビルドしたプログラムを実行し、出力（`.npy`など）を収集します。

### Split Mode
Phase 0-4を明示的に分離して実行するモード（デフォルト）。Singularityコンテナ内で実行されます。

### Single Mode
従来のレガシー実行モード。ホスト環境で直接実行し、Phase分離を行いません。

## メモリ関連

### MemGPT
階層的メモリ管理システム。`memory.enabled=true`で有効化されます（デフォルトで有効）。

### Core Memory (コアメモリ)
常にプロンプトに注入される重要な情報。LLMが自律的にキー値を設定・管理します。`core_max_chars`で容量制限（デフォルト: 10000文字、コードのフォールバック: 2000文字）。

### Recall Memory (リコールメモリ)
直近のイベントタイムライン。ノードの作成、コンパイル成功/失敗などのイベントが記録されます。`recall_max_events`で件数制限（デフォルト: 5件、コードのフォールバック: 20件）。

### Archival Memory (アーカイブメモリ)
長期保存用メモリ。FTS5による全文検索が可能。詳細なエラー情報や成功パターンを保存します。`retrieval_k`で検索時の上位k件を指定（デフォルト: 4件、コードのフォールバック: 8件）。

### Branch (ブランチ)
メモリの分岐。子ノードは親のメモリを継承し、書き込みは自身のブランチに隔離されます。

### Memory Pressure (メモリ圧力)
メモリ使用量に基づく圧力レベル：
- **low**: 70%未満
- **medium**: 70-85%
- **high**: 85-95%
- **critical**: 95%以上

### LLM Compression
LLMを使用してテキストを圧縮する機能。単純な切り捨てではなく、重要な情報を保持します。

## リソース関連

### Resource File
データセット、GitHubリポジトリ、HuggingFaceモデルを定義するJSONまたはYAMLファイル。`--resources`フラグで指定します。

### Local Resource
ホストシステムに存在し、コンテナにマウントされるリソース。

### Mount Path
コンテナ内でリソースがマウントされるパス（例: `/workspace/input/data`）。

### Staging
リソースをワークスペースにコピーまたはシンボリックリンクすること。

## 出力関連

### Experiment Output Filename
実験結果の出力ファイル名。`{experiment_name}_data.npy`形式で、実験名に基づいて動的に生成されます。例: `stability_oriented_autotuning_v2_data.npy`。これにより、どの実験が出力を生成したかが明確になります。

### Experiment Directory
`experiments/<timestamp>_<idea>_attempt_<id>/`形式のディレクトリ。実験の全成果物を含みます。

### Tree Visualization
`unified_tree_viz.html`で提供される探索木のHTML可視化。

### Token Tracker
LLM API呼び出しのトークン使用量を記録するシステム。`token_tracker.json`に出力。

### Final Memory
実行終了時に生成されるメモリサマリー。`final_memory_for_paper.md`と`final_memory_for_paper.json`。

## 論文生成関連

### Plot Aggregation
複数ノードから生成されたプロットを選択・集約するプロセス。

### VLM (Vision-Language Model)
画像を解析できるマルチモーダルLLM。プロット品質評価に使用。

### Writeup
実験結果からLaTeX論文を生成するプロセス。

### Review
生成された論文をNeurIPS形式でレビューするプロセス。

## 設定関連

### bfts_config.yaml
メインの設定ファイル。実験ごとに`experiments/<run>/`にコピーされます。

### Persona
`agent.role_description`で設定されるエージェントの役割（例: "HPC Researcher"）。プロンプト内の`{persona}`トークンが置換されます。未設定時のデフォルトは"AI researcher"。

### Singularity Image (SIF)
Singularityコンテナイメージ。`exec.singularity_image`で指定。

### Overlay
Singularityコンテナに書き込み可能なレイヤーを追加するためのイメージ。

## CLI関連

### launch_scientist_bfts.py
メインのランチャースクリプト。アイデア読み込み→実験→プロット→論文→レビューを統括。

### generate_paper.py
既存の実験ディレクトリからプロット/論文/レビューを生成するスクリプト。

### perform_ideation_temp_free.py
ワークショップ記述からアイデアJSONを生成するスクリプト。

## 関連プロジェクト

### AI-Scientist-v2
Sakana AIによる元プロジェクト。本フォークはHPC向けに拡張。

### AIDE
木探索コンポーネントのベースとなったプロジェクト。

### MemGPT
階層的メモリの概念の元となったプロジェクト。
