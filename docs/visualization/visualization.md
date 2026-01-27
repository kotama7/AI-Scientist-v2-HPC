# Visualization Tools

実験の実行結果を可視化するためのHTMLツールについて説明します。

## Overview

AI-Scientist-v2 は実験の進行状況と結果を可視化するために以下のHTMLファイルを生成します：

| ファイル | 目的 |
|---------|------|
| `unified_tree_viz.html` | ツリー探索の可視化（コード、プラン、メトリクス、プロット）- 全ステージ統合 |
| `tree_plot.html` | 各ステージ個別のツリー可視化 |
| `memory_database.html` | メモリ操作の詳細ビューア（own/inherited データの可視化） |

これらのファイルは `experiments/<experiment_name>/logs/<run-id>/` ディレクトリに生成されます。

## 生成場所

```
experiments/<experiment_name>/logs/<run-id>/
├── unified_tree_viz.html           # 全ステージ統合ビジュアライザー
├── memory_database.html            # メモリデータベースビューア
├── memory/                         # メモリデータベース
│   ├── memory.sqlite               # SQLiteデータベース
│   └── memory_calls.jsonl          # メモリ操作ログ
├── phase_logs/                     # フェーズ実行ログ
├── experiment_results/             # 実験結果（各ノード）
├── stage_1_initial_implementation_*/
│   ├── tree_data.json              # ステージ1のツリーデータ
│   ├── tree_plot.html              # ステージ1のツリー可視化
│   └── ...
├── stage_2_baseline_tuning_*/
│   ├── tree_data.json
│   ├── tree_plot.html
│   └── ...
├── stage_3_creative_research_*/
│   └── ...
└── stage_4_ablation_studies_*/
    └── ...
```

## 使い方

ローカルブラウザで開くか、Live Server などの開発サーバーを使用してください：

```bash
# 直接ブラウザで開く
firefox experiments/<experiment_name>/logs/0-run/unified_tree_viz.html

# または Live Server を使用（自動リロード機能あり）
# VSCode の Live Server 拡張機能などを使用
```

---

## unified_tree_viz.html

### 概要

p5.js を使用したインタラクティブなツリー探索ビジュアライザーです。実験の各ステージにおけるノード（実験試行）の関係性を視覚的に表示します。

### 画面構成

```
+------------------+------------------------+
|                  |                        |
|   Tree Canvas    |     Detail Panel       |
|   (左側 40%)     |     (右側 60%)         |
|                  |                        |
|   ノードを       |   - Plan               |
|   クリックで     |   - Exception Info     |
|   詳細表示       |   - Execution Time     |
|                  |   - Metrics            |
|                  |   - Memory Operations  |
|                  |   - Plots              |
|                  |   - VLM Feedback       |
|                  |   - Code               |
+------------------+------------------------+
```

### ステージタブ

画面上部のタブで4つのステージを切り替えられます：

| ステージ | 名称 | 説明 |
|---------|------|------|
| Stage 1 | Preliminary Investigation | 初期実装と動作確認 |
| Stage 2 | Baseline Tuning | ベースライン調整 |
| Stage 3 | Research Agenda Execution | 創造的研究の実行 |
| Stage 4 | Ablation Studies | アブレーション研究 |

完了していないステージのタブは無効化（グレーアウト）されます。

### ノードのインタラクション

- **クリック**: ノードを選択して詳細パネルに情報を表示
- **ホバー**: カーソルが手のアイコンに変化
- **選択状態**: チェックマークが表示され、アクセントカラーに変化

### 詳細パネルの内容

#### Plan
実験ノードの計画内容を表示します。

#### Exception Info
エラーが発生した場合、以下を表示：
- Exception Type: エラーの種類
- Details: エラーの詳細
- Stack Trace: スタックトレース

#### Execution Time
実験の実行時間（秒）とフィードバックを表示。

#### Metrics
定義されたメトリクスとデータセットごとの値をテーブル形式で表示：
- メトリクス名
- 説明
- 最適化方向（Minimize/Maximize）
- データセット別の値

#### Memory Operations
メモリ操作をフェーズごとに表示。詳細は「メモリパネル」を参照。

#### Plots
生成されたプロット画像を表示。

#### VLM Feedback
Vision Language Model からのフィードバックとプロット分析を表示：
- Plot Analysis: 各プロットの分析結果
- Key Findings: 主要な発見事項
- VLM Feedback Summary: 総合的なフィードバック
- Datasets Successfully Tested: テスト成功したデータセット

#### Code
実験のPythonコードをシンタックスハイライト付きで表示。

### メモリパネル

メモリ操作を詳細に可視化するパネルです。

#### フェーズナビゲーション
`◀ Prev` / `Next ▶` ボタンでフェーズを切り替えます：
- `phase0`: Phase 0 - Planning
- `phase1`: Phase 1 - Download/Install
- `phase2`: Phase 2 - Implementation
- `phase3`: Phase 3 - Evaluation
- `phase4`: Phase 4 - Analysis
- `summary`: Summary

#### フィルタボタン
操作タイプでフィルタリング：

| フィルタ | アイコン | 説明 |
|---------|---------|------|
| All | - | 全ての操作 |
| Reads | 📖 | メモリ読み取り |
| Writes | 💾 | メモリ書き込み |
| Deletes | 🗑️ | メモリ削除 |
| Forks | 🌿 | ノードフォーク |
| Recalls | 🔄 | リコール操作 |
| Maintenance | 🔧 | メンテナンス操作 |

#### 操作イベントの表示
各イベントには以下が表示されます：
- バッジ（操作カテゴリ）
- 操作名（op）
- メモリタイプ
- キー情報（該当する場合）
- タイムスタンプ
- ノードID、ブランチID
- 詳細情報（JSON形式）

---

## memory_database.html

### 概要

メモリデータベースの内容を詳細に閲覧するためのビューアーです。p5.js を使用したリサイズ可能なパネルレイアウトを採用しています。

### 画面構成

```
+------------------+|+------------------------+
|                  ||                        |
|   Tree Canvas    ||     Detail Panel       |
|   (ツリー表示)   ||     (タブ切替)         |
|                  ||                        |
+------------------+|+------------------------+
                    ^
                    リサイザー（ドラッグで調整）
```

### ビュータブ

6つのタブでメモリ情報を異なる視点から閲覧できます：

| タブ | 説明 |
|------|------|
| **Summary** | メモリ操作の概要統計と件数 |
| **Effective Memory** | LLMが実際に見るメモリ状態（own + inherited統合） |
| **Memory Flow** | メモリ操作とインジェクションのシーケンス |
| **By Phase** | フェーズごとのメモリ操作グループ |
| **Timeline** | 全操作の時系列表示 |
| **All Data** | own/inherited分離の詳細ビュー |

### Effective Memory ビュー

**LLMが実際に見るメモリ状態**を表示する最も重要なビューです。

```
┌─────────────────────────────────────────────┐
│  Effective Memory State                      │
│  This is the actual memory that the LLM sees │
└─────────────────────────────────────────────┘
```

#### 表示内容

- **Core KV**: 自ノードの値が祖先の同キー値を上書き
- **Events/Archival**: 自ノード + 祖先の全エントリ（時系列順）
- **視覚的区別**:
  - `[own]` 青色ボーダー: 自ノードで設定されたデータ
  - `[inherited]` グレー: 祖先から継承されたデータ

#### データ継承ルール

| データ種別 | 継承動作 |
|-----------|---------|
| **Core KV** | 同一キーは子ノードの値が優先（`updated_at`の最新） |
| **Events** | 全ての祖先イベントが可視（累積、Copy-on-Write除外適用） |
| **Archival** | 全ての祖先レコードが可視（累積） |

### Memory Flow ビュー

メモリ操作の**フロー**を可視化する新しいビューです。

#### 表示内容

- **Memory Injection** (💉 緑): LLMプロンプトに注入されたコンテキスト
  - Budget (chars): 文字数バジェット
  - Core Items: コアメモリ項目数
  - Recall Events: リコールイベント数
  - Archival Results: アーカイブ検索結果数
- **LLM Read Operations** (🔍 紫): LLMが開始した読み取り操作
- **Write Operations** (⚙️ 赤): メモリ更新操作

#### ラウンド構造

操作は「ラウンド」ごとにグループ化されます：
1. Memory Injection（新しいプロンプトへの注入）が新しいラウンドを開始
2. LLM Read Operationsはハイライト表示（再クエリをトリガーする可能性があるため）
3. Other Operationsはその他の操作

### Summary ビュー

#### Inheritance Chain
ノードの継承チェーン（祖先ノード）を表示。各祖先ノードはクリックで選択可能。

#### This Node's Memory Operations
自ノードの操作サマリー：
- Reads / Writes / Forks / System / LLM の件数
- フェーズ別の操作件数

#### Memory Contents
メモリコンテンツの件数（own と inherited を分離表示）：
- Core KV: キーバリューストア
- Events: リコールイベント
- Archival: アーカイブレコード

### All Data ビュー

own と inherited データを分離して詳細表示します。

#### This Node's Data
- Memory Operations: 自ノードのメモリ操作
- Core Memory (KV): 自ノードのコアメモリ
- Recall Events: 自ノードのリコールイベント
- Archival Records: 自ノードのアーカイブレコード

#### Inherited Data
- Ancestor Chain: 継承元の祖先ノード一覧
- Memory Operations / Core Memory / Recall Events / Archival Records: 継承されたデータ

### Copy-on-Write セマンティクス

メモリシステムは Copy-on-Write (CoW) セマンティクスを採用しています：

- **inherited_exclusions**: 統合されたイベントのID一覧。これらのイベントは inherited ビューから除外される
- **inherited_summaries**: 祖先イベントの統合サマリー。元のイベント群を要約したもの

### メモリ操作タイプ

| カテゴリ | 操作 | 説明 |
|---------|------|------|
| **Injection** | `render_for_prompt` | LLMプロンプトへのメモリ注入 |
| **Core** | `mem_core_get`, `mem_core_set`, `set_core`, `get_core`, `core_evict` | コアメモリ操作 |
| **Recall** | `mem_recall_append`, `mem_recall_search`, `recall_evict`, `recall_summarize` | リコールメモリ操作 |
| **Archival** | `mem_archival_write`, `mem_archival_search`, `mem_archival_get`, `write_archival`, `retrieve_archival` | アーカイブメモリ操作 |
| **Node** | `mem_node_fork`, `mem_node_read`, `mem_node_write` | ノード操作 |
| **Resources** | `mem_resources_index_update`, `mem_resources_snapshot_upsert` | リソース操作 |
| **Management** | `check_memory_pressure`, `consolidate`, `importance_evaluation` | メモリ管理 |
| **LLM** | `llm_core_set`, `llm_archival_write`, `llm_recall_append`, etc. | LLMが開始したメモリ操作 |

### その他の機能

#### 統計バー
メモリ操作の概要統計を表示：
- 総操作数
- 読み取り/書き込み/削除数
- その他の集計情報

#### セクション
折りたたみ可能なセクションで情報を整理：
- クリックで展開/折りたたみ
- バッジで件数を表示

#### バッジ

| バッジ | 色 | 説明 |
|--------|-----|------|
| `badge-read` | 緑 | 読み取り操作 |
| `badge-write` | 赤 | 書き込み操作 |
| `badge-fork` | 青 | フォーク操作 |
| `badge-system` | グレー | システム操作 |
| `badge-llm` | 紫 | LLM関連操作 |

---

## tree_plot.html

### 概要

各ステージディレクトリに生成される個別のツリー可視化ファイルです。`unified_tree_viz.html` と同じ機能を持ちますが、単一ステージのデータのみを表示します。

### 使用シーン

- 特定のステージだけを素早く確認したい場合
- ステージごとのツリーデータ（`tree_data.json`）と一緒に参照する場合

---

## カスタマイズ

### 背景色の変更

ブラウザの開発者コンソールで以下を実行：

```javascript
// unified_tree_viz.html
setBackgroundColor('#f0f0f0');

// または直接変数を更新
updateBackgroundColor('#ffffff');
```

### Live Server での自動リロード

`unified_tree_viz.html` にはローカルホスト（127.0.0.1 または localhost）で実行時に自動リロード機能が組み込まれています。ファイルが更新されると自動的にページがリロードされます。

---

## テンプレート構造

### Modular Template System (v2)

memory_database.html は新しいモジュラーテンプレートシステムを使用しています：

```
ai_scientist/treesearch/utils/templates/
├── memory_database_v2.html   # メインテンプレート（プレースホルダー付き）
├── memory_database.html      # レガシーテンプレート（後方互換性用）
└── assets/
    ├── common.css            # 共通スタイル
    ├── memory_database.css   # memory_database固有スタイル
    ├── memory_database.js    # メモリ操作レンダリングロジック
    ├── resizable.js          # リサイズ可能パネル機能
    └── tree_canvas.js        # p5.jsツリーキャンバス
```

生成時にプレースホルダー（`__COMMON_CSS__`, `__MEMORY_DATABASE_JS__` など）が実際のアセット内容で置換されます。

### unified_tree_viz テンプレート

```
ai_scientist/treesearch/utils/viz_templates/
├── template.html             # メインHTMLテンプレート
└── template.js               # JavaScript（ツリー可視化ロジック）
```

---

## トラブルシューティング

### ツリーが表示されない

1. `tree_data.json` が存在するか確認
2. ブラウザの開発者コンソールでエラーを確認
3. JSONデータの形式が正しいか確認

### プロットが表示されない

1. 画像パスが正しいか確認
2. 画像ファイルが存在するか確認
3. CORSエラーがないか確認（Live Server推奨）

### メモリパネルが空

1. `memory_events` データが `tree_data.json` に含まれているか確認
2. 該当ノードにメモリイベントが記録されているか確認
3. `memory_calls.jsonl` が生成されているか確認

### memory_database.html が大きすぎる

大規模な実験では memory_database.html が数十MB〜100MB以上になることがあります。これは全メモリデータがJSONとしてHTMLに埋め込まれるためです。

対処法：
- ブラウザのメモリ制限に注意（Chrome推奨）
- 必要なノードのみを選択して表示

---

## 関連ファイル

| ドキュメント | 関連コード |
|-------------|-----------|
| この文書 | `ai_scientist/treesearch/utils/` |
| [memory.md](../memory/memory.md) | `ai_scientist/memory/` |
| [outputs.md](../configuration/outputs.md) | 出力ディレクトリ構造 |
