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
|                  |   - Plot Plan          |
|                  |   - Plots              |
|                  |   - VLM Feedback       |
|                  |   - Code               |
|                  |   - Plot Code          |
+------------------+------------------------+
```

### ステージタブ（Stage 1–4ボタン）

画面上部に固定された4つのタブボタンで、探索ステージを切り替えます。

| ボタン | ステージ名 | 説明 | `selectStage()` の動作 |
|--------|-----------|------|------------------------|
| **Stage 1** | Preliminary Investigation | 初期実装と動作確認 | `stageData['Stage_1']` を読み込み、p5.jsツリーを再描画 |
| **Stage 2** | Baseline Tuning | ベースライン調整 | `stageData['Stage_2']` を読み込み、p5.jsツリーを再描画 |
| **Stage 3** | Research Agenda Execution | 創造的研究の実行 | `stageData['Stage_3']` を読み込み、p5.jsツリーを再描画 |
| **Stage 4** | Ablation Studies | アブレーション研究 | `stageData['Stage_4']` を読み込み、p5.jsツリーを再描画 |

**実装詳細**（`template.js`）：
- `selectStage(stageId)`: アクティブタブのCSSクラスを切り替え、`startSketch(stageId)` を呼び出して新しいp5.jsスケッチを生成
- `loadAllStageData(baseTreeData)`: 初期化時に各ステージの `tree_data.json` を `fetch()` で非同期ロード。`completed_stages` リストに含まれるステージのみがロードされる
- `updateTabVisibility()`: ロードされていないステージのタブには `disabled` クラスが付与されグレーアウト

### ノードのインタラクション

- **クリック** (`mousePressed`): ノードを選択して詳細パネルに情報を表示。`setNodeInfo()` を呼び出してすべてのパネルを更新
- **ホバー**: カーソルが手のアイコン (`HAND`) に変化（`isMouseOver()` 判定）
- **選択状態**: ノード色がアクセントカラー (`#1a439e`) に変化し、チェックマークが描画される
- **アニメーション**: ノード出現時にスケールアニメーション + ポップエフェクト（`appearProgress`, `popEffect`）。エッジの描画が完了すると子ノードが `visible = true` になる

### 詳細パネルの各セクション

ノードをクリックすると `setNodeInfo()` 関数が呼ばれ、`tree_data.json` から以下のデータが各HTMLパネルに描画されます。

#### Plan（`#plan`）
`treeData.plan[nodeIndex]` を `highlight.js` でシンタックスハイライトして表示。実験ノードの計画内容（Phase 0で生成された実行計画）が表示される。

#### Exception Info（`#exc_info`）
エラーが発生した場合、`treeData.exc_type`, `exc_info`, `exc_stack` から以下を表示：
- **Exception Type**: エラーの種類（例: `RuntimeError`）
- **Details**: エラーの詳細（JSON形式）
- **Stack Trace**: スタックトレース全文

エラーがない場合は「No exception info available」と表示。

#### Execution Time（`#exec_time`, `#exec_time_feedback`）
- `treeData.exec_time[nodeIndex]`: 実験の実行時間（秒）
- `treeData.exec_time_feedback[nodeIndex]`: 実行時間に対するフィードバック

#### Metrics（`#metrics`）
`treeData.metrics[nodeIndex]` のデータから、各メトリクスをテーブル形式で表示：
- **metric_name**: メトリクス名
- **description**: メトリクスの説明
- **lower_is_better**: 最適化方向（`true` → Minimize、`false` → Maximize）
- **data**: データセットごとの値（`dataset_name` と `value`）

`metrics.metric_names` 配列をイテレートし、各メトリクスについてデータセット別の値テーブルを生成。

#### Memory Operations（メモリパネル `#memory-panel`）

メモリ操作をフェーズごとにグループ化して表示する専用パネル。`treeData.memory_events[nodeIndex]` 配列を処理。

##### フェーズナビゲーションボタン

| ボタン | 関数 | 動作 |
|--------|------|------|
| **◀ Prev** | `shiftMemoryPhase(-1)` | 前のフェーズを表示（循環） |
| **Next ▶** | `shiftMemoryPhase(1)` | 次のフェーズを表示（循環） |

フェーズラベル（`#memory-phase-label`）に現在のフェーズ名が表示される。

**フェーズの分類** (`groupMemoryEvents` + `inferPhaseFromOp`):
- `phase` フィールドが明示的にある場合はそのまま使用
- ない場合は操作名から推定：
  - `node_fork`, `branch` → `node_setup`
  - `resources` → `resource_init`
  - `core_set`, `set_core`, `core_get`, `get_core` → `initialization`
  - `archival` → `archival_ops`
  - その他 → `system`

**フェーズの表示順序** (`sortMemoryPhases`):
1. `node_setup` → `resource_init` → `initialization` → `phase0` → `phase1` → `phase2` → `phase3` → `phase4` → `define_metrics` → `journal_summary` → `archival_ops` → `system`

各フェーズについて操作数のサマリーテーブル（`renderMemorySummary`）が表示され、その下に個別の操作イベントが描画される。

##### フィルタボタン

8つのフィルタボタンで操作タイプをフィルタリング：

| ボタン | `data-filter` | 分類される操作名 (`MEMORY_OP_CATEGORIES`) | 色 |
|--------|--------------|------------------------------------------|-----|
| **All** | `all` | 全ての操作 | - |
| **📖 Reads** | `reads` | `get_core`, `mem_core_get`, `render_for_prompt`, `mem_node_read`, `mem_archival_search`, `mem_archival_get`, `mem_recall_search`, `retrieve_archival` | `#4dabf7` |
| **💾 Writes** | `writes` | `set_core`, `mem_core_set`, `write_archival`, `mem_archival_write`, `mem_archival_update`, `mem_node_write`, `write_event` | `#69db7c` |
| **🗑️ Deletes** | `deletes` | `core_evict`, `core_delete`, `core_digest_compact` | `#ff6b6b` |
| **🌿 Forks** | `forks` | `mem_node_fork` | `#da77f2` |
| **🔄 Recalls** | `recalls` | `mem_recall_append` | `#ffd43b` |
| **📦 Resources** | `resources` | （template.htmlに存在するが、template.jsのカテゴリマッピングに未定義） | - |
| **🔧 Maintenance** | `maintenance` | `consolidate_recall_events`, `check_memory_pressure`, `auto_consolidate_memory`, `evaluate_importance_with_llm` | `#adb5bd` |

`setMemoryFilter(filter)` 関数がアクティブボタンのCSS切替 + `renderMemoryPhase()` を再呼び出しし、`filterEventsByCategory()` で表示イベントをフィルタリング。

##### 操作イベントの表示（`formatMemoryEvent`）

各イベントカードには以下が表示されます：
- **バッジ**: カテゴリ別のアイコンと色付きラベル（例: `📖 Reads`、`💾 Writes`）
- **操作名** (`op`): `mem_core_set`、`render_for_prompt` など
- **メモリタイプ** (`memory_type`): `core`、`archival`、`recall` など
- **キー情報**（該当する場合）:
  - `key`: コアメモリのキー名
  - `value_chars`: 値の文字数
  - `record_id`: アーカイブレコードID
- **メタ情報**: タイムスタンプ、`node_id`、`branch_id`
- **詳細情報**: JSON形式の `details` オブジェクト（展開可能な `<pre>` タグ内）

#### Plot Plan（`#plot_plan`）
`treeData.plot_plan[nodeIndex]` を表示。プロット生成の計画テキスト。

#### Plots（`#plots`）
`treeData.plots[nodeIndex]` 配列からプロット画像を表示。各画像は `<img>` タグで描画され、ロード失敗時にコンソールにエラーが出力される。

#### VLM Feedback（`#vlm_feedback`）
VLM (Vision Language Model) からのフィードバックを表示。以下の3つのサブセクションから構成：

1. **Plot Analysis** (`treeData.plot_analyses[nodeIndex]`): 各プロットの分析結果
   - `analysis.plot_path`: プロットファイル名
   - `analysis.analysis`: 分析テキスト
   - `analysis.key_findings`: 主要な発見事項リスト

2. **VLM Feedback Summary** (`treeData.vlm_feedback_summary[nodeIndex]`): 総合的なフィードバック

3. **Datasets Successfully Tested** (`treeData.datasets_successfully_tested[nodeIndex]`): テスト成功したデータセットのリスト

#### Code（`#code`）
`treeData.code[nodeIndex]` を `highlight.js` でPythonシンタックスハイライト付きで表示。実験のメインPythonコード。

#### Plot Code（`#plot_code`）
`treeData.plot_code[nodeIndex]` を `highlight.js` でPythonシンタックスハイライト付きで表示。プロット生成用コード。

---

## memory_database.html

### 概要

メモリデータベースの内容を詳細に閲覧するためのビューアーです。p5.js を使用したリサイズ可能なパネルレイアウトを採用しています。モジュラーテンプレートシステム（v2）により、`memory_database.js`、`tree_canvas.js`、`resizable.js`、`common.css`、`memory_database.css` が統合されます。

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

- **左パネル**: p5.jsによるツリーキャンバス。ノードクリックで右パネルが更新
- **リサイザー**: ドラッグで左右パネル比率を調整可能（`ResizablePanel` クラス、`localStorage` に比率保存）
- **右パネル**: 6つのタブで切り替わるメモリ情報ビュー

### ビュータブ（6つのボタン）

右パネル上部に配置された6つのタブボタンで表示内容を切り替えます。各ボタンは `switchView(view)` 関数を呼び、`currentView` を更新して `renderNodeContent()` を再呼び出しします。

| タブボタン | `data-view` | 呼び出される描画関数 | 説明 |
|-----------|-------------|---------------------|------|
| **Summary** | `summary` | `renderSummaryView()` | メモリ操作の概要統計と件数 |
| **Effective Memory** | `effective` | `renderEffectiveMemoryView()` | LLMが実際に見るメモリ状態（own + inherited統合） |
| **Memory Flow** | `memory-flow` | `renderMemoryFlowView()` | メモリ操作とインジェクションのシーケンス |
| **By Phase** | `by-phase` | `renderByPhaseView()` | フェーズごとのメモリ操作グループ |
| **Timeline** | `timeline` | `renderTimelineView()` | 全操作の時系列表示 |
| **All Data** | `all` | `renderAllDataView()` | own/inherited分離の詳細ビュー |

以下、各タブの表示内容と対応する関数の詳細を説明します。

---

### Summary タブ (`renderSummaryView`)

#### Inheritance Chain
ノードの継承チェーン（祖先ノード `nodeData.ancestors`）を `renderAncestorChain()` で表示。各祖先ノードはクリック可能で、`selectNodeByIndex(index)` により対応するノードに移動。

#### This Node's Memory Operations
`renderOperationsSummary()` で自ノードの操作サマリーを5カテゴリで表示：

| カテゴリ | `countByType()` での分類 | 表示色 |
|---------|-------------------------|--------|
| **Reads** | `render_for_prompt`, `mem_core_get`, `get_core`, `mem_archival_search`, `mem_archival_get`, `mem_recall_search`, `mem_node_read`, `retrieve_archival` | 青系 |
| **Writes** | `mem_core_set`, `set_core`, `mem_archival_write`, `mem_archival_update`, `write_archival`, `mem_node_write`, `mem_recall_append`, `recall_evict`, `recall_summarize`, `core_evict`, `mem_core_del` | 赤系 |
| **Forks** | `mem_node_fork`（ルート作成は除外: `parent_branch_id === null` の場合はカウントしない） | 青系 |
| **System** | `mem_resources_index_update`, `mem_resources_snapshot_upsert`, `apply_llm_memory_updates`, `check_memory_pressure`, `consolidate`, `importance_evaluation` | 灰色系 |
| **LLM** | `llm_core_set`, `llm_core_get`, `llm_core_delete`, `llm_archival_write`, `llm_archival_search`, `llm_archival_update`, `llm_recall_append`, `llm_recall_search`, `llm_recall_evict`, `llm_recall_summarize`, `llm_consolidate` | 紫系 |

#### Inherited Operations Summary
祖先から継承された操作のサマリー（上記と同じ5カテゴリ、opacity: 0.7 で薄く表示）。

#### Operations by Phase
`groupByPhase()` でフェーズ別にグループ化し、各フェーズの操作件数を表示。クリックすると **By Phase** タブに切り替わる。

フェーズ表示名 (`PHASE_LABELS`):

| phase キー | 表示名 |
|-----------|--------|
| `phase0` | Phase 0: Planning |
| `phase1` | Phase 1: Download/Install |
| `phase2` | Phase 2: Implementation |
| `phase3` | Phase 3: Evaluation |
| `phase4` | Phase 4: Analysis |
| `summary` | Summary |
| `memory_management` | Memory Management |
| `tree_structure` | Tree Structure |

#### Memory Contents
メモリコンテンツの件数（own と inherited を分離表示）：
- **Core KV (own)**: 自ノードの `own_core_kv` 件数
- **Events (own)**: 自ノードの `own_events` 件数
- **Archival (own)**: 自ノードの `own_archival` 件数
- **Core KV (inherited)** / **Events (inherited)** / **Archival (inherited)**: 祖先からの継承データ件数（クリックで All Data タブへ）

---

### Effective Memory タブ (`renderEffectiveMemoryView`)

**LLMが実際に見るメモリ状態**を表示する最も重要なビューです。ヘッダーにはグラデーション背景（`#1a472a` → `#16213e`）と緑色ボーダーで強調表示されます。

#### 統計バー
- Core KV Entries 数
- Recall Events 数
- Archival Records 数

#### Effective Core Memory
`nodeData.effective_core_kv` を表示。自ノードの値が祖先の同キー値を上書きする。

- **`[own]`** 青色ボーダー (`#4dabf7`): 自ノードで設定されたデータ（`ownCoreKeys` セットで判定）
- **`[inherited]`** グレー: 祖先から継承されたデータ

各エントリは `kv-key`（キー名）と `kv-value`（値）を表示。

#### Effective Recall Events
`nodeData.effective_events` を時系列順で表示。

- **kind**: イベント種別（`memory_injected` の場合は紫色 `#b197fc` で特別表示）
- **text**: イベント本文
- **meta**: Phase名、Tags
- `[own]` / `[inherited]` ラベルで出所を区別

#### Effective Archival Records
`nodeData.effective_archival` を表示。

- **tags**: タグ一覧（`archival-tag` スパン）
- **text**: アーカイブ本文
- `[own]` / `[inherited]` ラベルで出所を区別

#### データ継承ルール

| データ種別 | 継承動作 |
|-----------|---------|
| **Core KV** | 同一キーは子ノードの値が優先（`updated_at`の最新） |
| **Events** | 全ての祖先イベントが可視（累積、Copy-on-Write除外適用） |
| **Archival** | 全ての祖先レコードが可視（累積） |

---

### Memory Flow タブ (`renderMemoryFlowView`)

メモリ操作の**フロー**を可視化するビューです。操作をタイムスタンプ順にソートし、「ラウンド」単位でグループ化します。

#### ラウンド構造

操作は `render_for_prompt` 呼び出しを境界として「ラウンド」に分割されます：

1. **Memory Injection** (💉 緑 `#51cf66`): `render_for_prompt` 操作。LLMプロンプトに注入されたコンテキストを表示。新しいラウンドの開始を意味する。
   - **Budget (chars)**: `details.budget_chars` — メモリに割り当てられた文字数バジェット
   - **Core Items**: `details.core_count` — 注入されたコアメモリ項目数
   - **Recall Events**: `details.recall_count` — 注入されたリコールイベント数
   - **Archival Results**: `details.archival_count` — 注入されたアーカイブ検索結果数
   - **Task Hint**: `details.task_hint` — タスクヒント（存在する場合）
   - **Show Injected Context** ボタン: 折りたたみ可能。展開すると注入された Core Memory、Recall Events、Archival Search Results の実際のデータが表示される

2. **LLM Read Operations** (🔍 紫): `llm_` プレフィックスで始まる検索/取得操作（`llm_archival_search`, `llm_core_get` など）。ハイライト表示される（再クエリをトリガーする可能性があるため）。

3. **Other Operations** (⚙️): その他のメモリ更新操作。

各操作の詳細表示には、`renderMemoryCallDetails()` が呼ばれ、操作タイプごとに異なるクイックサマリーが生成されます（後述の「メモリ操作タイプ別の詳細表示」を参照）。

---

### By Phase タブ (`renderByPhaseView`)

`groupByPhase()` でフェーズごとにグループ化し、折りたたみ可能なセクション（`renderPhaseGroup`）で表示。

**フェーズ順序**: `phase0` → `phase1` → `phase2` → `phase3` → `phase4` → `summary` → `unknown`

各フェーズセクションには：
- **ヘッダー**: フェーズ名、操作件数 (`N ops`)
- **折りたたみ/展開ボタン** (`togglePhaseGroup`): `▼` アイコンをクリックで切り替え
- **操作リスト**: 各操作が `renderMemoryCall()` でカード表示

---

### Timeline タブ (`renderTimelineView`)

全操作をタイムスタンプ (`ts`) 順にソートして時系列表示。フェーズやカテゴリに関係なく、すべての `own_memory_calls` が一列に並びます。各操作は `renderMemoryCall()` + `renderMemoryCallDetails()` でカード表示されます。

---

### All Data タブ (`renderAllDataView`)

own と inherited データを完全に分離して詳細表示します。各セクションは `createSection()` で折りたたみ可能なUIとして生成されます。

#### This Node's Data（`data-group`）
- **Memory Operations**: 自ノードの `own_memory_calls` を `renderMemoryCall()` でリスト表示（デフォルト展開）
- **Core Memory (KV)**: `own_core_kv` を `renderCoreKV()` でkey-value表示
- **Recall Events**: `own_events` を `renderEvents()` で表示（kind、text、meta）
- **Archival Records**: `own_archival` を `renderArchival()` で表示（tags、text）

#### Inherited Data（`data-group inherited`）
- **Ancestor Chain**: `renderAncestorChain()` で祖先ノード一覧（クリックで移動）
- **Memory Operations**: `inherited_memory_calls` をグレー表示（デフォルト折りたたみ）
- **Core Memory (KV)**: `inherited_core_kv`（デフォルト折りたたみ）
- **Recall Events**: `inherited_events`（デフォルト折りたたみ）
- **Archival Records**: `inherited_archival`（デフォルト折りたたみ）

---

### メモリ操作タイプ別の詳細表示

`renderMemoryCallDetails()` は操作タイプごとに異なるクイックサマリーを生成します：

| 操作名 | 表示内容 |
|--------|---------|
| `render_for_prompt` | Budget (chars), Core items, Recall items, Archival items, Resources |
| `mem_recall_append` | Kind, Summary preview |
| `mem_node_fork` | Parent node ID, Child branch ID |
| `check_memory_pressure` | Pressure level, Usage percent |
| `mem_archival_write` / `write_archival` | Record ID, Content preview, Size (chars) |
| `mem_core_set` / `set_core` / `ingest_idea_md` | Key, Value preview, Size (chars), Importance |
| `get_core` / `mem_core_get` | Key, Found (Yes/No) |
| `core_evict` | Key, Reason |
| `mem_archival_search` / `mem_archival_get` | Query, Results count |
| その他 | 最初の4つのkey-valueペアを表示 |

すべての操作には **Show Full Details** ボタンがあり、展開すると以下が表示されます：
- **Value Content** (`details.value_preview`)
- **Text Content** (`details.text_preview`)
- **Summary** (`details.summary_preview`)
- **Tags** (`details.tags`)
- **All Details (JSON)**: `details` オブジェクト全体
- **Metadata**: `details` 以外の全フィールド（`op`, `phase`, `ts`, `node_id`, `branch_id` など）

### メモリ操作タイプの完全分類

`MEMORY_OP_TYPES` オブジェクト（`memory_database.js`）で定義される全操作：

| カテゴリ | 操作名 | タイプ | ラベル |
|---------|--------|--------|--------|
| **Injection** | `render_for_prompt` | read | Memory Injection |
| **Core** | `mem_core_get` | read | Core Get |
| **Core** | `mem_core_set` | write | Core Set |
| **Core** | `mem_core_del` | write | Core Delete |
| **Core** | `set_core` | write | Core Set |
| **Core** | `get_core` | read | Core Get |
| **Core** | `core_evict` | write | Core Evict |
| **Core** | `ingest_idea_md` | write | Ingest Idea MD |
| **Recall** | `mem_recall_append` | write | Recall Append |
| **Recall** | `mem_recall_search` | read | Recall Search |
| **Recall** | `recall_evict` | write | Recall Evict |
| **Recall** | `recall_summarize` | write | Recall Summarize |
| **Archival** | `mem_archival_write` | write | Archival Write |
| **Archival** | `mem_archival_update` | write | Archival Update |
| **Archival** | `mem_archival_search` | read | Archival Search |
| **Archival** | `mem_archival_get` | read | Archival Get |
| **Archival** | `write_archival` | write | Archival Write |
| **Archival** | `retrieve_archival` | read | Archival Retrieve |
| **Node** | `mem_node_fork` | fork | Node Fork |
| **Node** | `mem_node_read` | read | Node Read |
| **Node** | `mem_node_write` | write | Node Write |
| **Resources** | `mem_resources_index_update` | system | Resources Index Update |
| **Resources** | `mem_resources_snapshot_upsert` | system | Resources Snapshot |
| **Management** | `apply_llm_memory_updates` | system | LLM Memory Updates |
| **Management** | `check_memory_pressure` | system | Pressure Check |
| **Management** | `consolidate` | system | Consolidation |
| **Management** | `importance_evaluation` | system | Importance Eval |
| **LLM** | `llm_core_set` | llm | LLM Core Set |
| **LLM** | `llm_core_get` | llm | LLM Core Get |
| **LLM** | `llm_core_delete` | llm | LLM Core Delete |
| **LLM** | `llm_archival_write` | llm | LLM Archival Write |
| **LLM** | `llm_archival_search` | llm | LLM Archival Search |
| **LLM** | `llm_archival_update` | llm | LLM Archival Update |
| **LLM** | `llm_recall_append` | llm | LLM Recall Append |
| **LLM** | `llm_recall_search` | llm | LLM Recall Search |
| **LLM** | `llm_recall_evict` | llm | LLM Recall Evict |
| **LLM** | `llm_recall_summarize` | llm | LLM Recall Summarize |
| **LLM** | `llm_consolidate` | llm | LLM Consolidate |

### Copy-on-Write セマンティクス

メモリシステムは Copy-on-Write (CoW) セマンティクスを採用しています：

- **inherited_exclusions**: 統合されたイベントのID一覧。これらのイベントは inherited ビューから除外される
- **inherited_summaries**: 祖先イベントの統合サマリー。元のイベント群を要約したもの

### その他の機能

#### バッジ色

| 操作タイプ | バッジ色 | 用途 |
|-----------|---------|------|
| `read` | 緑系 | 読み取り操作 |
| `write` | 赤系 | 書き込み操作 |
| `fork` | 青系 | フォーク操作 |
| `system` | グレー | システム操作 |
| `llm` | 紫系 | LLM関連操作 |

#### 折りたたみセクション (`createSection`)
- クリックで展開/折りたたみ（`toggleSection`）
- バッジで件数を表示
- `collapsed` CSSクラスでトグル

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

`unified_tree_viz.html` にはローカルホスト（127.0.0.1 または localhost）で実行時に自動リロード機能が組み込まれています。ファイルが更新されると自動的にページがリロードされます（1秒間隔で `HEAD` リクエストにより `last-modified` ヘッダーを監視）。

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

生成時にプレースホルダー（`__COMMON_CSS__`, `__MEMORY_DATABASE_JS__`, `__RESIZABLE_JS__`, `__TREE_CANVAS_JS__`, `__MEMORY_DATABASE_CSS__`, `__JS_DATA__`, `__EXPERIMENT_NAME__`）が実際のアセット内容で置換されます。

### unified_tree_viz テンプレート

```
ai_scientist/treesearch/utils/viz_templates/
├── template.html             # メインHTMLテンプレート
└── template.js               # JavaScript（ツリー可視化ロジック）
```

`template.js` 内の `"PLACEHOLDER_TREE_DATA"` が生成時に実際の `tree_data.json` の内容で置換されます。

---

## トラブルシューティング

### ツリーが表示されない

1. `tree_data.json` が存在するか確認
2. ブラウザの開発者コンソールでエラーを確認
3. JSONデータの形式が正しいか確認（`layout` と `edges` 配列が必要）

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
