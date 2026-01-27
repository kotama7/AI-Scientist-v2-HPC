# コンセプト (Core Concepts)

このドキュメントでは、HPC-AutoResearchの核心的なコンセプトを説明します。
システムの設計思想と各コンポーネントの役割を理解することができます。

## 1. 木探索による自動研究 (Tree Search for Automated Research)

### なぜ木探索なのか？

研究プロセスは本質的に探索的です。最初のアイデアが必ずしも成功するとは限らず、
試行錯誤を繰り返しながら最良の解決策を見つけます。HPC-AutoResearchは、この
プロセスを**木探索**として形式化しています。

```
                          Root (アイデア)
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
         Draft 1           Draft 2           Draft 3
            │                 │                 │
       ┌────┼────┐       ┌────┼────┐           ✗
       │    │    │       │    │    │
    Debug Improve ✗   Debug Improve ✗
       │    │           │    │
       ✗   Best        ✗    ─┘
            │
       Hyperparam
            │
       Ablation
            │
         Final
```

### 木構造の利点

1. **並列探索**: 複数の実装を同時に試行可能
2. **ロールバック**: 失敗時に親ノードに戻って別のアプローチを試行
3. **最良優先**: 成功したノードを優先して深掘り
4. **履歴活用**: 過去の失敗から学習

### ノードの種類

| ノード種類 | 目的 | 親からの入力 |
|-----------|------|-------------|
| **Draft** | 初期実装を生成 | アイデア記述 |
| **Debug** | エラーを修正 | 失敗したコード + エラーログ |
| **Improve** | 性能を改善 | 動作するコード + メトリクス |
| **Hyperparam** | パラメータ調整 | ベストコード + 調整履歴 |
| **Ablation** | 構成要素の効果検証 | 最終コード |

### Stage Definition Details

Four main stages are defined in the implementation (`agent_manager.py`):

| Stage | Internal Name | Purpose |
|-------|---------------|---------|
| Stage 1 | `initial_implementation` | Generate initial implementation draft, verify functionality |
| Stage 2 | `baseline_tuning` | Baseline tuning, evaluation on additional datasets |
| Stage 3 | `creative_research` | Creative improvements, execute experiment plans |
| Stage 4 | `ablation_studies` | Ablation studies, verify risk factors |

Stage goals are defined in prompt files:
- `prompt/agent/manager/stages/stage1_goals.txt`
- `prompt/agent/manager/stages/stage2_goals.txt`
- `prompt/agent/manager/stages/stage3_goals.txt`
- `prompt/agent/manager/stages/stage4_goals.txt`

### Node Detailed Attributes

Each node is defined by the `Node` dataclass in `journal.py` with the following key attributes:

| Category | Attributes | Description |
|----------|------------|-------------|
| Basic Info | `id`, `step`, `ctime` | Unique ID, step number, creation time |
| Relations | `parent`, `children`, `branch_id` | Parent node, child nodes, memory branch |
| Code | `code`, `plan`, `phase_artifacts` | Generated code, plan, per-phase artifacts |
| Execution | `_term_out`, `exec_time`, `exc_type` | Output, execution time, exception info |
| Evaluation | `metric`, `analysis`, `is_buggy` | Metrics, analysis results, buggy flag |
| VLM | `plot_analyses`, `vlm_feedback_summary` | Plot analyses, VLM feedback |
| Special | `ablation_name`, `is_seed_node` | Ablation name, seed node flag |
| Inheritance | `inherited_from_node_id`, `worker_sif_path` | Source node ID, reused SIF path |

### Multi-Seed Evaluation

At the completion of each main stage, the best node undergoes multi-seed evaluation:

```
Best Node ──┬──▶ Seed 1 ──▶ Run ──▶ Metrics
            ├──▶ Seed 2 ──▶ Run ──▶ Metrics
            └──▶ Seed 3 ──▶ Run ──▶ Metrics
                           ↓
                 Seed Aggregation Node
                 (mean/std/statistical plots)
```

Configuration (`bfts_config.yaml`):
```yaml
agent:
  multi_seed_eval:
    num_seeds: 3  # Number of evaluation seeds
```

## 2. フェーズ分離アーキテクチャ (Split-Phase Architecture)

### なぜフェーズを分離するのか？

HPC環境では、依存関係のインストールからコード実行までの各ステップが複雑です。
フェーズ分離により：

1. **責任の明確化**: 各フェーズは単一の責任を持つ
2. **エラー特定**: どのフェーズで問題が発生したかが明確
3. **再試行戦略**: フェーズごとに異なる再試行ロジックを適用可能
4. **コンテナ活用**: Singularity内での実行を前提とした設計

### フェーズ詳細

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 0: プランニング                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: 環境を分析し、Phase 1-4の実行計画を策定                                 │
│                                                                              │
│ 入力:                                                                        │
│   - アイデア/タスク記述                                                       │
│   - 環境情報（OS、CPU、GPU、コンパイラ、ライブラリ）                          │
│   - 過去の実行履歴（あれば）                                                  │
│                                                                              │
│ 出力 (phase0_plan.json):                                                     │
│   - goal_summary: 目標の要約                                                  │
│   - implementation_strategy: 実装戦略                                        │
│   - dependencies: 必要な依存関係 (apt, pip, source)                          │
│   - download_commands_seed: ダウンロードコマンド                              │
│   - compile_plan: ビルド設定                                                  │
│   - compile_commands: ビルドコマンド                                          │
│   - run_commands: 実行コマンド                                               │
│   - phase_guidance: 各フェーズへのガイダンス                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Phase 1: ダウンロード/インストール                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: 依存関係をインストールし、実行環境を準備                                 │
│                                                                              │
│ 実行環境: Singularityコンテナ内                                               │
│                                                                              │
│ 特徴:                                                                        │
│   - 反復的インストーラー: 最大12ステップで段階的に構築                        │
│   - コマンド種類: apt-get, pip install, ソースビルド                          │
│   - 進捗追跡: 各ステップの結果（exit_code, stdout, stderr）を記録            │
│   - エラー回復: LLMが失敗原因を分析し、次のコマンドを決定                     │
│                                                                              │
│ 書き込みモード:                                                              │
│   - tmpfs: 高速だがメモリ制限あり                                             │
│   - overlay: 永続化可能だがやや低速                                           │
│   - none: 読み取り専用（依存関係がベースイメージに含まれる場合）              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Phase 2: コーディング                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: 実験コードを生成し、ワークスペースに配置                                 │
│                                                                              │
│ 出力構造:                                                                    │
│   workspace/                                                                 │
│   ├── src/                                                                   │
│   │   ├── main.c                                                             │
│   │   └── utils.h                                                            │
│   ├── Makefile                                                               │
│   └── working/                                                               │
│       └── (実行時出力がここに生成)                                            │
│                                                                              │
│ LLM出力形式:                                                                 │
│   - file_tree: ディレクトリ構造                                               │
│   - files: {path, mode, content} のリスト                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Phase 3: コンパイル                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: ソースコードをビルド                                                    │
│                                                                              │
│ 対応言語/ツール:                                                             │
│   - C/C++ (gcc, g++, clang)                                                  │
│   - CUDA (nvcc)                                                              │
│   - Fortran (gfortran)                                                       │
│   - Make/CMake                                                               │
│                                                                              │
│ build_plan構造:                                                              │
│   - language: プログラミング言語                                              │
│   - compiler_selected: 使用コンパイラ                                        │
│   - cflags: コンパイルフラグ                                                  │
│   - ldflags: リンクフラグ                                                     │
│   - output: 出力バイナリパス                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            Phase 4: 実行                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 目的: ビルドしたプログラムを実行し、結果を収集                                 │
│                                                                              │
│ 期待される出力:                                                              │
│   - working/experiment_data.npy (デフォルト)                                 │
│   - または expected_outputs で指定されたファイル                              │
│                                                                              │
│ 実行後処理:                                                                  │
│   1. メトリクス抽出: stdout/stderrからスピードアップ等を解析                  │
│   2. プロット生成: .npyデータから可視化コードを生成・実行                      │
│   3. VLM分析: 生成された画像をVLMで品質評価                                   │
│   4. ノードサマリー: 結果の要約を生成                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. MemGPT階層的メモリ (MemGPT-Style Hierarchical Memory)

### なぜ階層的メモリが必要なのか？

LLMには**コンテキストウィンドウ**の制限があります。長時間の研究プロセスでは、
すべての情報をプロンプトに含めることは不可能です。MemGPTスタイルのメモリは
この問題を解決します：

1. **コンテキスト管理**: 予算内で最も関連性の高い情報を注入
2. **長期記憶**: 過去の成功/失敗パターンを保持
3. **分岐継承**: 子ノードが親の学習を引き継ぐ

### メモリ層

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Core Memory (コアメモリ)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 特徴: 常にプロンプトに含まれる重要情報                                        │
│ 容量: core_max_chars (デフォルト 16000文字)                                   │
│                                                                              │
│ 典型的な内容:                                                                │
│   - RESOURCE_INDEX: 利用可能なリソースのダイジェスト (自動保存、別セクション)   │
│   - LLMが設定したキー: optimal_threads, best_flags など (予約キーなし)        │
│                                                                              │
│ 管理:                                                                        │
│   - 重要度 (1-5): 高いほど残りやすい                                          │
│   - TTL: 有効期限付きエントリ                                                 │
│   - 自動退避: 容量超過時、低重要度エントリをArchivalへ移動                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Recall Memory (リコールメモリ)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 特徴: 最近のイベントタイムライン                                               │
│ 容量: recall_max_events (デフォルト 20件)                                     │
│                                                                              │
│ 記録されるイベント:                                                          │
│   - node_created: ノード作成                                                  │
│   - phase1_complete/failed: Phase 1結果                                       │
│   - compile_complete/failed: コンパイル結果                                   │
│   - run_complete/failed: 実行結果                                             │
│   - metrics_extracted: メトリクス抽出                                         │
│   - node_result: ノード最終結果                                               │
│                                                                              │
│ 管理:                                                                        │
│   - FIFO: 古いイベントから削除                                                │
│   - 統合: 類似イベントをまとめて要約                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       Archival Memory (アーカイブメモリ)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 特徴: 長期保存用の検索可能ストレージ                                           │
│ 容量: 無制限（SQLiteバックエンド）                                            │
│                                                                              │
│ 検索:                                                                        │
│   - FTS5全文検索（利用可能時）                                                │
│   - キーワード検索（フォールバック）                                           │
│   - タグベース検索                                                            │
│                                                                              │
│ 典型的なタグ:                                                                │
│   - PHASE0_INTERNAL: 完全なPhase 0計画                                       │
│   - IDEA_MD: アイデアマークダウン全文                                         │
│   - PERFORMANCE: 性能関連の発見                                               │
│   - ERROR: エラーパターン                                                     │
│   - LLM_INSIGHT: LLMが記録した洞察                                           │
│                                                                              │
│ 注入:                                                                        │
│   - retrieval_k (デフォルト 8): 関連性上位k件をプロンプトに含める              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 分岐継承モデル

```
             ROOT (branch_id: 0)
            ┌─────────────────┐
            │ Core: {...}     │
            │ Recall: [e1,e2] │
            │ Archival: {...} │
            └────────┬────────┘
                     │ fork
        ┌────────────┴────────────┐
        │                         │
   Node A (branch_id: 1)     Node B (branch_id: 2)
  ┌─────────────────┐       ┌─────────────────┐
  │ Core: 継承+追加 │       │ Core: 継承+追加 │
  │ Recall: [e1,e2, │       │ Recall: [e1,e2, │
  │         e3_A]   │       │         e3_B]   │
  │ Archival: 検索可│       │ Archival: 検索可│
  └─────────────────┘       └─────────────────┘
        │                         │
        │ 書き込み隔離           │ 書き込み隔離
        │ (AはBの内容を見えない) │ (BはAの内容を見えない)
```

## 4. LLMメモリ操作 (LLM Memory Operations)

LLMは `<memory_update>` ブロックを使用してメモリを直接操作できます：

```json
<memory_update>
{
  "core": {
    "optimal_threads": "8",
    "best_compiler_flags": "-O3 -march=native"
  },
  "archival": [
    {
      "text": "8スレッドがこのワークロードで最適",
      "tags": ["PERFORMANCE", "THREADING"]
    }
  ],
  "archival_search": {
    "query": "compilation errors",
    "k": 3
  }
}
</memory_update>
```

### 操作種類

| 操作 | 対象 | 説明 |
|-----|------|------|
| `core` | Core | キー値設定 |
| `core_get` | Core | キー値取得 |
| `core_delete` | Core | キー削除 |
| `archival` | Archival | レコード追加 |
| `archival_search` | Archival | 検索 |
| `recall` | Recall | イベント追加 |
| `recall_search` | Recall | イベント検索 |
| `consolidate` | All | メモリ統合 |

## 5. リソースシステム (Resource System)

Injects external data, repositories, and models into prompts and containers:

```json
{
  "local": [
    {
      "name": "dataset",
      "host_path": "/shared/data",
      "mount_path": "/workspace/input/data",
      "read_only": true
    }
  ],
  "github": [
    {
      "name": "library",
      "repo": "https://github.com/org/lib.git",
      "dest": "/workspace/third_party/lib"
    }
  ],
  "items": [
    {
      "name": "template_code",
      "class": "template",
      "source": "local",
      "resource": "dataset",
      "path": "baseline",
      "include_files": ["main.c", "Makefile"]
    }
  ]
}
```

### Class-based Injection Rules

| Class | Phase 0 | Phase 1 | Phase 2 | Phase 3/4 |
|-------|---------|---------|---------|-----------|
| template | ✓ | ✓ | ✓ | - |
| document | ✓ | ✓ | ✓ | ✓ |
| setup | ✓ | ✓ | - | - |
| library | meta only | ✓ | - | - |
| dataset | meta only | ✓ | - | - |
| model | meta only | ✓ | - | - |

## 6. Persona System

Customizes the agent's role:

```yaml
agent:
  role_description: "HPC Researcher"
```

Effects:
- `{persona}` tokens are replaced with the configured role
- Applied recursively to all prompts

## 7. Post-Processing Pipeline

After Phase 4 execution, the following post-processing is automatically performed:

### VLM Analysis Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VLM Analysis Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Plot Code Generation                                                     │
│     └── LLM generates visualization code from .npy data                      │
│                                                                              │
│  2. Plot Execution                                                           │
│     └── Execute generated Python code to produce PNG images                  │
│                                                                              │
│  3. Image Encoding                                                           │
│     └── Base64 encode generated images                                       │
│                                                                              │
│  4. VLM Invocation (vlm/clients.py)                                          │
│     Input:                                                                   │
│       - Research idea text                                                   │
│       - Base64 encoded images                                                │
│       - VLM_ANALYSIS_PROMPT_TEMPLATE                                         │
│     Output:                                                                  │
│       - Image quality assessment                                             │
│       - Data visualization appropriateness                                   │
│       - Improvement suggestions                                              │
│                                                                              │
│  5. Result Storage                                                           │
│     └── Store in Node attributes:                                            │
│         - plot_analyses: Analysis results for each plot                      │
│         - vlm_feedback_summary: VLM feedback summary                         │
│         - datasets_successfully_tested: Successfully tested datasets         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Paper Generation Pipeline (Writeup Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Paper Generation Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Collect Experiment Summaries (load_exp_summaries)                        │
│     └── Collect results, metrics, and plots from each node                   │
│                                                                              │
│  2. Plot Aggregation (aggregate_plots)                                       │
│     └── LLM selects best plots and generates aggregation script              │
│     └── Improvement through reflection steps                                 │
│                                                                              │
│  3. Citation Gathering (gather_citations)                                    │
│     └── Uses Semantic Scholar API                                            │
│     └── Automatically search and collect related papers                      │
│                                                                              │
│  4. LaTeX Generation (perform_writeup)                                       │
│     Structure:                                                               │
│       - Abstract                                                             │
│       - Introduction                                                         │
│       - Method                                                               │
│       - Experiments                                                          │
│       - Conclusion                                                           │
│     └── Quality improvement through reflection steps                         │
│                                                                              │
│  5. PDF Generation (compile_latex)                                           │
│     └── Compile with pdflatex                                                │
│     └── Page limit check (detect_pages_before_impact)                        │
│                                                                              │
│  6. Review (Optional)                                                        │
│     └── LLM Review: NeurIPS-style evaluation                                 │
│     └── VLM Review: Figure quality assessment                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Extraction Flow

```
Execution Output (stdout/stderr)
        │
        ▼
   Parse Metrics Plan
   (LLM generates parser code)
        │
        ▼
   Execute Parser
        │
        ▼
   Store MetricValue
   - name: Metric name (e.g., "speedup")
   - value: Numeric value
   - direction: "higher_is_better" / "lower_is_better"
```

## 関連ドキュメント

- [glossary.md](glossary.md) - 用語集
- [workflow.md](workflow.md) - ワークフロー概要
- [../architecture/execution-flow.md](../architecture/execution-flow.md) - 標準実行フロー
- [../memory/memory.md](../memory/memory.md) - メモリシステム詳細
- [../architecture/resource-files.md](../architecture/resource-files.md) - リソースファイル詳細
