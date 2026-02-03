# 実装サマリー: 論文メモリ生成機能の改善
## 日付: 2026-02-03

## 概要

このドキュメントは、"Not found in memory"問題を解決し、一貫した高品質な論文生成出力を保証するために、論文メモリ生成システムに加えられた主要な改善をまとめたものです。

## 解決された問題

### 1. ハードウェア環境情報の欠落

**元の問題**:
- 生成された論文でCPU、OS、コンパイラ、ツール情報が"Not found in memory"と表示される
- PHASE0_INTERNALエントリはアーカイブメモリに存在するがLLMから見えない
- アーカイブ内の位置: 96-297番目（全2,555エントリ中）
- LLMは作成日時順の上位30件のみを参照

**根本原因**:
- アーカイブエントリが`created_at DESC`（新しい順）でソート
- PHASE0_INTERNALは実験初期（Phase 0）に作成
- 多数の新しいエントリ（統合、削除）がPHASE0_INTERNALを30位以降に押し出す
- `_prepare_memory_summary()`は最初の30件のみをLLMに渡す

**実装した解決策**:
1. **優先タグシステム** (`memgpt_store.py:5054-5089`)
   - 優先タグを定義: `PHASE0_INTERNAL`, `IDEA_MD`, `ROOT_IDEA`
   - アーカイブエントリを優先/非優先に分離
   - 優先エントリを常に上位30枠に含める
   - 残りの枠を最新の非優先エントリで埋める

2. **ハードウェア情報の自動抽出・注入** (`memgpt_store.py:3977-4102`, `5327`)
   - 正規表現を使用してPHASE0_INTERNALエントリからハードウェア情報を抽出
   - Core Memoryに注入: `hardware_cpu`, `hardware_os`, `hardware_compiler`, `hardware_tools`
   - `generate_final_memory_for_paper()`で自動実行
   - 4つのキーがすべて存在する場合は抽出をスキップ（効率化）

**結果**:
- ハードウェア情報がすべての生成論文に一貫して表示される
- "Not found in memory"が9件以上から3件（予想される）に削減
- `launch_scientist_bfts.py`と`regenerate_memory_env.sh`の両方のパスで動作

### 2. VLMフィードバックサマリーの表示エラー

**元の問題**:
- VLMフィードバックサマリーが1文字ずつの番号付きリストとして表示される
- 例: "1. A", "2. c", "3. r", "4. o", "5. s", "6. s", ...

**根本原因**:
- `vlm_feedback_summary`がノードデータに文字列として保存されている
- コードはリストとして想定してイテレート
- 文字列をイテレートすると個々の文字をループする

**実装した解決策** (`memgpt_store.py:5642-5651`, `5701-5709`):
```python
if isinstance(vlm_feedback, str):
    md_sections.append(vlm_feedback)  # 段落として表示
elif isinstance(vlm_feedback, list):
    for i, feedback in enumerate(vlm_feedback, 1):
        md_sections.append(f"{i}. {feedback}")  # 番号付きリストとして表示
```

**結果**:
- VLMフィードバックサマリーが一貫した段落として正しく表示される
- 文字列とリスト形式の両方を適切に処理

### 3. 実行パス間の実装の不一致

**元の問題**:
- `launch_scientist_bfts.py`と`regenerate_memory_env.sh`で異なるコードパス
- `regenerate_memory_env.sh`に独自のハードウェア抽出コードが存在
- 初回生成と再生成で結果が異なる可能性

**実装した解決策**:
- ハードウェア抽出ロジックを`memgpt_store.py`の共有メソッドに移動
- 両パスともに同じ実装で`generate_final_memory_for_paper()`を呼び出す
- `regenerate_memory_env.sh`から重複コードを削除

**結果**:
- 両実行パスで同一の出力を生成
- メモリ生成ロジックの単一の情報源
- 保守とデバッグが容易

## 実装詳細

### コード変更

#### ファイル: `ai_scientist/memory/memgpt_store.py`

**追加された新規メソッド**:
```python
# 行 ~3977-4040
def _extract_hardware_info_from_archival(self, branch_id: str) -> dict[str, str | None]:
    """正規表現を使用してPHASE0_INTERNALエントリからハードウェア情報を抽出。"""
    # 返り値: cpu_model, cpu_sockets, cpu_cores, numa_nodes, os, compiler, compiler_version, tools

# 行 ~4041-4102
def _inject_hardware_info_to_core(self, branch_id: str) -> None:
    """ハードウェア情報を抽出してCore Memoryに注入。"""
    # 既存をチェック、欠落している場合は抽出、Core Memoryに注入
```

**変更されたメソッド**:
```python
# 行 ~5054-5089 (_prepare_memory_summary内)
# 優先タグシステムの実装
priority_tags = {"PHASE0_INTERNAL", "IDEA_MD", "ROOT_IDEA"}
# 優先/非優先エントリを分離
# 上位30件を選択: 優先が最初、次に最新

# 行 ~5327 (generate_final_memory_for_paper内)
self._inject_hardware_info_to_core(root_branch_id)  # セクション生成前に自動注入

# 行 ~5642-5651, 5701-5709
# VLMフィードバックサマリーの型処理
if isinstance(vlm_feedback, str):
    # 文字列形式を処理
elif isinstance(vlm_feedback, list):
    # リスト形式を処理
```

#### ファイル: `regenerate_memory_env.sh`

**削除されたコード**:
- 227-261行: ハードウェア情報の抽出と注入（memgpt_store.pyに移動）

**簡素化されたコード**:
```bash
# ハードウェア情報の抽出と注入は自動的に処理される
# generate_paper_memory_from_manager() -> generate_final_memory_for_paper()
# _inject_hardware_info_to_core()メソッド経由
print("\nGenerating final memory for paper...")
print("(Hardware info will be automatically extracted and injected)\n")
```

### データフロー

```
generate_final_memory_for_paper()
    │
    ├─ _inject_hardware_info_to_core(root_branch_id)
    │   ├─ 既存のハードウェア情報をCore Memoryでチェック
    │   └─ 欠落している場合:
    │       ├─ _extract_hardware_info_from_archival()
    │       │   └─ PHASE0_INTERNALエントリをクエリ（最大20件）
    │       └─ 各ハードウェアフィールドに対してset_core()
    │
    ├─ _build_paper_sections()
    │   ├─ 3層メモリを収集（Core、Recall、Archival）
    │   ├─ _prepare_memory_summary()
    │   │   ├─ 優先タグシステム: エントリを分離
    │   │   └─ 上位30件を選択: 優先 + 最新
    │   └─ _generate_sections_with_llm()
    │       └─ LLMが参照: Core Memory + 上位30件のArchival
    │
    └─ マークダウン/JSON出力を生成
```

## テスト結果

### 実装前
```
"Not found in memory"の出現: 9件以上
- CPU情報: Not found
- OS情報: Not found
- コンパイラ情報: Not found
- ツール情報: Not found
- VLMフィードバック: 1文字ずつ表示
- パスが異なる: launch_scientist_bfts.py ≠ regenerate_memory_env.sh
```

### 実装後
```
"Not found in memory"の出現: 3件（予想される）
✅ CPU情報: AMD EPYC 9554, 2 socket(s), 128 cores, 2 NUMA node(s)
✅ OS情報: Ubuntu 22.04
✅ コンパイラ情報: gcc 11.4.0
✅ ツール情報: numactl, perf, taskset, hwloc-ls, lscpu, numastat
✅ VLMフィードバック: 一貫した段落表示
✅ パスが同一: launch_scientist_bfts.py == regenerate_memory_env.sh

残っている"Not found"（予想される）:
- OpenMP runtimeバージョン（Phase 0で記録されていない）
- 具体的なスレッド数セット（PHASE0_INTERNALに列挙されていない）
- バックグラウンド負荷制御の詳細（機能が実装されていない）
```

### 検証コマンド
```bash
# 再生成して結果を確認
cd /home/users/takanori.kotama/workplace
./regenerate_memory_env.sh ~/workplace/AI-Scientist-v2-HPC/experiments/<experiment_dir> 0-run

# "Not found"の出現回数をカウント
grep -c "Not found" <experiment_dir>/0-run/memory/final_memory_for_paper.md
# 期待値: 3

# Core Memoryのハードウェア情報を確認
sqlite3 <experiment_dir>/0-run/memory/memory.sqlite \
  "SELECT key, value FROM core_kv WHERE key LIKE 'hardware%';"
# 期待値: 4行（hardware_cpu, hardware_os, hardware_compiler, hardware_tools）
```

## 影響評価

### パフォーマンス
- **メモリ取得**: 大きな変更なし（依然として200件のアーカイブエントリを取得）
- **抽出オーバーヘッド**: 最小限（20件のPHASE0_INTERNALエントリ、キャッシュされた正規表現パターン）
- **LLMコスト**: 変更なし（依然として30件のアーカイブエントリをLLMに渡す）

### 信頼性
- **実装前**: 一貫性のない結果、重要な情報の欠落
- **実装後**: 一貫した再現可能な結果、完全なハードウェア情報

### 保守性
- **実装前**: 複数ファイルに重複ロジック
- **実装後**: `memgpt_store.py`に単一の情報源

### 後方互換性
- ✅ 破壊的変更なし
- ✅ 既存のメモリデータベースは移行なしで動作
- ✅ 古い実験を新しいコードで再生成可能

## 今後の改善

### 潜在的な強化
1. **設定可能な優先タグ**: ユーザーが設定でカスタム優先タグを指定可能に
2. **スマートなアーカイブ制限**: 優先コンテンツに基づいて30エントリ制限を動的調整
3. **強化された正規表現パターン**: より多くのハードウェア/コンパイラ形式をサポート
4. **検証警告**: 期待されるPHASE0_INTERNAL情報が欠落している場合にログ記録
5. **ハードウェア情報のエクスポート**: 外部ツール用に`final_writeup_memory.json`に含める

### 既知の制限
1. 正規表現パターンがすべてのハードウェア記述形式に一致しない可能性
2. 優先システムは合計30エントリに制限（LLMコンテキストサイズとのトレードオフ）
3. ハードウェア情報抽出にはPHASE0_INTERNALエントリの存在が必要
4. `vlm_feedback_summary`の型処理は文字列またはリストを想定（辞書/その他は非対応）

## ドキュメント更新

### 新規ファイル
- `docs/memory/hardware-info-injection.md`: 自動ハードウェア情報抽出の詳細ガイド
- `docs/memory/IMPLEMENTATION_SUMMARY_20260203.md`: 英語版実装サマリー
- `docs/memory/IMPLEMENTATION_SUMMARY_20260203_ja.md`: 日本語版実装サマリー（このドキュメント）

### 更新されたファイル
- `docs/memory/memory-for-paper.md`: 実装詳細、解決された問題、変更履歴を追加

### 関連ドキュメント
- [memory-for-paper.md](memory-for-paper.md) - 論文メモリ生成の概要
- [hardware-info-injection.md](hardware-info-injection.md) - ハードウェア抽出の実装
- [memory.md](memory.md) - メモリシステムアーキテクチャ
- [memgpt-implementation.md](memgpt-implementation.md) - MemGPT実装の詳細

## 貢献者

- 実装: Claude Code (AIアシスタント)
- テスト: 実験実行 2026-01-30_16-22-06_stability_oriented_autotuning_v2_attempt_0
- ドキュメント: このサマリー

---

**最終更新**: 2026-02-03
**バージョン**: 1.0
**ステータス**: ✅ 本番環境対応
