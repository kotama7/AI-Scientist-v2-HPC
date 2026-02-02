# Few-shot Example Customization

このドキュメントでは、論文レビューのfew-shot例をカスタマイズする方法を説明します。

## 概要

Few-shot例は、LLMに論文レビューの品質とスタイルを示すための参考例です。デフォルトではAI/機械学習論文の例が使用されていますが、HPC、物理学、生物学など他分野の論文に変更できます。

## Few-shot例の構造

各例には3つのファイルが必要です：

```
ai_scientist/fewshot_examples/
├── example_name.pdf      # 論文PDF
├── example_name.txt      # PDFから抽出したテキスト（オプション）
└── example_name.json     # レビュー結果
```

### JSONフォーマット

```json
{
    "review": "{\"Summary\": \"...\", \"Strengths\": [...], \"Weaknesses\": [...], \"Originality\": 4, ...}"
}
```

## 方法1: 自動生成スクリプトを使用（推奨）

### 基本的な使い方

```bash
# 単一のPDFから生成
python generate_fewshot_examples.py \
    --pdf papers/your_hpc_paper.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --model gpt-4o-2024-11-20
```

### HPC論文用にカスタマイズ

```bash
python generate_fewshot_examples.py \
    --pdf papers/himeno_benchmark.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --model gpt-4o-2024-11-20 \
    --custom-prompt "Review this HPC paper focusing on: 1) Parallel scalability (strong/weak scaling), 2) Performance analysis and bottlenecks, 3) Reproducibility of benchmark conditions, 4) Comparison with baseline implementations." \
    --num-reflections 5
```

### 複数のPDFから一括生成

```bash
# HPC論文のディレクトリから生成
for pdf in papers/hpc/*.pdf; do
    python generate_fewshot_examples.py \
        --pdf "$pdf" \
        --output-dir ai_scientist/fewshot_examples_hpc \
        --custom-prompt "Focus on HPC scalability, performance, and reproducibility"
done
```

### テキスト抽出のみ（レビュー無し）

```bash
# レビュー生成をスキップしてテキストのみ抽出
python generate_fewshot_examples.py \
    --pdf papers/example.pdf \
    --output-dir ai_scientist/fewshot_examples \
    --dry-run
```

## 方法2: 手動で作成

### ステップ1: PDFテキスト抽出

```python
from ai_scientist.review import load_paper

pdf_path = "papers/your_paper.pdf"
text = load_paper(pdf_path)

with open("ai_scientist/fewshot_examples/your_paper.txt", "w") as f:
    f.write(text)
```

### ステップ2: レビュー生成

```python
from ai_scientist.llm import create_client
from ai_scientist.review import perform_review
import json

client, model = create_client("gpt-4o-2024-11-20")
review = perform_review(
    text=text,
    model=model,
    client=client,
    num_reflections=3,
    num_fs_examples=0  # 既存の例を使わない
)

# JSONとして保存
review_wrapper = {"review": json.dumps(review, indent=4)}
with open("ai_scientist/fewshot_examples/your_paper.json", "w") as f:
    json.dump(review_wrapper, f, indent=4)
```

### ステップ3: PDFをコピー

```bash
cp papers/your_paper.pdf ai_scientist/fewshot_examples/
```

## 方法3: Few-shot例のリストを変更

生成したfew-shot例を使用するには、コード内のリストを変更します。

### 既存のリストを確認

[ai_scientist/review/llm_review.py:40-50](ai_scientist/review/llm_review.py#L40-L50)

```python
fewshot_papers = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(parent_dir, "fewshot_examples/attention.pdf"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(parent_dir, "fewshot_examples/132_automated_relational.json"),
    os.path.join(parent_dir, "fewshot_examples/attention.json"),
    os.path.join(parent_dir, "fewshot_examples/2_carpe_diem.json"),
]
```

### HPC例に置き換え

```python
# HPC論文の例に変更
fewshot_papers = [
    os.path.join(parent_dir, "fewshot_examples/himeno_benchmark.pdf"),
    os.path.join(parent_dir, "fewshot_examples/mpi_performance.pdf"),
    os.path.join(parent_dir, "fewshot_examples/gpu_acceleration.pdf"),
]

fewshot_reviews = [
    os.path.join(parent_dir, "fewshot_examples/himeno_benchmark.json"),
    os.path.join(parent_dir, "fewshot_examples/mpi_performance.json"),
    os.path.join(parent_dir, "fewshot_examples/gpu_acceleration.json"),
]
```

### 実行時に動的に指定（拡張版）

より柔軟な実装：

```python
# ai_scientist/review/llm_review.py を編集
import os

def get_fewshot_paths(domain="ai"):
    """Get few-shot example paths by domain.

    Args:
        domain: "ai", "hpc", "physics", "biology", etc.

    Returns:
        Tuple of (fewshot_papers, fewshot_reviews)
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    domain_examples = {
        "ai": [
            "132_automated_relational",
            "attention",
            "2_carpe_diem",
        ],
        "hpc": [
            "himeno_benchmark",
            "mpi_scalability",
            "gpu_acceleration",
        ],
    }

    examples = domain_examples.get(domain, domain_examples["ai"])

    fewshot_papers = [
        os.path.join(parent_dir, f"fewshot_examples/{ex}.pdf")
        for ex in examples
    ]
    fewshot_reviews = [
        os.path.join(parent_dir, f"fewshot_examples/{ex}.json")
        for ex in examples
    ]

    return fewshot_papers, fewshot_reviews

# デフォルトはAI
fewshot_papers, fewshot_reviews = get_fewshot_paths(
    domain=os.environ.get("REVIEW_DOMAIN", "ai")
)
```

使用時：
```bash
# HPCドメインでレビュー
REVIEW_DOMAIN=hpc python generate_paper.py --experiment-dir experiments/xxx
```

## HPC論文のfew-shot例作成のベストプラクティス

### 1. 適切な論文を選ぶ

良い例：
- ✅ 査読付き国際会議・ジャーナル（SC、ISC、TPDS等）
- ✅ 明確な性能評価とスケーラビリティ分析
- ✅ 再現性の高いベンチマーク
- ✅ 多様なトピック（MPI、GPU、ハイブリッド並列等）

避けるべき：
- ❌ プレプリントや未査読論文（レビュー品質が不明）
- ❌ 特殊すぎる環境・ハードウェアに依存
- ❌ 理論のみで実験がない論文

### 2. レビューの質を確認

自動生成後、以下を確認：
- **Strengths/Weaknesses**: HPC固有の観点（スケーラビリティ、性能解析）が含まれているか
- **Significance**: ベンチマーク結果の妥当性が評価されているか
- **Soundness**: 実験条件の再現性が考慮されているか

不足があれば、JSONを手動編集して品質を向上させます。

### 3. 多様性を確保

3つの例で異なる側面をカバー：
1. **並列アルゴリズム**: MPI、OpenMP、ハイブリッド
2. **アクセラレータ**: GPU、FPGA最適化
3. **ベンチマーク・性能解析**: 実世界アプリケーション

## トラブルシューティング

### PDF読み込みエラー

```
Error: Failed to extract text from PDF
```

**解決策**:
- PDFが破損していないか確認
- OCRが必要な画像PDFの場合、テキスト埋め込み版を使用
- 手動でテキストをコピーして`.txt`ファイルを作成

### レビュー品質が低い

```
生成されたレビューが浅い、または不正確
```

**解決策**:
- `--num-reflections` を5以上に増やす
- `--custom-prompt` でより具体的な指示を追加
- より高性能なモデル（`o1-preview-2024-09-12`等）を使用
- 生成後にJSONを手動編集

### Few-shot例が認識されない

```
FileNotFoundError: fewshot_examples/xxx.pdf
```

**解決策**:
- ファイル名が `llm_review.py` のリストと一致しているか確認
- 拡張子が正しいか確認（`.pdf`, `.json`）
- パスが相対パスとして正しいか確認

## 関連ファイル

- [generate_fewshot_examples.py](../../generate_fewshot_examples.py) - 自動生成スクリプト
- [ai_scientist/review/llm_review.py](../../ai_scientist/review/llm_review.py) - Few-shotリスト定義
- [ai_scientist/fewshot_examples/](../../ai_scientist/fewshot_examples/) - 既存の例

## 参考：HPC論文の推奨ソース

高品質なHPC論文の入手先：
- **SC (Supercomputing)**: https://sc23.supercomputing.org/proceedings/
- **ISC High Performance**: https://www.isc-hpc.com/
- **TPDS (IEEE Trans. on Parallel and Distributed Systems)**: https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=71
- **PPoPP (Principles and Practice of Parallel Programming)**: https://ppopp23.sigplan.org/
- **arXiv cs.DC**: https://arxiv.org/list/cs.DC/recent (査読前だが最新研究)
