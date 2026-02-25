# System Knowledge vs Agent Behavior 分析ガイド

## 🎯 目的

**「エージェントは確率的に探索しているため成功率が変動するが、システム内部（フェロモン分布）は早期に確実に最適解を学習している」**ことを視覚的に証明する。

---

## 📊 グラフの構成

### 2つの系列

#### 1. **Agent Behavior（青破線）** - 確率的な挙動
- **データソース**: `generation_stats.csv`の`optimal_count / num_ants`
- **意味**: 実際にその世代で最適解を通ったアリの割合
- **特性**: 
  - ε=0.1のランダム探索の影響で変動
  - 100%には到達しない（意図的な探索を継続）
  - 探索の多様性を維持

#### 2. **System Knowledge（赤実線）** - 決定論的な収束
- **データソース**: `generation_stats.csv`の`interest_hit`
- **意味**: フェロモン最大のエッジだけを貪欲に選んだ結果が最適解だったか（0 or 1）
- **特性**:
  - 早期に100%に収束（システムが最適解を特定）
  - 収束後は100%を維持（知識の確実性）
  - 探索の影響を受けない

---

## 🔍 論文での議論の展開

### 1. 背景（問題提起）

> **ε-greedy法などの確率的探索を用いる場合、エージェントの挙動（青い破線）だけを見ていては、システムがいつ最適解を学習し終えたのか正確に判断できない。**

### 2. 事実（グラフの観察）

> **図6を見ると、青い破線（Agent Behavior）は探索の影響で約80-90%付近で振動しており、100%には到達しない。しかし、フェロモン最大のエッジを貪欲に選ぶ赤い実線（System Knowledge）に着目すると、約112世代付近で最適解到達率が1.0（100%）に張り付いていることがわかる。**

### 3. 洞察（Insight）

> **この乖離は重要である。システムは112世代という早期の段階で、内部的には最適解を『完全に特定』している。それ以降の青線のゆらぎは、知識不足による迷いではなく、解を確定させた上で、さらなる環境変化に備えて意図的に行っている探索（Monitoring）の結果であると解釈できる。**

### 4. 結論（優位性）

> **つまり、提案手法は『探索の多様性（Behaviorの分散）』を維持しつつ、『知識の確実性（Knowledgeの収束）』を早期に確立できる、ロバストな学習特性を持っていることが実証された。**

---

## 💡 期待される結果

### Strict制約環境（5ms）
```
System Knowledge: 約20-30世代で100%収束
Agent Behavior:   約60-70%で安定（探索の影響）
収束世代:        非常に早い（解空間が狭い）
```

### Manual環境（静的）
```
System Knowledge: 約100-150世代で100%収束
Agent Behavior:   約80-85%で安定
収束世代:        中程度
```

### Dynamic環境（帯域変動）
```
System Knowledge: 変動に応じて95-100%で推移
Agent Behavior:   約75-85%で推移（より多くの探索）
収束世代:        変動により再学習が発生
```

---

## 🚀 実行手順

### Step 1: 実験を実行（simulations: 100推奨）

```bash
cd /Users/asaken_n47/Documents/aco/Ant-Colony-Optimization/aco_moo_routing

# config.yamlを編集:
# - method: "proposed"
# - graph_type: "manual"（または他の環境）
# - simulations: 100  # 複数回実行で統計的に有意な結果
# - generations: 1000

python experiments/run_experiment.py
```

### Step 2: グラフを生成

#### 単一環境の分析

```bash
# Manual環境の例
python analysis/plot_knowledge_vs_behavior.py \
  --results-dir results/proposed/manual/bandwidth_only \
  --num-ants 10 \
  --output results/analysis/manual_knowledge_vs_behavior

# 遅延制約環境の例（Unique Optimalを使用）
python analysis/plot_knowledge_vs_behavior.py \
  --results-dir results/proposed/delay_constraint_10ms/delay_constraint \
  --num-ants 10 \
  --use-unique \
  --output results/analysis/delay_10ms_knowledge_vs_behavior
```

#### 複数環境の比較

```bash
# Manual, Static, Dynamicの3環境を比較
python analysis/plot_knowledge_vs_behavior.py \
  --results-dirs \
    results/proposed/manual/bandwidth_only \
    results/proposed/static/bandwidth_only \
    results/proposed/bandwidth_fluctuation/bandwidth_only \
  --labels "Manual" "Static" "Dynamic" \
  --num-ants 10 \
  --output results/analysis/knowledge_vs_behavior_comparison
```

---

## 📈 グラフの解釈ポイント

### 1. **収束世代の特定**

赤実線が95%を超えた世代 = システムが最適解を学習した世代

```
例: Manual環境で112世代
→ 「提案手法は112世代で最適解を特定した」
```

### 2. **Agent BehaviorとSystem Knowledgeの差**

```
System: 100%（完全な知識）
Agent:  85%（ε=0.1の探索）
差:     15% = 意図的な探索の割合
```

### 3. **Dynamic環境での適応性**

```
帯域変動時: System Knowledgeが一時的に低下
→ 変化を検知し、再学習
→ 数世代で再収束
```

---

## 📝 generation_stats.csvの列の説明

| 列名 | 説明 | 用途 |
|------|------|------|
| generation | 世代番号 | X軸 |
| optimal_count | 最適解を発見したアリの数 | Agent Behavior（Any Optimal） |
| unique_optimal_count | 一意な最適解を発見したアリの数 | Agent Behavior（Unique Optimal） |
| interest_hit | フェロモン貪欲解が最適解だったか（0/1） | System Knowledge |
| num_ants_reached | ゴールに到達したアリの数 | 参考情報 |
| avg_bandwidth | 平均ボトルネック帯域 | 参考情報 |

---

## 🎓 理論的背景

### ε-Greedy法の特性

```python
if random.random() < ε:  # 10%の確率
    # 探索（Exploration）: ランダム選択
    next_node = random.choice(candidates)
else:  # 90%の確率
    # 活用（Exploitation）: フェロモンとヒューリスティックで選択
    next_node = probabilistic_selection(...)
```

**Agent Behaviorが100%にならない理由**:
- 10%の確率でランダム選択
- 90%の確率でも確率的選択（必ずしも最良を選ぶとは限らない）
- → 理論上の上限: 約90-95%

**System Knowledgeが100%になる理由**:
- 完全に貪欲（Greedy）: フェロモン最大のエッジを確定的に選択
- 探索の影響を受けない
- フェロモン分布が最適解に収束すれば、常に最適解を選択

---

## 📚 関連ドキュメント

1. **plot_knowledge_vs_behavior.py** - グラフ生成スクリプト
2. **compare_delay_constraint.py** - 遅延制約環境の比較
3. **FINAL_FIX_SUMMARY.md** - 全修正のまとめ

---

## 🎯 論文での活用例

### Figure 6のキャプション

```latex
\begin{figure}[tb]
 \centering
 \includegraphics[width=\columnwidth]{figures/06_knowledge_vs_behavior.eps}
 \caption{Convergence of probabilistic agent behavior vs. deterministic system knowledge.
 The red solid line represents the selection rate of the optimal path using deterministic 
 routing (max-pheromone selection), indicating the system's internal knowledge. 
 The blue dashed line represents the actual selection rate of agents operating under 
 an ε-greedy policy (ε=0.1). The system achieves near-perfect knowledge convergence 
 at generation 112, while agents maintain exploration diversity with 85\% exploitation.}
 \label{fig:knowledge_vs_behavior}
\end{figure}
```

### 本文での言及

```latex
\subsection{フェロモン分布に基づく内部知識の収束}

図\ref{fig:knowledge_vs_behavior}に、エージェントの確率的挙動とシステムの決定論的知識の収束を示す。
赤い実線（System Knowledge）は、各世代でフェロモン量が最大のエッジのみを貪欲に選択した場合の
最適解到達率を表し、青い破線（Agent Behavior）は、実際にε-greedy法（ε=0.1）で探索を行った
エージェントの最適解到達率を表す。

図から、System Knowledgeは約112世代で100\%に到達し、その後も維持されている。
一方、Agent Behaviorは約85\%付近で安定しており、100\%には到達しない。
この15\%の差は、ε=0.1のランダム探索と確率的選択による意図的な探索の結果である。

この結果は、提案手法が早期にシステム内部の知識を確立しつつ、エージェントの探索多様性を
維持できることを示している。すなわち、112世代以降の青線の変動は知識不足による迷いではなく、
確立された知識を基に、環境変化への適応のために意図的に行っている探索（Monitoring）の
結果であると解釈できる。
```

---

## 🧪 拡張アイデア

### 1. 収束世代のアノテーション追加

グラフに収束世代を矢印で示す：

```python
if conv_gen is not None:
    ax.annotate(
        f'Convergence\nat gen. {conv_gen}',
        xy=(conv_gen, 100),
        xytext=(conv_gen + 100, 90),
        arrowprops=dict(arrowstyle='->', lw=2, color='red'),
        fontsize=14,
        ha='left'
    )
```

### 2. 乖離率の時系列表示

System KnowledgeとAgent Behaviorの差を別のサブプロットで表示：

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# ax1: 元のグラフ
# ax2: 乖離率（System - Agent）
divergence = [(s - a) * 100 for s, a in zip(system_mean, agent_mean)]
ax2.plot(generations, divergence, color='purple', linewidth=2)
ax2.set_ylabel("Knowledge-Behavior Gap [%]")
```

### 3. 動的環境での再学習の可視化

帯域変動のタイミングを縦線で表示：

```python
# 帯域変動が発生した世代を取得（別途ログから）
bandwidth_change_gens = [100, 200, 300, ...]
for gen in bandwidth_change_gens:
    ax.axvline(x=gen, color='gray', linestyle=':', alpha=0.5)
```


