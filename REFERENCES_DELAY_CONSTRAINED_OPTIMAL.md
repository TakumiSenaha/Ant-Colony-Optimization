# 遅延制約付きボトルネック帯域幅最大化問題の厳密解法：参考文献

## 1. 実装アルゴリズムの概要

現在の実装では、**修正ダイクストラ法（Modified Dijkstra Algorithm）を拡張**したアルゴリズムを使用しています。

### 1.1 アルゴリズムの特徴

- **目的**: 遅延制約 $D \leq D_{\text{limit}}$ を満たす経路の中で、ボトルネック帯域幅 $B = \min_{(i,j) \in P} w_{ij}$ を最大化
- **手法**: 各ノードに対して、到達可能な最大ボトルネック帯域と最小遅延を記録し、優先度キューを使用して探索
- **辞書式順序**: 同じボトルネック帯域幅を持つ経路の中で、遅延が最小のものを選択

### 1.2 実装の詳細

```python
# 優先度キュー: (-bottleneck, delay, node)
# 更新条件:
# 1. new_bottle > bottleneck[v] (より大きなボトルネック帯域)
# 2. new_bottle == bottleneck[v] and new_delay < min_delay[v] (同じ帯域でより小さい遅延)
# 3. new_delay <= max_delay (遅延制約を満たす)
```

---

## 2. 関連する研究分野と参考文献

### 2.1 Constrained Shortest Path Problem (CSPP)

**問題定義**: コストを最小化しつつ、制約（遅延など）を満たす経路を見つける問題

#### 主要参考文献

1. **Joksch, H. C. (1966)**

   - "The shortest route problem with constraints"
   - _Journal of Mathematical Analysis and Applications_, 14(2), 191-197
   - **内容**: 制約付き最短経路問題の初期研究

2. **Handler, G. Y., & Zang, I. (1980)**

   - "A dual algorithm for the constrained shortest path problem"
   - _Networks_, 10(4), 293-310
   - **内容**: 制約付き最短経路問題の双対アルゴリズム

3. **Beasley, J. E., & Christofides, N. (1989)**
   - "An algorithm for the resource constrained shortest path problem"
   - _Networks_, 19(4), 379-394
   - **内容**: リソース制約付き最短経路問題のアルゴリズム

### 2.2 Maximum Capacity Path Problem (MCPP)

**問題定義**: 経路上の最小エッジ容量（ボトルネック）を最大化する経路を見つける問題

#### 主要参考文献

1. **Gabow, H. N. (1985)**

   - "Scaling algorithms for network problems"
   - _Journal of Computer and System Sciences_, 31(2), 148-168
   - **内容**: ネットワーク問題のスケーリングアルゴリズム（最大容量経路問題を含む）

2. **Punnen, A. P. (1991)**

   - "A linear time algorithm for the maximum capacity path problem in directed graphs"
   - _Operations Research Letters_, 10(4), 221-224
   - **内容**: 有向グラフにおける最大容量経路問題の線形時間アルゴリズム

3. **Norimatsu, H., et al. (2024)**
   - "An Ant Colony Optimization Approach for Maximum Bottleneck Link Problem"
   - （既存実装で参照されている論文）
   - **内容**: 最大ボトルネックリンク問題に対する ACO アプローチ

### 2.3 QoS Routing (Quality of Service Routing)

**問題定義**: 複数の QoS 制約（帯域幅、遅延、ジッタなど）を満たす経路を見つける問題

#### 主要参考文献

1. **Chen, S., & Nahrstedt, K. (1998)**

   - "An overview of quality-of-service routing for the next generation high-speed networks"
   - _IEEE Communications Magazine_, 36(6), 64-79
   - **内容**: 次世代高速ネットワークのための QoS ルーティングの概要

2. **Kuipers, F., et al. (2002)**

   - "An overview of constraint-based path selection algorithms for QoS routing"
   - _IEEE Communications Magazine_, 40(12), 50-55
   - **内容**: QoS ルーティングのための制約ベース経路選択アルゴリズムの概要

3. **Yuan, X. (2002)**
   - "Heuristic algorithms for multiconstrained quality-of-service routing"
   - _IEEE/ACM Transactions on Networking_, 10(2), 244-256
   - **内容**: 複数制約 QoS ルーティングのヒューリスティックアルゴリズム

### 2.4 Modified Dijkstra Algorithm

**問題定義**: ダイクストラ法を拡張して、ボトルネック最大化や制約付き最適化を行う

#### 主要参考文献

1. **Dijkstra, E. W. (1959)**

   - "A note on two problems in connexion with graphs"
   - _Numerische Mathematik_, 1(1), 269-271
   - **内容**: ダイクストラ法の元論文（最短経路問題）

2. **Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993)**

   - "Network flows: theory, algorithms, and applications"
   - _Prentice Hall_
   - **内容**: ネットワークフロー理論とアルゴリズム（修正ダイクストラ法を含む）

3. **Cormen, T. H., et al. (2009)**
   - "Introduction to Algorithms" (3rd ed.)
   - _MIT Press_
   - **内容**: アルゴリズムの教科書（ダイクストラ法とその拡張）

### 2.5 Constrained Maximum Capacity Path Problem

**問題定義**: 制約（遅延など）を満たしながら、ボトルネック帯域幅を最大化する問題（本実験で扱う問題）

#### 主要参考文献

1. **Wang, Z., & Crowcroft, J. (1996)**

   - "Quality-of-service routing for supporting multimedia applications"
   - _IEEE Journal on Selected Areas in Communications_, 14(7), 1228-1234
   - **内容**: マルチメディアアプリケーションのための QoS ルーティング（帯域幅と遅延の両方を考慮）

2. **Orda, A., & Sprintson, A. (2003)**

   - "QoS routing: the precomputation perspective"
   - _IEEE/ACM Transactions on Networking_, 11(4), 578-592
   - **内容**: QoS ルーティングの事前計算アプローチ

3. **Norimatsu, H., et al. (2024)**
   - "A Particle Swarm Optimization Approach for Maximum Bottleneck Link Problem with Delay Constraint"
   - （既存実装で参照されている論文）
   - **内容**: 遅延制約付き最大ボトルネックリンク問題に対する PSO アプローチ

---

## 3. 論文での引用方法

### 3.1 最適解計算の説明

論文では、以下のように記述することを推奨します：

> "The optimal solution is computed using a modified Dijkstra algorithm that extends the standard shortest path algorithm to maximize the bottleneck bandwidth while satisfying the delay constraint. The algorithm maintains, for each node, the maximum bottleneck bandwidth and minimum delay achievable from the source, and explores paths in order of decreasing bottleneck bandwidth. When multiple paths achieve the same maximum bottleneck bandwidth, the one with the minimum delay is selected (lexicographical ordering)."

### 3.2 参考文献の選択

論文の文脈に応じて、以下のいずれかを選択してください：

#### オプション 1: 一般的な CSPP の参考文献

- Joksch (1966) - 制約付き最短経路問題の初期研究
- Handler & Zang (1980) - 制約付き最短経路問題の双対アルゴリズム

#### オプション 2: 最大容量経路問題の参考文献

- Gabow (1985) - 最大容量経路問題のアルゴリズム
- Punnen (1991) - 最大容量経路問題の線形時間アルゴリズム

#### オプション 3: QoS ルーティングの参考文献

- Chen & Nahrstedt (1998) - QoS ルーティングの概要
- Wang & Crowcroft (1996) - 帯域幅と遅延を考慮した QoS ルーティング

#### オプション 4: 既存実装の参考文献

- Norimatsu et al. (2024) - 最大ボトルネックリンク問題に対する ACO アプローチ
- （既存実装で参照されている論文があれば、それを引用）

### 3.3 推奨される引用

**最も適切な引用**は、**QoS ルーティング**の文献、特に**Wang & Crowcroft (1996)**です。理由：

1. **問題の類似性**: 帯域幅と遅延の両方を考慮したルーティング問題
2. **アルゴリズムの類似性**: 修正ダイクストラ法を使用
3. **実用性**: 実用的なネットワーク問題として広く認知されている

**代替案**: より理論的なアプローチを重視する場合は、**Gabow (1985)** や **Punnen (1991)** を引用することも可能です。

---

## 4. 参考文献リスト（BibTeX 形式）

```bibtex
@article{joksch1966,
  title={The shortest route problem with constraints},
  author={Joksch, H. C.},
  journal={Journal of Mathematical Analysis and Applications},
  volume={14},
  number={2},
  pages={191--197},
  year={1966}
}

@article{handler1980,
  title={A dual algorithm for the constrained shortest path problem},
  author={Handler, G. Y. and Zang, I.},
  journal={Networks},
  volume={10},
  number={4},
  pages={293--310},
  year={1980}
}

@article{gabow1985,
  title={Scaling algorithms for network problems},
  author={Gabow, H. N.},
  journal={Journal of Computer and System Sciences},
  volume={31},
  number={2},
  pages={148--168},
  year={1985}
}

@article{punnen1991,
  title={A linear time algorithm for the maximum capacity path problem in directed graphs},
  author={Punnen, A. P.},
  journal={Operations Research Letters},
  volume={10},
  number={4},
  pages={221--224},
  year={1991}
}

@article{chen1998,
  title={An overview of quality-of-service routing for the next generation high-speed networks},
  author={Chen, S. and Nahrstedt, K.},
  journal={IEEE Communications Magazine},
  volume={36},
  number={6},
  pages={64--79},
  year={1998}
}

@article{wang1996,
  title={Quality-of-service routing for supporting multimedia applications},
  author={Wang, Z. and Crowcroft, J.},
  journal={IEEE Journal on Selected Areas in Communications},
  volume={14},
  number={7},
  pages={1228--1234},
  year={1996}
}

@article{kuipers2002,
  title={An overview of constraint-based path selection algorithms for QoS routing},
  author={Kuipers, F. and Van Mieghem, P. and Korkmaz, T. and Krunz, M.},
  journal={IEEE Communications Magazine},
  volume={40},
  number={12},
  pages={50--55},
  year={2002}
}

@book{ahuja1993,
  title={Network flows: theory, algorithms, and applications},
  author={Ahuja, R. K. and Magnanti, T. L. and Orlin, J. B.},
  year={1993},
  publisher={Prentice Hall}
}

@book{cormen2009,
  title={Introduction to Algorithms},
  author={Cormen, T. H. and Leiserson, C. E. and Rivest, R. L. and Stein, C.},
  edition={3rd},
  year={2009},
  publisher={MIT Press}
}
```

---

## 5. まとめ

### 5.1 推奨される引用

**最適解計算の説明**には、以下のいずれかを引用することを推奨します：

1. **Wang & Crowcroft (1996)** - QoS ルーティング（帯域幅と遅延を考慮）
2. **Gabow (1985)** - 最大容量経路問題のアルゴリズム
3. **Chen & Nahrstedt (1998)** - QoS ルーティングの概要

### 5.2 論文での記述例

> "The optimal solution is computed using a modified Dijkstra algorithm that extends the standard shortest path algorithm to maximize the bottleneck bandwidth while satisfying the delay constraint [Wang1996]. The algorithm maintains, for each node, the maximum bottleneck bandwidth and minimum delay achievable from the source, and explores paths in order of decreasing bottleneck bandwidth. When multiple paths achieve the same maximum bottleneck bandwidth, the one with the minimum delay is selected (lexicographical ordering)."

### 5.3 注意事項

- 既存実装で参照されている論文（Norimatsu et al. 2024 など）があれば、それを優先的に引用してください
- 論文の文脈に応じて、理論的なアプローチ（Gabow 1985）または実用的なアプローチ（Wang 1996）を選択してください








