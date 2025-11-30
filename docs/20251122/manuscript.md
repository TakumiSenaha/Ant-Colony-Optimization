### 1. Background

First, the Background. Traditional IP networks have a problem: Traffic concentrates on specific servers.

To solve this, we focus on ICN (Information-Centric Networking).

In ICN, routers hold content caches.

We can get content from nearby routers instead of distant servers. But here is the problem: We don’t know where the content is, and the content moves. So, we need dynamic routing to track it.

### 2. Related Work: ACO

To solve this, we use ACO (Ant Colony Optimization).

This method models the foraging behavior of ants. Ants find food, and on their way back, they leave Pheromones (a scent).

Other ants then follow the strong Pheromones.

Old Pheromones evaporate over time.

This helps the system adapt when the path changes.

Basically, the idea is: 'We mark the good paths with Pheromones.

### Problem: Maximize Bottleneck

Conventional ACO usually focuses on delay or hop count.

It looks at the 'Total Sum' of weights. But our goal is different.

We want to discover the route with the maximum link bandwidth.

We want to maximize the 'Bottleneck Bandwidth'.

Unlike the sum, we look only at the 'Minimum Link' and try to maximize it.

Please look at the Green Path.

The narrowest part is 70. This is the best score among all paths.

### 4. Proposed Method: Autonomous Learning

We improved the ACO rules: Path Selection, Deposit, and Evaporation.

The key innovation is 'Autonomous Node Learning.'

When an ant finds content and returns, it tells the nodes: 'I found a path with THIS bandwidth.'

The node remembers this value as its Memory (or Score).

Using this memory, we control the pheromone:

- Good Path: If the new path is better than the node's memory, we add More pheromone.
- Bad Path: If the path is bad, we make it evaporate Fast. We make the node 'Forget' the bad path quickly.

Here is the logic: The memory represents the best bandwidth reached so far through this node.

If a connected edge is narrower than this memory, it cannot possibly be the optimal path.

Therefore, we make it evaporate significantly to avoid it."

### 5. Simulation Results

We tested this in a simulation. Look at Graph 1 (Static Environment).

Start and Goal are fixed.

You can see the selection rate goes up to 80%. The ants successfully find the optimal path.

Next, Graph 2. Here, the network bandwidth changes every generation (using an AR-1 model).

Even in this difficult situation, the rate stays around 60%. We achieved this high adaptability without increasing the calculation cost.
Thank you."

### SUPPLEMENTARY_SCRIPT

1. Background, ICN Details.

About ICN: "ICN is a revolutionary architecture that doesn't rely on TCP/IP. In this system, each router holds a cache. This happens when a router has previously fetched the content, or when content is intentionally distributed across the network."

2. Related Work, Problem Characteristics and ACO.

About the Bottleneck Problem: "A unique characteristic of the bottleneck maximization problem is that even if you choose a wide link now, you might be forced into a narrow path later. No matter how good the rest of the path is, if you pass through just one narrow link, the entire path's score is determined by that single link."

About ACO Mechanism: "ACO uses pheromones for probabilistic exploration combined with randomness. When ants find content, they deposit pheromones on their way back, and these pheromones evaporate over time. Repeating this finds the optimal path."

Why Randomness? "Randomness is crucial for two reasons: to prevent getting stuck in local optima, and because initially, we don't even know where the content is."

3. Proposed Method, Node Learning and Bonus.

About Node Learning: "Regarding node learning: When an ant returns, it notifies the node of the bottleneck value it experienced. The node learns, 'If you go through me, you can reach the goal with this bandwidth.' This tells the node that any connected link narrower than this value is essentially useless, allowing it to control pheromones effectively."

About the Bonus: "Regarding the bonus: To speed up convergence, if a path is found that is better than what is currently remembered, we proactively deposit more pheromones on that path."

4. Simulation, Model and Conditions.

About the BA Model: "We use the BA model, which creates a scale-free network. It can represent networks with 'popular' nodes, or hubs, similar to the real Internet."

About Environment 1, Static: "In Environment 1, the request node, provider node, and bandwidths are all fixed. Basically, we are testing if the method can purely find the single static optimal solution."

About Environment 2, Dynamic: "In Environment 2, the nodes are fixed, but the bandwidth fluctuates using a first-order autoregressive, or AR(1), model."

About Verification, Modified Dijkstra: "In both cases, we pre-calculate the true optimal solution using the Modified Dijkstra's Algorithm. We then verify if our routing matches this solution to calculate the selection rate."

About the Graph, Averaging: "To ensure accuracy, we run simulations on 100 completely different networks. The graph shows the average probability—out of these 100 runs—that the optimal path was selected in each generation."

---

### 1. 実装・コストに関する質問 (Implementation & Cost)

**Q1. 蟻（制御パケット）を増やすと、逆にネットワークが混雑しませんか？**
(Doesn't releasing so many ant packets cause network congestion?)

> **Answer:**
> "**That is a very important point.** (ごもっともな点です)
> Increasing control packets does consume some bandwidth.
> **However,** sending large video data over a _narrow_ path causes much worse congestion. Using a small amount of bandwidth for ants to find a _wide_ pipe prevents this major problem."
> （制御パケットは確かに帯域を使いますが、狭い道に大容量データを流して起こる致命的な輻輳を防ぐため、トータルではメリットがあります。）

**Q2. パケットヘッダが大きくなるのは問題ないですか？**
(Is the increase in packet header size a problem?)

> **Answer:**
> "**You raise a valid concern.** (ごもっともな懸念です)
> **But** the increase is only a few bytes for storing the integer bandwidth value.
> Compared to the size of video data, this overhead is negligible."
> （確かにヘッダは増えますが、たった数バイトの整数値です。動画データのサイズに比べれば無視できるレベルです。）

**Q3. 計算コスト（負荷）は増えませんか？**
(Does this increase the computational load on the routers?)

> **Answer:**
> "**That is true to some extent.** (ある程度はその通りです)
> **However,** the calculation is very simple: just updating a maximum value ($K_v$).
> It is much lighter than centralized algorithms like Dijkstra that calculate the entire network."
> （計算は増えますが、「最大値の更新」という単純なものなので、ネットワーク全体を計算するダイクストラ法などに比べれば遥かに軽量です。）

---

### 2. アルゴリズム・パラメータに関する質問 (Algorithm & Parameters)

**Q4. なぜリングバッファのサイズは「10」なのですか？**
(Why did you choose a Ring Buffer size of 10?)

> **Answer:**
> "**It is a trade-off between stability and speed.** (安定性と速度のトレードオフです)
> If it's too small, it overreacts to noise. If it's too large, it reacts too slowly.
> In our experiments, **10 was the best balance**."
> （小さすぎるとノイズに反応しすぎ、大きすぎると反応が遅れるため、実験的に 10 がベストバランスでした。）

**Q5. なぜランダム探索を「10%」も入れるのですか？**
(Why is the random exploration rate set to 10%?)

> **Answer:**
> "**I agree that it generates non-optimal traffic.** (非効率な通信を生むのは認めます)
> **But without this randomness,** ants would get stuck on old paths. In a dynamic environment where content moves, we _must_ explore new paths."
> （確かに無駄な通信ですが、これがないと古い道に固執してしまいます。動的環境では新しい道の開拓が必須です。）

**Q6. 帯域幅を最大化すると、逆に「遅延」が大きくなりませんか？**
(Doesn't maximizing bandwidth increase the delay?)

> **Answer:**
> "**That is absolutely correct.** (全くその通りです)
> Wide paths can be physically longer.
> That is exactly why our **Future Work** includes 'Multi-Objective Optimization' to balance both bandwidth and delay."
> （広い道は遠回りの可能性があります。だからこそ、今後の課題として帯域と遅延を両立させる「多目的最適化」を挙げています。）

---

### 3. 結果の評価に関する質問 (Evaluation)

**Q7. 動的環境での「60%」という正解率は低くないですか？**
(Is a 60% selection rate really sufficient?)

> **Answer:**
> "**I acknowledge that 60% might look low.** (低く見えることは認めます)
> **However,** please consider that the bandwidth changes **every single generation** in this simulation.
> Maintaining 60% in such a harsh environment proves the system is **highly adaptive**."
> （毎世代変動する過酷な環境で 60%を維持できていることは、システムが破綻せず高い適応性を持っていることの証明です。）

**Q8. なぜ AR モデル（自己回帰モデル）を使ったのですか？**
(Why did you use the AR model for fluctuation?)

> **Answer:**
> "**Because it mimics real network traffic.** (実際のトラフィックを模倣するためです)
> Real bandwidth doesn't change randomly from 0 to 100 in an instant. It has continuity. The AR model reproduces this realistic fluctuation."
> （現実の帯域は一瞬で 0 から 100 になったりはせず、連続性があります。AR モデルはそのリアルな変動を再現できるからです。）

---

### 4. 他の手法との比較 (Comparison with Other Methods)

**Q10. SDN（集中制御）を使えば、もっと簡単に最適経路が出せるのではありませんか？**
(Wouldn't it be easier to find the optimal path using a centralized controller like SDN?)

> **Answer:**
> "**That is a valid point.** SDN provides a global view.
> **However,** in a highly dynamic network where content moves frequently, the delay between the controller collecting data and calculating routes becomes a bottleneck.
> Our distributed method allows nodes to react **instantly** to local changes without waiting for a central server."
> （ごもっともです。SDN は全体を把握できます。しかし、コンテンツが頻繁に移動するような動的環境では、コントローラが情報を収集して計算するまでの「遅延」がボトルネックになります。分散型である本手法は、中央を待たずに局所的な変化に即座に反応できます。）

**Q11. 最近流行りの「深層強化学習（Deep RL）」は使わないのですか？**
(Why didn't you use Deep Reinforcement Learning?)

> **Answer:**
> "**That is an interesting alternative.** AI is powerful.
> **But** Deep Learning requires huge computational resources and a long training phase.
> Our ACO method is extremely **lightweight** and performs **'online learning'** (learns while running), making it more suitable for real-time routing on resource-constrained routers."
> （興味深い代替案です。AI は強力ですが、膨大な計算リソースと長い学習時間を必要とします。ACO は極めて軽量であり、走りながら学習する「オンライン学習」が可能なので、リソースの限られたルータには適しています。）

---

### 5. スケーラビリティ・拡張性 (Scalability)

**Q12. ノード数が 100 ではなく、1,000 や 10,000 になっても機能しますか？**
(Does this method scale to 1,000 or 10,000 nodes?)

> **Answer:**
> "**That is a critical question for practical application.**
> Since ACO is a distributed algorithm, it theoretically scales better than centralized ones.
> **However,** with more nodes, we might need to increase the number of ants or the pheromone lifetime to ensure convergence. We plan to test scalability in future simulations."
> （実用化において重要な質問です。ACO は分散型なので、理論上は集中型よりスケーラビリティがあります。ただ、ノード数が増えれば、収束させるために蟻の数やフェロモンの寿命を調整する必要があるでしょう。今後の検証課題です。）

**Q13. 「リングバッファ」の代わりに「指数移動平均（EMA）」を使わなかったのはなぜですか？**
(Why did you use a Ring Buffer instead of an Exponential Moving Average?)

> **Answer:**
> "**That is a technical trade-off.** EMA is memory-efficient.
> **However,** EMA never completely 'forgets' old values; they just get smaller.
> We wanted a mechanism that **completely discards** old information after $N$ steps to adapt quickly to sudden bandwidth drops. The Ring Buffer provides this hard cut-off."
> （技術的なトレードオフです。EMA はメモリ効率が良いですが、古い値を完全には「忘却」しません。我々は、急激な帯域低下に適応するため、N ステップ後に古い情報を「完全に破棄」するメカニズムを求めていました。リングバッファならそれが可能です。）

---

### 6. ICN 特有の質問 (ICN Specifics)

**Q14. キャッシュを持っているルータが複数ある場合、どうなりますか？**
(What happens if multiple routers hold the same cache?)

> **Answer:**
> "**The ants will naturally find the 'better' one.**
> ACO is probabilistic. If there are multiple sources, ants will explore paths to all of them. Eventually, more pheromones will accumulate on the path to the source that offers the **wider bottleneck bandwidth**, naturally guiding users to the better cache."
> （蟻は自然と「良い方」を見つけます。ACO は確率的なので、ソースが複数あれば蟻は両方を探索します。最終的に、より「ボトルネック帯域が広い」ソースへの経路に多くのフェロモンが蓄積され、ユーザを良いキャッシュへ誘導します。）

**Q15. 提案手法における「世代（Generation）」とは、現実時間でいうと何秒くらいですか？**
(What does one "Generation" represent in real time?)

> **Answer:**
> "**It depends on the traffic frequency.**
> One generation corresponds to one cycle of probing packets.
> If we send probes every 100 milliseconds, then 10 generations would be 1 second. The absolute time isn't fixed, but the _relative_ speed of adaptation is what matters."
> （トラフィックの頻度によります。1 世代は、探索パケットの 1 サイクルに相当します。もし 100ms ごとにプローブを送るなら、10 世代は 1 秒になります。絶対的な時間ではなく、適応の「相対的な速さ」が重要です。）

---

### 7. 弱点をつく質問 (Addressing Weaknesses)

**Q16. 最初の数世代はパフォーマンスが悪そうですが、どう対策しますか？**
(Performance seems poor in the first few generations. How do you handle that?)

> **Answer:**
> "**That is true.** ACO needs a 'warm-up' period to accumulate pheromones.
> **However,** in a real operation, we wouldn't start from zero. We could initialize the pheromones using static routing info (like Hop Count) to give it a head start."
> （その通りです。ACO はフェロモンが溜まるまでの「ウォームアップ」期間が必要です。しかし実際の運用ではゼロから始めず、静的ルーティング情報（ホップ数など）で初期値を設定して、スタートダッシュを切ることが可能です。）

**Q17. 蟻がループ（無限回廊）に陥ることはありませんか？**
(Do the ants ever get stuck in a loop?)

> **Answer:**
> "**We prevent that.**
> The ants carry a 'Tabu List' (memory of visited nodes) to prevent visiting the same node twice in one trip. This guarantees loop-free exploration."
> （それは防いでいます。蟻は「タブーリスト（訪問済みリスト）」を持っており、1 回の旅で同じノードを 2 度通らないようにしています。これによりループフリーな探索が保証されます。）

---

### 8. 安定性と振動（Stability & Oscillation）

**Q18. 全員が良い経路に殺到して、逆にその経路が混雑する「経路振動（Route Flapping）」は起きませんか？**
(Doesn't this cause "Route Flapping," where everyone rushes to the good path and congests it?)

> **Answer:**
> "**That is a classic routing problem.** (それは古典的なルーティングの問題ですね)
> If the algorithm were deterministic (like standard OSPF), everyone would switch at once, causing oscillation.
> **However,** ACO is **probabilistic**. Even if a path is good, some ants (and users) still choose other paths. This randomness acts as a damper, preventing sudden, synchronized switching."
> （もし決定的アルゴリズムなら全員が一斉に切り替えて振動します。しかし、ACO は「確率的」です。良い道であっても、一部の蟻（ユーザ）は他の道を選びます。この確率性がクッションとなり、急激な一斉切り替えを防ぎます。）

---

### 9. パラメータの妥当性（Parameter Sensitivity）

**Q19. ボーナス「1.5」やペナルティ「0.5」という数字はどうやって決めましたか？最適ですか？**
(How did you choose the parameters 1.5 and 0.5? Are they optimal?)

> **Answer:**
> "**They were chosen empirically.** (経験的に決定しました)
> We tested several values in preliminary experiments.
> If the bonus is too high, it converges too fast to a local optimum. If too low, it's too slow.
> 1.5 and 0.5 offered the best trade-off in this specific topology. Optimizing these parameters using Machine Learning is part of our **Future Work**."
> （予備実験でいくつかの値を試しました。ボーナスが高すぎると局所解への収束が早すぎ、低すぎると遅くなります。このトポロジでは 1.5 と 0.5 が最良のトレードオフでした。機械学習等でこれを自動最適化するのは今後の課題です。）

---

### 10. トラフィックモデルのリアリティ（Traffic Reality）

**Q20. AR モデルはなめらかすぎませんか？ 実際のネットはもっと「バースト（突発）的」ですよ。**
(Isn't the AR model too smooth? Real internet traffic is more "bursty".)

> **Answer:**
> "**You are right.** Real traffic has self-similarity and burstiness.
> The AR(1) model was a first step to test "continuous fluctuation."
> **However,** our method uses a **Ring Buffer**, which is designed to handle sudden changes (bursts) by discarding old data. So, we believe it will perform well even in bursty traffic. We plan to test this with real traffic traces."
> （その通りです。実際のトラフィックはバースト的です。AR モデルは「連続的な変動」をテストする第一歩でした。しかし、提案手法の「リングバッファ」は、古いデータを捨てて急激な変化（バースト）に対応するよう設計されています。ですので、バースト的な環境でも機能すると考えています。実トレースでの検証も計画しています。）

---

### 11. セキュリティ（Security）

**Q21. 悪意あるノードが「ここは帯域が広いぞ」と嘘をついたらどうなりますか？（ブラックホール攻撃）**
(What if a malicious node lies and reports a fake high bandwidth?)

> **Answer:**
> "**That is a security vulnerability.** (それはセキュリティ上の脆弱性です)
> Currently, we assume all nodes are honest.
> If a node lies, it could attract all traffic and drop it (Blackhole attack).
> To prevent this, we would need a **Trust Mechanism** where nodes verify each other's reports, but that is out of the scope of this routing research."
> （現在は全ノードが正直であると仮定しています。嘘をつかれるとトラフィックを吸い寄せられてしまいます。これを防ぐには、ノードがお互いの報告を検証する「信頼メカニズム」が必要ですが、それは本研究の範囲外です。）

### 12. そもそも論（Fundamental Questions）

**Q22. 動画配信なら、経路を変えるより「画質を落とす（ABR）」方が現実的ではないですか？**
(For video streaming, isn't Adaptive Bitrate (ABR) more practical than changing routes?)

> **Answer:**
> "**ABR is certainly the standard solution.** (ABR は確かに標準的な解決策です)
> **However,** ABR lowers the user experience (resolution).
> Our goal is to maintain **High Quality** by finding a better path _before_ lowering the quality. This routing method works **underneath** ABR to provide the best possible foundation."
> （ABR は画質を落として解決しますが、ユーザ体験は下がります。我々の目的は、画質を落とす「前に」良い道を見つけて、高品質を維持することです。このルーティングは、ABR の「下層」で動き、最良の土台を提供します。）

**Q23. ノードが学習する間、メモリ消費量は問題になりませんか？**
(Is memory consumption for node learning a concern?)

> **Answer:**
> "**Not really.**
> Even with a Ring Buffer of size 10, we only store 10 integers per destination flow.
> Modern routers have gigabytes of RAM. Storing a few kilobytes for routing tables is negligible compared to the router's capacity."
> （いいえ。リングバッファが 10 個でも、フローごとに整数を 10 個覚えるだけです。現代のルータは GB 単位のメモリを持っているので、ルーティングテーブルに数 KB 増える程度は無視できるレベルです。）

想定外の質問が来て答えられない時の対処法です。

- **"I haven't tested that specific scenario yet, but it is a very interesting direction for future work."**
  （その特定のシナリオはまだテストしていませんが、今後の研究として非常に興味深い方向性です。）

- **"Could you elaborate on that?"**
  （もう少し詳しく教えていただけますか？）

- **"That is out of the scope of this simulation, but theoretically..."**
  （シミュレーションの範囲外ですが、理論的には...）

---

### 1. アリに関する素朴な疑問 (About the Ants)

**Q1. 「コンピュータの中に本物のアリがいるんですか？」**
(Are there real ants in the network?)

> **Answer:**
> "**No, they are software agents.** (いいえ、ソフトウェアのエージェントです)
> They act like ants. Just like real ants find sugar without a map, our software ants find data without a central map."
> （本物ではなく、アリのように振る舞うプログラムです。本物のアリが地図なしで砂糖を見つけるように、このソフトも地図なしでデータを見つけます。）

**Q2. 「なぜアリなんですか？ AI じゃダメなんですか？」**
(Why ants? Why not AI?)

> **Answer:**
> "**Because ants are simple and fast.** (アリは単純で速いからです)
> AI needs a huge brain (GPU). Ants are tiny. They work well in thousands of small routers without using much power."
> （AI は巨大な頭脳（GPU）が必要ですが、アリはちっぽけです。何千もの小さなルータの中で、電力を使わずに働くにはアリが最適なんです。）

---

### 2. 「なぜこれが必要？」という疑問 (Why is this needed?)

**Q3. 「最短経路（一番近い道）を行けばいいんじゃないですか？」**
(Why don't you just take the shortest path?)

> **Answer:**
> "**Think of a highway vs. a narrow shortcut.** (高速道路と、狭い近道を想像してください)
> The shortcut is shorter, but it gets jammed easily.
> For watching videos, a **wide highway** is better, even if it's a bit longer. That's what we look for."
> （近道は距離は短いですが、すぐ渋滞します。動画を見るなら、少し遠回りでも「道幅の広い高速道路」の方がいいですよね。我々はそれを探しているんです。）

**Q4. 「『コンテンツが移動する』ってどういうことですか？」**
(What do you mean by "content moves"?)

> **Answer:**
> "**Imagine watching a movie from a moving car.** (走っている車から映画を見る場面を想像してください)
> Or, the video you are watching is stored on another person's smartphone, and they are walking away.
> The source of the video is moving, so the path must change instantly."
> （例えば、走っている車からデータをもらう場合や、あなたが探しているデータが誰かの歩いているスマホの中にある場合です。データの場所が動くので、道順もすぐに変えないといけないんです。）

---

### 3. 仕組みについての疑問 (How it works)

**Q5. 「フェロモンって、コンピュータのどこにあるんですか？」**
(Where is the "pheromone" in the computer?)

> **Answer:**
> "**It's just a number stored in the router's memory.** (ルータのメモリにあるただの「数字」です)
> A high number means 'Good Path.' A low number means 'Bad Path.'
> We just call it 'pheromone' because it fades away over time like a scent."
> （高い数字なら「良い道」、低いなら「悪い道」です。匂いのように時間が経つと消えていくので、かっこよく「フェロモン」と呼んでいるだけです。）

**Q6. 「もしアリが道に迷ったらどうなりますか？」**
(What happens if the ants get lost?)

> **Answer:**
> "**They just die (disappear).** (死んで消えるだけです)
> That's the beauty of it. We send many ants. Even if some get lost, others will find the goal. It doesn't break the system."
> （そのまま消えてなくなるだけです。そこが良いところで、たくさんアリを放つので、何匹か迷っても他がゴールすれば問題ありません。システムは壊れないんです。）

---

難しい説明になりそうなときに、これを使うと空気が和らぎます。

- **"It works just like Google Maps, but for moving targets."**
  （動くターゲット版の Google マップみたいなものです。）
- **"It's like finding the widest water pipe instead of the shortest one."**
  （一番短いパイプじゃなくて、一番太いパイプを探すようなものです。）
- **"Imagine thousands of tiny robots exploring the network."**
  （何千もの小さなロボットがネットワークを探検していると想像してください。）

### ネットワーク帯域幅変動メカニズム

### Network Bandwidth Fluctuation Mechanism

本シミュレーションでは、**AR(1)モデル（1 次自己回帰モデル）**を用いて、ネットワークエッジの帯域幅を動的に変動させています。
In this simulation, the bandwidth of network edges is dynamically fluctuated using an **AR(1) Model (First-Order Autoregressive Model)**.

AR(1)モデルは時系列データの変動をモデル化する統計的手法で、ネットワークトラフィックの現実的な変動パターンを再現するために選択されました。
The AR(1) model is a statistical method for modeling time series data fluctuations, selected to replicate realistic network traffic fluctuation patterns.

---

### 1. AR(1)モデルによる利用率の更新

### 1. Utilization Update via AR(1) Model

#### 数学的定式化

#### Mathematical Formulation

各エッジの利用率は以下の式で更新されます：
The utilization rate of each edge is updated by the following formula:

```
利用率(t+1) = (1 - φ) × 平均利用率 + φ × 利用率(t) + ε(t)
           = 0.02 + 0.95 × 利用率(t) + ε(t)
```

```
Utilization(t+1) = (1 - φ) × Mean Utilization + φ × Utilization(t) + ε(t)
           = 0.02 + 0.95 × Utilization(t) + ε(t)
```

ここで、`ε(t)` は時刻 `t` におけるランダムノイズ項です。
Here, `ε(t)` is the random noise term at time `t`.

#### パラメータの詳細

#### Parameter Details

**平均利用率（Mean Utilization）**: 0.4（40%）
**Mean Utilization**: 0.4 (40%)

- **選択根拠**: ISP（インターネットサービスプロバイダ）の一般的な運用マージンに基づく
- **Rationale**: Based on typical operational margins of ISPs (Internet Service Providers)
- 実際のネットワークでは、常に 100%の帯域を使用することはなく、余裕を持たせて運用される
- In real networks, bandwidth is not used at 100% capacity; operational margins are maintained
- 40%の平均利用率は、60%の可用帯域を確保することを意味する
- A 40% mean utilization means 60% available bandwidth is maintained

**自己相関係数（φ, phi）**: 0.95
**Autocorrelation Coefficient (φ, phi)**: 0.95

- **選択根拠**: ネットワークトラフィックの高い自己相関特性に基づく
- **Rationale**: Based on the high autocorrelation characteristics of network traffic
- 直前の値に 95%依存するため、急激な変化は起こりにくい
- Since it depends 95% on the previous value, abrupt changes are unlikely
- 現実のネットワークトラフィックは時間的に連続的で、急激な変動は稀である
- Real network traffic is temporally continuous, with rare abrupt fluctuations
- 半減期（影響が半分になるまでの時間）: 約 14 世代
- Half-life (time for influence to halve): Approximately 14 generations

**ノイズ分散（Noise Variance）**: 0.000975
**Noise Variance**: 0.000975

- **標準偏差**: √0.000975 ≈ 0.0312（約 3.12%）
- **Standard Deviation**: √0.000975 ≈ 0.0312 (approx. 3.12%)
- **生成方法**: 正規分布（ガウス分布）N(0, 0.000975) から生成
- **Generation Method**: Generated from a Normal (Gaussian) distribution N(0, 0.000975)
- **選択根拠**: 平均利用率と自己相関係数から逆算された値
- **Rationale**: Calculated inversely from mean utilization and autocorrelation coefficient
- 95%信頼区間: 約 ±6.3%の変動範囲
- 95% confidence interval: Approximately ±6.3% fluctuation range

#### AR(1)モデルの特性

#### AR(1) Model Characteristics

**定常性（Stationarity）**:

- AR(1)モデルは定常過程であり、長期的には平均値に収束する
- The AR(1) model is a stationary process that converges to the mean in the long term
- 自己相関係数 |φ| < 1 のため、システムは安定している
- Since |φ| < 1, the system is stable

**平均回帰性（Mean Reversion）**:

- 長期的には平均利用率 40%に収束する
- Converges to a 40% mean utilization in the long term
- 収束速度: 約 20 世代で 63%収束（1 - 1/e ≈ 0.632）
- Convergence speed: Approximately 63% convergence in 20 generations (1 - 1/e ≈ 0.632)
- 極端な状態（高帯域・低帯域）が長期間持続しない
- Extreme states (very high or very low bandwidth) do not persist for long periods

**自己相関構造（Autocorrelation Structure）**:

- 1 次ラグの自己相関: 0.95
- First-order lag autocorrelation: 0.95
- k 次ラグの自己相関: 0.95^k
- k-th order lag autocorrelation: 0.95^k
- 例: 10 世代前の値との相関は約 0.60（0.95^10）
- Example: Correlation with value 10 generations ago is approximately 0.60 (0.95^10)

---

### 2. 利用率から可用帯域への変換

### 2. Conversion from Utilization to Available Bandwidth

#### 変換式

#### Conversion Formula

```
可用帯域(t) = キャパシティ × (1 - 利用率(t))
```

```
Available Bandwidth(t) = Capacity × (1 - Utilization(t))
```

#### 実装の詳細

#### Implementation Details

**10Mbps 刻みの丸め処理**:
**Rounding to 10Mbps increments**:

計算された可用帯域は、実装上の都合により 10Mbps 刻みに丸められます：
The calculated available bandwidth is rounded to 10Mbps increments for implementation purposes:

```python
available_bandwidth = ((int(available_bandwidth) + 5) // 10) * 10
```

この処理により、帯域幅の値が離散化され、より現実的なネットワーク設定に近づきます。
This processing discretizes bandwidth values, making them closer to realistic network settings.

**双方向エッジの扱い**:
**Bidirectional Edge Handling**:

- 各エッジ (u, v) に対して、双方向 (u, v) と (v, u) の両方が独立して変動する
- For each edge (u, v), both directions (u, v) and (v, u) fluctuate independently
- 各方向に独立した利用率状態が保持される
- Independent utilization states are maintained for each direction
- 初期化時には、各方向の利用率は 0.3 ～ 0.5 の範囲でランダムに設定される
- During initialization, utilization for each direction is randomly set in the range 0.3-0.5

#### 計算例

#### Calculation Example

**例 1: 標準的なケース**
**Example 1: Standard Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.4（40%）
- Utilization: 0.4 (40%)
- 可用帯域: 100 × (1 - 0.4) = 60Mbps
- Available Bandwidth: 100 × (1 - 0.4) = 60Mbps

**例 2: 高負荷ケース**
**Example 2: High Load Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.7（70%）
- Utilization: 0.7 (70%)
- 可用帯域: 100 × (1 - 0.7) = 30Mbps
- Available Bandwidth: 100 × (1 - 0.7) = 30Mbps

**例 3: 低負荷ケース**
**Example 3: Low Load Case**

- キャパシティ: 100Mbps
- Capacity: 100Mbps
- 利用率: 0.2（20%）
- Utilization: 0.2 (20%)
- 可用帯域: 100 × (1 - 0.2) = 80Mbps
- Available Bandwidth: 100 × (1 - 0.2) = 80Mbps

---

### 3. ノイズの役割と特性

### 3. Role and Characteristics of Noise

#### ノイズとは？

#### What is Noise?

**ノイズ**は、予測不可能なランダムな変動成分です。
**Noise** is an unpredictable, random fluctuation component.

#### 統計的特性

#### Statistical Properties

- **分布**: 正規分布（ガウス分布）N(0, σ²)
- **Distribution**: Normal (Gaussian) distribution N(0, σ²)
- **平均**: 0（ゼロ平均）
- **Mean**: 0 (zero mean)
- **分散**: 0.000975
- **Variance**: 0.000975
- **標準偏差**: √0.000975 ≈ 0.0312（約 3.12%）
- **Standard Deviation**: √0.000975 ≈ 0.0312 (approx. 3.12%)

#### ノイズの役割

#### Role of Noise

1. **完全な予測を防ぐ**
1. **Prevents perfect predictability**

   - ノイズ項により、将来の帯域幅を完全に予測することは不可能
   - The noise term makes it impossible to perfectly predict future bandwidth
   - アルゴリズムは常に探索と適応を続ける必要がある
   - Algorithms must continuously explore and adapt

1. **探索の必要性を保証**
1. **Ensures the necessity of exploration**

   - 確率的変動により、最適解が時間とともに変化する可能性がある
   - Probabilistic fluctuations mean optimal solutions may change over time
   - 探索戦略の重要性が高まる
   - The importance of exploration strategies increases

1. **現実的なネットワークトラフィックの再現**
1. **Replicates realistic network traffic**
   - 実際のネットワークでは、予期しないトラフィック変動が発生する
   - Real networks experience unexpected traffic fluctuations
   - ノイズ項は、このような予測困難な変動をモデル化する
   - The noise term models such unpredictable fluctuations

#### ノイズの影響範囲

#### Noise Impact Range

- **68%信頼区間**: ±3.12%（1 標準偏差）
- **68% Confidence Interval**: ±3.12% (1 standard deviation)
- **95%信頼区間**: ±6.24%（2 標準偏差）
- **95% Confidence Interval**: ±6.24% (2 standard deviations)
- **99.7%信頼区間**: ±9.36%（3 標準偏差）
- **99.7% Confidence Interval**: ±9.36% (3 standard deviations)

---

## 変動の特徴

## Characteristics of Fluctuation

### 1. 高い自己相関（φ = 0.95）

### 1. High Autocorrelation (φ = 0.95)

#### 自己相関の意味

#### Meaning of Autocorrelation

- 現在の値は直前の値に 95%依存している
- The current value is 95% dependent on the previous value
- 急激な変化は起こりにくい（現実のネットワークトラフィックに近い）
- Abrupt changes are unlikely (similar to realistic network traffic)
- 時間的に連続的な変動パターンを示す
- Shows temporally continuous fluctuation patterns

#### 半減期の計算

#### Half-Life Calculation

半減期（影響が半分になるまでの時間）は以下の式で計算されます：
The half-life (time for influence to halve) is calculated by:

```
半減期 = ln(0.5) / ln(φ) ≈ 13.5世代
```

```
Half-life = ln(0.5) / ln(φ) ≈ 13.5 generations
```

実用的には約 14 世代で、直前の値の影響が半分になります。
Practically, the influence of the previous value halves in approximately 14 generations.

#### 自己相関の減衰

#### Autocorrelation Decay

- 1 世代前: 0.95
- 1 generation ago: 0.95
- 5 世代前: 0.95^5 ≈ 0.77
- 5 generations ago: 0.95^5 ≈ 0.77
- 10 世代前: 0.95^10 ≈ 0.60
- 10 generations ago: 0.95^10 ≈ 0.60
- 20 世代前: 0.95^20 ≈ 0.36
- 20 generations ago: 0.95^20 ≈ 0.36

---

### 2. 平均回帰性

### 2. Mean Reversion

#### 平均回帰のメカニズム

#### Mean Reversion Mechanism

AR(1)モデルは、以下の項により平均値への回帰が実現されます：
The AR(1) model achieves mean reversion through the following term:

```
(1 - φ) × 平均利用率 = 0.05 × 0.4 = 0.02
```

```
(1 - φ) × Mean Utilization = 0.05 × 0.4 = 0.02
```

この項により、利用率が平均値から離れると、自動的に平均値に戻る力が働きます。
This term creates a force that automatically returns utilization to the mean when it deviates.

#### 収束速度

#### Convergence Speed

- **時定数（Time Constant）**: 1 / (1 - φ) = 1 / 0.05 = 20 世代
- **Time Constant**: 1 / (1 - φ) = 1 / 0.05 = 20 generations
- **63%収束**: 約 20 世代で初期偏差の 63%が解消される
- **63% Convergence**: Approximately 63% of initial deviation is resolved in 20 generations
- **95%収束**: 約 60 世代で初期偏差の 95%が解消される
- **95% Convergence**: Approximately 95% of initial deviation is resolved in 60 generations

#### 極端な状態の持続時間

#### Duration of Extreme States

- 極端な状態（高帯域・低帯域）が長期間持続しない
- Extreme states (very high or very low bandwidth) do not persist for long periods
- 平均回帰により、長期的には平均利用率 40%に収束する
- Mean reversion causes long-term convergence to 40% mean utilization

---

### 3. 確率的変動

### 3. Stochastic Fluctuation

#### 確率性の重要性

#### Importance of Stochasticity

- ノイズ項により完全な予測は不可能
- Perfect prediction is impossible due to the noise term
- 探索の必要性を保証する
- This guarantees the necessity of exploration
- アルゴリズムの適応能力が試される
- Tests the adaptive capabilities of algorithms

#### 変動の予測可能性

#### Predictability of Fluctuations

- **短期予測**: 高い自己相関により、1 ～ 2 世代先の予測は比較的容易
- **Short-term Prediction**: High autocorrelation makes 1-2 generation ahead predictions relatively easy
- **中期予測**: 5 ～ 10 世代先の予測は、ノイズの累積により不確実性が増す
- **Medium-term Prediction**: 5-10 generation ahead predictions become more uncertain due to noise accumulation
- **長期予測**: 20 世代以上先の予測は、平均値への収束のみが確実
- **Long-term Prediction**: For 20+ generations ahead, only convergence to the mean is certain

---

### 4. 変動範囲の制限

### 4. Fluctuation Range Limit

#### クリッピング処理

#### Clipping Process

利用率は **0.05 ～ 0.95** の範囲にクリップされます：
The utilization rate is **clipped** to the range [0.05, 0.95]:

```python
new_utilization = max(0.05, min(0.95, new_utilization))
```

#### クリッピングの理由

#### Reasons for Clipping

1. **現実的な範囲の維持**
1. **Maintaining realistic ranges**

   - 0%や 100%の利用率は現実的ではない
   - 0% or 100% utilization is unrealistic
   - 常に最小 5%の可用帯域、最大 95%の利用率を保証
   - Always guarantees minimum 5% available bandwidth, maximum 95% utilization

1. **数値的安定性**
1. **Numerical stability**

   - 極端な値による数値計算の不安定性を防ぐ
   - Prevents numerical instability from extreme values
   - アルゴリズムの収束性を保証
   - Guarantees algorithm convergence

1. **シミュレーションの妥当性**
1. **Simulation validity**
   - 現実的なネットワーク条件を反映
   - Reflects realistic network conditions
   - テストケースの妥当性を維持
   - Maintains test case validity

#### クリッピングの影響

#### Impact of Clipping

- 利用率が 0.05 未満になった場合: 0.05 にクリップされ、可用帯域は最大 95%になる
- When utilization falls below 0.05: Clipped to 0.05, available bandwidth becomes maximum 95%
- 利用率が 0.95 を超えた場合: 0.95 にクリップされ、可用帯域は最小 5%になる
- When utilization exceeds 0.95: Clipped to 0.95, available bandwidth becomes minimum 5%

---

## 変動のタイミング

## Fluctuation Timing

### 更新間隔

### Update Interval

- **更新間隔**: `BANDWIDTH_UPDATE_INTERVAL` 世代ごと
- **Update Interval**: Every `BANDWIDTH_UPDATE_INTERVAL` generations
- **現在の設定**: 10000 世代ごと（実質的にほぼ固定）
- **Current Setting**: Every 10,000 generations (i.e., virtually static)

### 更新条件

### Update Condition

帯域幅の更新は、以下の条件を満たす世代でのみ実行されます：
Bandwidth updates are executed only in generations that meet the following condition:

```python
if generation % BANDWIDTH_UPDATE_INTERVAL != 0:
    return False  # 更新なし
```

### 更新間隔の選択

### Update Interval Selection

**現在の設定（10000 世代）**:
**Current Setting (10,000 generations)**:

- 実質的に帯域幅は固定されている
- Bandwidth is effectively fixed
- 静的環境でのアルゴリズム性能を評価するために使用
- Used to evaluate algorithm performance in static environments
- 変動の影響を最小限に抑える
- Minimizes the impact of fluctuations

**動的環境の設定例**:
**Dynamic Environment Setting Examples**:

- `BANDWIDTH_UPDATE_INTERVAL = 1`: 毎世代更新（最も動的）
- `BANDWIDTH_UPDATE_INTERVAL = 1`: Update every generation (most dynamic)
- `BANDWIDTH_UPDATE_INTERVAL = 10`: 10 世代ごと更新
- `BANDWIDTH_UPDATE_INTERVAL = 10`: Update every 10 generations
- `BANDWIDTH_UPDATE_INTERVAL = 100`: 100 世代ごと更新
- `BANDWIDTH_UPDATE_INTERVAL = 100`: Update every 100 generations

---

## 変動対象エッジ

## Target Edges for Fluctuation

### 選択の原則

### Selection Principle

**全エッジではなく、選択されたエッジのみが変動**
**Fluctuation is applied only to selected edges, not all edges**

この設計により、以下の利点があります：
This design provides the following benefits:

1. **計算効率**: 全エッジを更新する必要がない
1. **Computational Efficiency**: No need to update all edges
1. **現実性**: 実際のネットワークでは、すべてのリンクが同時に変動するわけではない
1. **Realism**: In real networks, not all links fluctuate simultaneously
1. **テストの焦点**: 重要なエッジ（ハブノードなど）に焦点を当てる
1. **Test Focus**: Focus on important edges (e.g., hub nodes)

### デフォルト設定

### Default Setting

- **デフォルト設定**: ハブノード（次数の高いノード）に接続するエッジの約 10%
- **Default Setting**: Approx. 10% of edges connected to hub nodes (nodes with high degree)
- **設定パラメータ**: `FLUCTUATION_PERCENTAGE = 0.1`
- **Configuration Parameter**: `FLUCTUATION_PERCENTAGE = 0.1`

### エッジ選択方法

### Edge Selection Methods

選択方法は `EDGE_SELECTION_METHOD` で指定可能です：
The selection method can be specified via `EDGE_SELECTION_METHOD`:

#### 1. ハブノード選択（`"hub"`）- 推奨

#### 1. Hub Node Selection (`"hub"`) - Recommended

- **説明**: 次数の高いノード（ハブノード）に接続するエッジを選択
- **Description**: Selects edges connected to nodes with high degree (hub nodes)
- **選択プロセス**:
- **Selection Process**:

  1. 全ノードの次数（隣接ノード数）を計算
  1. Calculate degree (number of adjacent nodes) for all nodes
  1. 次数の高い順にソート
  1. Sort by degree in descending order
  1. 上位 10%のノードをハブノードとして選択
  1. Select top 10% of nodes as hub nodes
  1. ハブノードに接続するすべてのエッジを変動対象とする
  1. All edges connected to hub nodes become fluctuation targets

- **利点**: ネットワークの重要な部分（ハブ）に焦点を当てる
- **Advantage**: Focuses on important parts of the network (hubs)
- **現実性**: 実際のネットワークでは、ハブノード周辺のトラフィック変動が大きい
- **Realism**: In real networks, traffic fluctuations around hub nodes are significant

#### 2. 完全ランダム選択（`"random"`）

#### 2. Completely Random Selection (`"random"`)

- **説明**: 全エッジからランダムに 10%を選択
- **Description**: Randomly selects 10% of all edges
- **選択タイミング**: `RANDOM_SELECTION_TIMING` で指定可能
- **Selection Timing**: Can be specified via `RANDOM_SELECTION_TIMING`
  - `"fixed"`: シミュレーション開始時に固定（再現可能）
  - `"fixed"`: Fixed at simulation start (reproducible)
  - `"dynamic"`: 毎回ランダムに選択
  - `"dynamic"`: Randomly selected each time

#### 3. 媒介中心性選択（`"betweenness"`）

#### 3. Betweenness Centrality Selection (`"betweenness"`)

- **説明**: エッジ媒介中心性（Edge Betweenness Centrality）が高いエッジを選択
- **Description**: Selects edges with high Edge Betweenness Centrality
- **計算方法**: NetworkX の `edge_betweenness_centrality()` 関数を使用
- **Calculation Method**: Uses NetworkX's `edge_betweenness_centrality()` function
- **利点**: 多くの最短経路が通過する重要なエッジに焦点を当てる
- **Advantage**: Focuses on important edges that many shortest paths traverse

#### 4. 柔軟なハブノード選択（`"hub_partial"`, `"hub_random"`）

#### 4. Flexible Hub Node Selection (`"hub_partial"`, `"hub_random"`)

- **説明**: ハブノードの隣接エッジを部分的に選択
- **Description**: Partially selects adjacent edges of hub nodes
- **パラメータ**: `HUB_NEIGHBOR_EDGE_RATIO` で隣接エッジの選択割合を指定
- **Parameter**: `HUB_NEIGHBOR_EDGE_RATIO` specifies the selection ratio of adjacent edges
- **選択方法**: `HUB_NEIGHBOR_SELECTION_METHOD` で指定
- **Selection Method**: Specified via `HUB_NEIGHBOR_SELECTION_METHOD`
  - `"degree"`: 次数順で選択
  - `"degree"`: Select by degree order
  - `"random"`: ランダムに選択
  - `"random"`: Random selection

### 選択エッジの初期化

### Initialization of Selected Edges

選択されたエッジの初期利用率は、以下のように設定されます：
The initial utilization of selected edges is set as follows:

```python
util_uv = random.uniform(0.3, 0.5)  # エッジ (u, v) の方向
util_vu = random.uniform(0.3, 0.5)  # エッジ (v, u) の方向
```

- 各方向の利用率は独立して 0.3 ～ 0.5 の範囲でランダムに設定される
- Utilization for each direction is independently set randomly in the range 0.3-0.5
- 初期状態から平均利用率（0.4）に近い値から開始する
- Starts from values close to the mean utilization (0.4) from the initial state

---

## 実装の詳細

## Implementation Details

### 実装ファイル

### Implementation Files

#### 設定ファイル

#### Configuration File

- **ファイル**: `src/bandwidth_fluctuation_config.py`
- **File**: `src/bandwidth_fluctuation_config.py`
- **役割**: 帯域変動の全パラメータを一元管理
- **Role**: Centralized management of all bandwidth fluctuation parameters
- **主要パラメータ**:
- **Main Parameters**:
  - `BANDWIDTH_UPDATE_INTERVAL`: 更新間隔（デフォルト: 10000）
  - `BANDWIDTH_UPDATE_INTERVAL`: Update interval (default: 10000)
  - `EDGE_SELECTION_METHOD`: エッジ選択方法（デフォルト: "hub"）
  - `EDGE_SELECTION_METHOD`: Edge selection method (default: "hub")
  - `FLUCTUATION_PERCENTAGE`: 変動対象エッジの割合（デフォルト: 0.1）
  - `FLUCTUATION_PERCENTAGE`: Ratio of fluctuating edges (default: 0.1)
  - `MEAN_UTILIZATION`: 平均利用率（デフォルト: 0.4）
  - `MEAN_UTILIZATION`: Mean utilization (default: 0.4)
  - `AR_COEFFICIENT`: 自己相関係数（デフォルト: 0.95）
  - `AR_COEFFICIENT`: Autocorrelation coefficient (default: 0.95)
  - `NOISE_VARIANCE`: ノイズ分散（デフォルト: 0.000975）
  - `NOISE_VARIANCE`: Noise variance (default: 0.000975)

#### メイン実装関数

#### Main Implementation Functions

**初期化関数**:
**Initialization Function**:

```python
def initialize_ar1_states(
    graph: nx.Graph,
    fluctuating_edges: List[Tuple[int, int]] | None = None
) -> Dict[Tuple[int, int], float]:
```

- 選択されたエッジの AR(1)状態を初期化
- Initializes AR(1) states for selected edges
- 各エッジの初期利用率を 0.3 ～ 0.5 の範囲でランダムに設定
- Randomly sets initial utilization for each edge in the range 0.3-0.5
- 初期可用帯域を計算してグラフに設定
- Calculates and sets initial available bandwidth in the graph

**更新関数**:
**Update Function**:

```python
def update_available_bandwidth_ar1(
    graph: nx.Graph,
    edge_states: Dict[Tuple[int, int], Dict],
    generation: int
) -> bool:
```

- AR(1)モデルに基づいて帯域幅を更新
- Updates bandwidth based on AR(1) model
- `BANDWIDTH_UPDATE_INTERVAL` 世代ごとにのみ実行
- Executes only every `BANDWIDTH_UPDATE_INTERVAL` generations
- 更新された場合は `True` を返す
- Returns `True` if updated

#### モジュール実装（新規）

#### Module Implementation (New)

- **ファイル**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **File**: `aco_moo_routing/src/aco_routing/modules/bandwidth_fluctuation.py`
- **クラス**: `AR1Model`
- **Class**: `AR1Model`
- **役割**: オブジェクト指向的な実装で、複数の変動モデルに対応
- **Role**: Object-oriented implementation supporting multiple fluctuation models

### 実装の流れ

### Implementation Flow

1. **初期化フェーズ**
1. **Initialization Phase**

   - エッジ選択関数を呼び出して変動対象エッジを決定
   - Call edge selection function to determine fluctuating edges
   - `initialize_ar1_states()` で各エッジの初期状態を設定
   - Set initial state for each edge with `initialize_ar1_states()`
   - グラフの `weight` 属性に初期可用帯域を設定
   - Set initial available bandwidth in graph's `weight` attribute

1. **更新フェーズ（各世代）**
1. **Update Phase (Each Generation)**

   - `update_available_bandwidth_ar1()` を呼び出す
   - Call `update_available_bandwidth_ar1()`
   - 更新間隔をチェック（`generation % BANDWIDTH_UPDATE_INTERVAL == 0`）
   - Check update interval (`generation % BANDWIDTH_UPDATE_INTERVAL == 0`)
   - 条件を満たす場合、AR(1)モデルで利用率を更新
   - If condition is met, update utilization with AR(1) model
   - 可用帯域を再計算してグラフを更新
   - Recalculate available bandwidth and update graph

1. **状態管理**
1. **State Management**

   - 各エッジの利用率状態は `edge_states` 辞書で管理
   - Utilization state for each edge is managed in `edge_states` dictionary
   - キー: `(u, v)` タプル（エッジの方向）
   - Key: `(u, v)` tuple (edge direction)
   - 値: `{"utilization": float}` 辞書
   - Value: `{"utilization": float}` dictionary

### グラフ属性の更新

### Graph Attribute Updates

更新時に以下のグラフ属性が更新されます：
The following graph attributes are updated during updates:

- `graph[u][v]["weight"]`: 可用帯域（10Mbps 刻み）
- `graph[u][v]["weight"]`: Available bandwidth (rounded to 10Mbps)
- `graph[u][v]["local_min_bandwidth"]`: 現在の可用帯域（最小値として）
- `graph[u][v]["local_min_bandwidth"]`: Current available bandwidth (as minimum)
- `graph[u][v]["local_max_bandwidth"]`: 現在の可用帯域（最大値として）
- `graph[u][v]["local_max_bandwidth"]`: Current available bandwidth (as maximum)
- `graph[u][v]["original_weight"]`: 元のキャパシティ（変更されない）
- `graph[u][v]["original_weight"]`: Original capacity (unchanged)

---

## 統計的特性

## Statistical Properties

### 定常分布

### Stationary Distribution

長期的には、利用率は以下の正規分布に従います：
In the long term, utilization follows the following normal distribution:

```
利用率 ~ N(μ, σ²/(1 - φ²))
      ~ N(0.4, 0.000975/(1 - 0.95²))
      ~ N(0.4, 0.01026)
```

```
Utilization ~ N(μ, σ²/(1 - φ²))
           ~ N(0.4, 0.000975/(1 - 0.95²))
           ~ N(0.4, 0.01026)
```

- **平均**: 0.4（40%）
- **Mean**: 0.4 (40%)
- **標準偏差**: √0.01026 ≈ 0.101（約 10.1%）
- **Standard Deviation**: √0.01026 ≈ 0.101 (approx. 10.1%)

### 変動の大きさ

### Magnitude of Fluctuation

- **短期変動（1 世代）**: ノイズの標準偏差 ≈ ±3.12%
- **Short-term Fluctuation (1 generation)**: Noise standard deviation ≈ ±3.12%
- **長期変動（定常分布）**: 標準偏差 ≈ ±10.1%
- **Long-term Fluctuation (stationary distribution)**: Standard deviation ≈ ±10.1%
- **95%信頼区間**: 約 0.2 ～ 0.6（20%～ 60%の利用率）
- **95% Confidence Interval**: Approximately 0.2-0.6 (20%-60% utilization)

### 自己共分散関数

### Autocovariance Function

k 次ラグの自己共分散は以下の式で与えられます：
The autocovariance at lag k is given by:

```
γ(k) = σ² × φ^k / (1 - φ²)
     = 0.000975 × 0.95^k / (1 - 0.95²)
```

```
γ(k) = σ² × φ^k / (1 - φ²)
     = 0.000975 × 0.95^k / (1 - 0.95²)
```

- **k=0（分散）**: ≈ 0.01026
- **k=0 (variance)**: ≈ 0.01026
- **k=1（1 次ラグ）**: ≈ 0.00975
- **k=1 (first-order lag)**: ≈ 0.00975
- **k=10（10 次ラグ）**: ≈ 0.00588
- **k=10 (10th-order lag)**: ≈ 0.00588

---

## 簡潔な説明（質問されたとき用）

## Concise Explanation (For Q&A)

### Q: このネットワークはどのように変動するか？

### Q: How does this network fluctuate?

**A**: AR(1)モデル（1 次自己回帰モデル）により、各エッジの利用率を更新し、それを可用帯域に変換して変動させます。
**A**: It uses an AR(1) model to update the "utilization rate" of each edge, which is then converted into "available bandwidth."

#### 1. 利用率の更新

#### 1. Utilization Update

```
利用率(t+1) = 0.02 + 0.95 × 利用率(t) + ノイズ
```

```
Utilization(t+1) = 0.02 + 0.95 × Utilization(t) + Noise
```

- 直前の値に 95%依存（高い自己相関）
- Highly autocorrelated (95% dependent on the previous value)
  - ランダムノイズで予測不能な変動も含む
- Includes random noise for unpredictable changes
- 長期的には平均利用率 40%に収束
- Reverts to a 40% mean utilization over the long term

#### 2. 可用帯域の計算

#### 2. Available Bandwidth Calculation

```
可用帯域 = キャパシティ × (1 - 利用率)
```

```
Available Bandwidth = Capacity × (1 - Utilization)
```

- 利用率が上がる → 可用帯域が減る
- Higher utilization → Lower available bandwidth
  - 利用率が下がる → 可用帯域が増える
- Lower utilization → Higher available bandwidth

#### 3. 変動タイミング

#### 3. Fluctuation Timing

- **現在の設定**: 10000 世代ごと（実質的にほぼ固定）
- **Current Setting**: Every 10,000 generations (i.e., virtually static)
- **動的環境の例**: 1 世代ごと、10 世代ごとなど
- **Dynamic Environment Examples**: Every generation, every 10 generations, etc.

#### 4. 変動対象

#### 4. Fluctuation Target

- **デフォルト**: 全エッジの約 10%（ハブノードに接続するエッジ）
- **Default**: Approx. 10% of all edges (connected to hub nodes)
- **選択方法**: ハブノード選択、ランダム選択、媒介中心性選択など
- **Selection Methods**: Hub node selection, random selection, betweenness centrality selection, etc.

---
