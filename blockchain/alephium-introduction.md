# alephium introduction

区块链的可扩展性问题的解决方案基本上可以划分为两类，分片和分层，其中分片算法的着重点主要在于:

* 保证跨分片交易的原子性
* 保存在单个分片上的数据也需要满足可扩展性，理想状态下单个分片不需要保存其他分片的数据
* 分片安全性不会随着分片数量的增多而降低

同时在设计分片算法时可能还需要兼顾考虑共识算法(PoW or PoS)，本文介绍[alephium](https://alephium.org/)的BlockFlow分片算法，基于PoW和UTXO模型。主要内容分为原理和实现两部分，原理部分主要来源于[alephium论文](https://github.com/alephium/white-paper/blob/master/alephium.pdf)，实现部分主要参考alephium的[scala版本的实现](https://github.com/alephium/alephium).

## principle

### basic idea

alephium将用户划分为`G`个group，对于其中的任意两个group `i`和`j`，通过一条单独的分片来记录所有从`i`中的用户发往到`j`中的用户的交易。因此alephium一共由`G * G`个分片组成，即`G * G`条链。不同分片的交易可以并行提交并执行，同时需要满足下面两个条件:

* 对于group `i`, `j`和`k`，从`i`到`j`的链(即 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> )中所有的交易输入都来自于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}"> 的未花费的交易输出(UTXO)。
* 每个未花费的UTXO只能被使用一次，即group `i`中的一个UTXO不可能同时出现在 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}"> 和 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 中，这里`k`和`j`为任意的group。

在其他的一些分片方案中通常可以看到将跨分片交易设计为两阶段提交的形式，而alephium以交易分片链的方式在单步内解决了跨分片交易的问题，这样做的优点为:

* 一旦对应的块finalized(PoW通常可以采用类似于polkadot的GRANDPA或者是以太坊casper的FFG)，也代表跨分片的交易已经完成。而两阶段提交的方式可能需要等到相应的两条分片链的块finalized才能保证跨分片交易完成。

当然这样做的代价就是单一的分片链需要保存更多的数据，即对于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 中的交易，所有的交易输入来源于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}">；所有的输出用于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}">，其中`k`为任意的group，因此为了保证交易的输入输出依赖关系和避免双花，<img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}"> 依赖于另外`2G - 1`条分片的数据。当我们将系统从`G`个group扩展为`G+1`个group时，每条分片链只需要额外依赖两个分片链的数据，因此满足可扩展性。

**NOTE**: 理论上对于UTXO模型的交易，交易可以有多个输入输出，而且多个输入和多个输出的group可能也会不同，后面在实现部分会详细讨论alephium对交易的处理。

### BlockFlow

我们上面描述的即为`BlockFlow`，在[alephium论文](https://github.com/alephium/white-paper/blob/master/alephium.pdf)中的定义如下:

A blockflow is G * G forks with one fork for each <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}_{i, j \in G}"> such that these forks have correct input/output dependencies and no double spending.

为了满足正确的输入输出依赖关系及避免双花，<img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}">中的每个block都需要依赖`2G - 1`条链的block，下面给出一些定义:

* <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 为 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的最新的块hash，为了不失一般性，我们这里假设 `i != j`
* <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^{'}"> 为 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的parent hash
* <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">的依赖函数: <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> = <img src="https://render.githubusercontent.com/render/math?math=\{H_{i, j}^{'}\} \cup \{H_{i, r}\}_{r \neq j} \cup \{H_{k, l_k}\}_{k \neq i, l_k \in G}">
* <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \to H_{i, j}^{'}">, <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \to H_{i, r}">, <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \to H_{k, l_k}"> 表示块之前的依赖关系
* <img src="https://render.githubusercontent.com/render/math?math=H1 \leq H2"> 表示H1, H2位于同一条链，并且H1为H2的祖先块

<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})">一共有`2G - 1`个依赖hash，其中<img src="https://render.githubusercontent.com/render/math?math=\{H_{k, l_k}\}_{k \neq i, l_k \in G}"> 一共有`G - 1`个hash，我们将其成为<img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">的Input依赖；<img src="https://render.githubusercontent.com/render/math?math=\{H_{i, j}^{'}\} \cup \{H_{i, r}\}_{r \neq j}"> 一共`G`个hash，我们将其称为<img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">的Output依赖。

对于将每个<img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k}">的Input依赖定义为{<img src="https://render.githubusercontent.com/render/math?math=H_{k, l}">}, 其中<img src="https://render.githubusercontent.com/render/math?math=k \neq i, l \neq l_k, l \in G">，满足<img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k} \to H_{k, l}">，则:

* <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})"> = <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j}) \cup \{H_{k, l}\}_{k \neq i, l \neq l_k}">
* 如果对于所有的<img src="https://render.githubusercontent.com/render/math?math=H_{a, b} \in D(H1), H_{a, b}^* \in D(H2)">，<img src="https://render.githubusercontent.com/render/math?math=H_{a, b} \lt H_{a, b}^*">成立，则 <img src="https://render.githubusercontent.com/render/math?math=D(H1) \lt D(H2)"> 成立

根据定义，当<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})">确定之后，我们也可以得到一个确定性的<img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})">，因此当<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})">满足下面的条件时，我们称<img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})">为一个有效的BlockFlow。

1. 当 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^* \lt H_{i, j}"> 并且 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^* \to H_{i, r}^*"> 成立时，<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 需要保证 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}^* \leq H_{i, r}">
2. 当 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^* \lt H_{i, j}"> 并且 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^* \to H_{k, m_k}^* k \neq i"> 成立时，<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 需要保证对于 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k} \in D(H_{i, j})">，<img src="https://render.githubusercontent.com/render/math?math=H_{k, m_K}^* \leq H_{k, m_k}">
3. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}, r \neq j">，如果 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, r})"> 中存在 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r} \to H_{i, l}^*, l \in G">，则 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 需要保证Output依赖<img src="https://render.githubusercontent.com/render/math?math=H_{i, l}"> 满足 <img src="https://render.githubusercontent.com/render/math?math=H_{i, l} ^* \leq H_{i, l}">
4. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}, r \neq j">，如果 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, r})"> 中存在 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r} \to H_{k, m_k}^*, k \neq i, m_k \in G">，则 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 需要保证对于 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_K} \in D(H_{i, j})">，<img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}^* \leq H_{k, m_k}">
5. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k}, k \neq i, l_k \in G">，如果 <img src="https://render.githubusercontent.com/render/math?math=d(H_{k, l_k})"> 中存在 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k} \to H_{s, m_s}^*, s \neq k, m_s \in G">，则 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 需要保证对于 <img src="https://render.githubusercontent.com/render/math?math=H_{s, m_s} \in D(H_{i, j})">，<img src="https://render.githubusercontent.com/render/math?math=H_{s, m_s}^* \leq H_{s, m_s}">

下面我们来详细的描述每一个条件的具体内容:

1. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 的祖先块 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^*">，<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j}^*)"> 中的Output依赖<img src="https://render.githubusercontent.com/render/math?math=H_{i, r}^*">与对应的 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Output依赖<img src="https://render.githubusercontent.com/render/math?math=H_{i, r}"> 必须满足 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}^* \leq H_{i, r}">
2. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 的祖先块 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^*">, <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j}^*)"> 中的Input依赖<img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}^*">与对应的 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})"> 中的<img src="https://render.githubusercontent.com/render/math?math=H_{k, mk}"> 必须满足 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}^* \leq H_{k, m_k}">
3. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}">，<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, r})"> 中的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, l}^*">，则对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> Output依赖中的 <img src="https://render.githubusercontent.com/render/math?math=H_{i, l}">，必须满足 <img src="https://render.githubusercontent.com/render/math?math=H_{i, l}^* \leq H_{i, l}">
4. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}">, <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, r})"> 中的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}^*"> 与对应的 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})"> 中的 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}"> 必须满足 <img src="https://render.githubusercontent.com/render/math?math=H_{k, m_k}^* \leq H_{k, m_k}">
5. 对于 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 中的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k}">, <img src="https://render.githubusercontent.com/render/math?math=d(H_{k, l_k})"> 中的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{s, m_s}^*">与对应的 <img src="https://render.githubusercontent.com/render/math?math=D(H_{s, m_s})"> 中的 <img src="https://render.githubusercontent.com/render/math?math=H_{s, m_s}"> 必须满足 <img src="https://render.githubusercontent.com/render/math?math=H_{s, m_s}^* \leq H_{s, m_s}">

除此之外，我们还需要满足:

1. <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">的所有交易输入都来自于<img src="https://render.githubusercontent.com/render/math?math=H_{k, i}, k \in G">的交易输出
2. <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">的交易输入与<img src="https://render.githubusercontent.com/render/math?math=H_{i, k}">的交易输入的交集为空，其中<img src="https://render.githubusercontent.com/render/math?math=j \in G, k \in G, j \neq k">

当上述条件满足时，我们可以得到:

1. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 的parent block <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^{'}">，<img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j}^{'}) \lt D(H_{i, j})">成立
2. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 的Output依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{i, r}">，<img src="https://render.githubusercontent.com/render/math?math=D(H_{i, r}) \lt D(H_{i, j})">成立
3. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k}">, <img src="https://render.githubusercontent.com/render/math?math=D(H_{k, l_k}) \lt D(H_{i, j})">成立
4. 对于 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l_k}"> 的Input依赖 <img src="https://render.githubusercontent.com/render/math?math=H_{k, l}">，从2可以得出 <img src="https://render.githubusercontent.com/render/math?math=D(H_{k, l}) \lt D(H_{k, l_k})">，因此 <img src="https://render.githubusercontent.com/render/math?math=D(H_{k, l}) \lt D(H_{i, j})">

证明过程很简单，都可以从上面的5个条件推导出来，具体可以参考论文中的描述。

### security

现在我们来分析分片的安全性，假设恶意节点想要双花 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 上的UTXO，则需要对 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 分叉出一条链，这里记为 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}^f">，因为 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的所有的UTXO都将用于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{j, k}, k \in G">，因此恶意节点需要继续分叉所有的 <img src="https://render.githubusercontent.com/render/math?math=Chain_{j, k}">, 以此类推，如果恶意节点想要双花 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 上的UTXO，则需要对所有的链 <img src="https://render.githubusercontent.com/render/math?math=Chain_{m, n}, m, n \in G"> 进行分叉。因此想要分叉独立的分片仍然需要全网50%以上的算力，并且分片安全性不会随着分片数量的增加而降低。

### mining

对于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}">，为了找出最新的块<img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">，使得<img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})"> 满足上述的条件，论文中提出了一种启发式的mining算法:

* 对于两个BlockFlow, <img src="https://render.githubusercontent.com/render/math?math=BF^1"> 和 <img src="https://render.githubusercontent.com/render/math?math=BF^2">，如果对于所有的 <img src="https://render.githubusercontent.com/render/math?math=i \in G"> 满足 <img src="https://render.githubusercontent.com/render/math?math=\{H_{i, j}^1\}_{j \in G} \lt \{H_{i, j}^2\}_{j \in G}"> 或者 <img src="https://render.githubusercontent.com/render/math?math=\{H_{i, j}^2\}_{j \in G} \lt \{H_{i, j}^1\}_{j \in G}">，其中 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^1 \in BF^1, H_{i, j}^2 \in BF^2">, 则<img src="https://render.githubusercontent.com/render/math?math=BF^1"> 和 <img src="https://render.githubusercontent.com/render/math?math=BF^2"> 是兼容的。

**NOTE**: 注意上述的条件并不等价于 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j}^1) < D(H_{i, j}^2)"> 或者 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j}^2) < D(H_{i, j}^1)">，上述的条件更宽松。

根据上述定义，我们可以得出下面的结论:

* 如果两个有效的BlockFlow <img src="https://render.githubusercontent.com/render/math?math=BF^1"> 和 <img src="https://render.githubusercontent.com/render/math?math=BF^2"> 是兼容的，则相对应的分片链都是互相依赖的，因此 <img src="https://render.githubusercontent.com/render/math?math=BF^1 \cup BF^2"> 也是一个有效的BlockFlow，<img src="https://render.githubusercontent.com/render/math?math=BF^1 \cup BF^2"> 表示将对应的分片链扩展至最长链。

因此为了找到 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})">, 首先获取当前的best BlockFlow(<img src="https://render.githubusercontent.com/render/math?math=D(H_{a, b})">)，然后对于所有的 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, r}_{r \in G}"> 的所有最新块 <img src="https://render.githubusercontent.com/render/math?math=\{H_{i, r}\}">，找出所有与 <img src="https://render.githubusercontent.com/render/math?math=D(H_{a, b})"> 兼容的 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, r})">；同时找出所有与 <img src="https://render.githubusercontent.com/render/math?math=D(H_{a, b})"> 兼容的 <img src="https://render.githubusercontent.com/render/math?math=D(H_{m, n}">), 将所有兼容的BlockFlow执行合并操作，最终得出的BlockFlow能够唯一确定一个有效的 <img src="https://render.githubusercontent.com/render/math?math=d(H_{i, j})">，然后miner打包有效的交易计算出nonce并广播最新的区块。

## implementation

在介绍alephium的实现之前，我们先简单分析一下分片之间的数据依赖，我们先来考虑基于UTXO的交易，先给出一些定义:

* 用户: 用户是指系统的使用者，通常会通过发送交易来改变链上的状态，在这里讨论的分片系统中，每个用户可能会有多个地址，每个地址对应于一条分片，因此这里说的在group `i`上的用户其实是指用户在group `i`上的地址
* 地址: 通过密码学生成的二进制字符串，一个用户在多个分片上的不同地址可以通过加入分片后缀来区分
* 分片 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 负责处理从group `i`中的地址发往group `j`中的地址的所有交易，交易的所有输入必须来源于`i`，所有的输出必须都是`j`

对于分片链 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}">，依赖的数据有:

* <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}_{k \in G}"> 的所有区块，<img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的所有交易输出会直接引用 <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}"> 的交易输出
* <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}_{k \in G}"> 的所有区块，用于验证交易避免双花
* <img src="https://render.githubusercontent.com/render/math?math=Chain_{m, n}"> , 其中<img src="https://render.githubusercontent.com/render/math?math=m, n \in G"> 并且 <img src="https://render.githubusercontent.com/render/math?math=m \neq i, n \neq i"> 的所有区块头，用于计算有效的BlockFlow

现在我们需要考虑一个问题: **数据如何存储才能够保证group `i`上用户的资产不会被在不同的分片链 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}, Chain_{i, k}, j \neq k"> 上双花？**

如果直接在每个分片链上存储一份相当于将用户的资产扩大了G倍，因此解决上述问题最简单的方法就是对于group `i`用户的数据只保存一份，这也是当前alephium的实现方式，下面我们详细描述alephium的存储实现。

### storage

当前alephium的实现将系统从逻辑上划分为`G * G`个分片链 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}_{i \in G, j \in G}">，而物理存储上仍然是划分为`G`个存储单元，即一个节点负责存储一个group中用户及相关的数据。

**NOTE**: 当前alephium的实现还抽象出了broker的概念，即一个broker(可以等同于节点的概念)负责处理多个group，不同的broker可以负责不同的group，为了方便讨论我们在这里忽略broker，相当于每个节点负责一个group。

下面来详细看一下具体的数据存储:

```scala
protected val intraGroupChains: AVector[BlockChainWithState] = {
  AVector.tabulate(brokerConfig.groupNumPerBroker) { groupShift =>
    val group        = brokerConfig.groupFrom + groupShift
    val genesisBlock = genesisBlocks(group)(group)
    blockchainWithStateBuilder(genesisBlock, updateState)
  }
}

private val inBlockChains: AVector[AVector[BlockChain]] =
  AVector.tabulate(brokerConfig.groupNumPerBroker, groups - brokerConfig.groupNumPerBroker) {
    (toShift, k) =>
      val mainGroup    = brokerConfig.groupFrom + toShift
      val fromIndex    = if (k < brokerConfig.groupFrom) k else k + brokerConfig.groupNumPerBroker
      val genesisBlock = genesisBlocks(fromIndex)(mainGroup)
      blockchainBuilder(genesisBlock)
  }

private val outBlockChains: AVector[AVector[BlockChain]] =
  AVector.tabulate(brokerConfig.groupNumPerBroker, groups) { (fromShift, to) =>
    val mainGroup = brokerConfig.groupFrom + fromShift
    if (mainGroup == to) {
      intraGroupChains(fromShift)
    } else {
      val genesisBlock = genesisBlocks(mainGroup)(to)
      blockchainBuilder(genesisBlock)
    }
  }

private val blockHeaderChains: AVector[AVector[BlockHeaderChain]] =
  AVector.tabulate(groups, groups) { case (from, to) =>
    if (brokerConfig.containsRaw(from)) {
      val fromShift = from - brokerConfig.groupFrom
      outBlockChains(fromShift)(to)
    } else if (brokerConfig.containsRaw(to)) {
      val toShift = to - brokerConfig.groupFrom
      val fromIndex =
        if (from < brokerConfig.groupFrom) from else from - brokerConfig.groupNumPerBroker
      inBlockChains(toShift)(fromIndex)
    } else {
      val genesisHeader = genesisBlocks(from)(to).header
      blockheaderChainBuilder(genesisHeader)
    }
  }
```

这里我们假设`groupNumPerBroker`为1，因此对于group `i`:

* intraBlockChain: 数量为1，类型为[BlockChainWithState](https://github.com/alephium/alephium/blob/master/flow/src/main/scala/org/alephium/flow/core/BlockChainWithState.scala)，用于记录 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}"> 的所有交易、区块及group `i`上所有用户的状态即UTXO
* inBlockChains: 数量为`G - 1`，类型为[BlockChain](https://github.com/alephium/alephium/blob/master/flow/src/main/scala/org/alephium/flow/core/BlockChain.scala)，用于记录 <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}_{k \neq i}"> 的所有区块
* outBlockChains: 数量为`G`(包括intraBlockChain)，类型为[BlockChain](https://github.com/alephium/alephium/blob/master/flow/src/main/scala/org/alephium/flow/core/BlockChain.scala)，用于记录 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}"> 的所有区块及交易，并建立交易索引方便查询
* blockHeaderChains: 数量为`G * G`(包括`inBlockChains`和`outBlockChains`)，用于记录 <img src="https://render.githubusercontent.com/render/math?math=Chain_{m, n}_{m, n \in G}"> 的区块头，用于计算BlockFlow

存储状态更新为:

* blockHeaderChains: 当从网络中收到 <img src="https://render.githubusercontent.com/render/math?math=Chain_{m, n}"> 的BlockHeader，验证BlockHeader并保存
* inBlockChains: 当从网络中收到 <img src="https://render.githubusercontent.com/render/math?math=Chain_{k, i}_{k \neq i}"> 的Block，验证Block并保存
* outBlockChains: 当从网络其他节点或miner收到 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, k}_{k \neq i}"> 的block，验证Block，保存并更新交易索引
* intraBlockChain: 当从网络其他节点或miner接收到 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}"> 的Block(hash为 <img src="https://render.githubusercontent.com/render/math?math=H_{i, i}">)，验证有效后，计算:
  * <img src="https://render.githubusercontent.com/render/math?math=H_{i, i}^{'} \in D(H_{i, i})">，并计算<img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j}^{'})">
  * 对于所有的 <img src="https://render.githubusercontent.com/render/math?math=H_{k, i} \in D(H_{i, i}), H_{k, i}^{'} \in D(H_{i, i}^{'}), k \neq i">，计算 <img src="https://render.githubusercontent.com/render/math?math=bs1 = diff(H_{k, i}, H_{k, i}^{'})">，`diff`用于计算同一条链上两个hash之间的区块
  * 对于所有的 <img src="https://render.githubusercontent.com/render/math?math=H_{i, k} \in D(H_{i, i}), H_{i, k}^{'} \in D(H_{i, i}^{'})">，计算 <img src="https://render.githubusercontent.com/render/math?math=bs2 = diff(H_{i, k}, H_{i, k}^{'})">
  * 对于 <img src="https://render.githubusercontent.com/render/math?math=block \in bs1">，将block中的所有交易的输出追加到状态存储中
  * 对于 <img src="https://render.githubusercontent.com/render/math?math=block \in bs2">，从状态存储中删除block中的所有交易的输入引用的交易输出，**如果交易的输出目标group为`i`的话，将输出追加到状态存储中**

上面我们假设 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 中所有的交易输出目标group都为`j`，而当前alephium的实现允许 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的交易输出目标group为`i`或`j`。即便如此，交易的原子性仍然能够得到保证，比如假设 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \in Chain_{i, j}"> 中存在一笔交易`tx`，其输出中同时有目标group为`i`和`j`的输出。

* 如果存在 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \in D(H_{j, k}_{k \in G}), i \neq j">，其`tx`中的目标group为`j`的输出必然会被存储到 <img src="https://render.githubusercontent.com/render/math?math=Chain_{j, j}">。
* 如果 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j} \notin D(H_{j, k})_{k \in G}">，那么必然有一个跟 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}"> 同样高度的 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^{'}"> 使得 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^{'} \in D(H_{j, k})"> 成立，无论 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}^{'}"> 中是否存在`tx`，都能保证交易的原子性。

alephium底层使用的存储结构类似于以太坊的MPT树，因此当出现分叉时，根据fork choice rule，直接将canonical BlockHeader写入`BlockHeaderChain`即可。

另外，尽管当前alephium的实现是通过 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}_{i \in G}"> 的区块来更新状态的，但 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}"> 跟其他分片并没有本质区别，换而言之，完全可以根据 <img src="https://render.githubusercontent.com/render/math?math=Chain{i, j}_{j \in G, i \neq j}"> 来驱动group `i`的状态更新。

### mining

mining的实现跟论文中的描述有一些差异，当前mining计算 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> BlockFlow的步骤如下所示:

1. 获取当前的best BlockFlow，记为`BF`
2. 对于所有的 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, r}_{r \in G}"> 的最新块 <img src="https://render.githubusercontent.com/render/math?math=\{H_{i, r}\}">, 计算出 <img src="https://render.githubusercontent.com/render/math?math=\{D(H_{i, r})\}">，将所有与BF兼容的BlockFlow合并
3. 对于所有的 <img src="https://render.githubusercontent.com/render/math?math=Chain_{j, j}_{j \in G, j \neq i}"> 的最新块 <img src="https://render.githubusercontent.com/render/math?math=\{H_{j, j}\}">，计算出 <img src="https://render.githubusercontent.com/render/math?math=\{D(H_{j, j})\}">，将所有与BF兼容的BlockFlow合并

即当前的实现并没有对所有的 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的最新块计算BlockFlow，但这并不影响mining的正确性。

### fork choice rule

alephium当前以heavist weight为标准来决定canonical chain，对于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, j}"> 的最新块 <img src="https://render.githubusercontent.com/render/math?math=H_{i, j}">，weight计算如下:

1. <img src="https://render.githubusercontent.com/render/math?math=newBf = D(H_{i, j})">, 选取 <img src="https://render.githubusercontent.com/render/math?math=D(H_{i, j})"> 中的属于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}"> 的hash，记为 <img src="https://render.githubusercontent.com/render/math?math=H_{i, i}^{'}">
2. 计算 <img src="https://render.githubusercontent.com/render/math?math=oldBf = D(H_{i, i}^{'})">, 并计算 <img src="https://render.githubusercontent.com/render/math?math=diffs = bfDiff(newBf, oldBf)">, 其中`bfDiff`计算两个BlockFlow之间的所有区块
3. <img src="https://render.githubusercontent.com/render/math?math=base = accWeight(H_{i, i}^{'})">, `weight = base + diffs.map(_.weight).sum`, `accWeight`计算当前块的累计`weight`

这里`weight`的计算依赖于 <img src="https://render.githubusercontent.com/render/math?math=Chain_{i, i}">，但同样的我们也可以选择其他的链为依据来计算weight，对于正确性没有影响。

### verify

对于Block、BlockHeader及Transaction的验证除了跟BlockFlow相关的验证外基本和其他区块链系统的验证规则相同。

### VM

TODO

## resharding

随着系统的运行可能后面会需要resharding，我们这里只考虑增加分组的情况，一种最简单的做法就是，保存当前所有用户所在的分组信息不变，对于新的用户将其分配到新的group，当然无论如何，resharding都避免不了hard fork。

另外对于分片交易不均衡的问题，我认为并不一定要在底层去解决，完全可以尝试从上层入手，比如在将不同的用户/合约分布到多个分片上来减少单个分片的压力。

## conclude

对于PoW的分片链，为了保证交易的原子性及分片的可扩展性必须要仔细考虑分片算法的设计，alephium的分片算法原理较为复杂，也比较巧妙，BlockFlow可以看作是当前块对以往块的确认，而如果想要恶意分叉一条分片链则需要对所有后续的依赖链进行分叉，因此在安全性的前提下保证了可扩展性。

后面有时间我会再介绍一些其他的PoW的分片链，并跟alephium做个比较。

## reference

1. https://github.com/alephium/alephium
2. https://github.com/alephium/white-paper/blob/master/alephium.pdf

