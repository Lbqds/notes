# BFT tolerance consensus

由于工作需要，经常会看一些BFT容错的共识算法，但很少会去自己实现，所以有时看了之后不久就忘了，所以在这里记录一下，避免以后忘了之后再去翻论文。

## PBFT

PBFT对网络的假设是半同步网络，即在未知的GST之后网络中non-faulty节点之间的消息传递有一个上限 ∆。同时对于网络中，假设最多可以容忍`f`个恶意节点，则至少要求节点数为`3f+1`个节点。因为由于最多有`f`个恶意节点，恶意节点可以无限地增加消息延时，所以算法必须保证当non-faulty节点收到n-f个响应时必须能够继续进行下去；但也可能是`f`个non-faulty节点没有响应，所以收到的`n-f`个响应中可能包括了`f`个恶意节点的响应，因此为了保证算法的正确性，必须保证`n-f-f>f`，即`n>=3f+1`。

为了便于描述，假设网络中存在R个节点，每个节点的id为`0`到`R-1`。整个时间线由连续递增的view组成，每个view中id为`view mod |R|`的节点为primary，其他节点为backup。当出现异常时，会通过view-change协议变更primary。正常情况下，对于某个proposal的共识包括两轮投票，`prepare`和`commit`。首先，client将签名的请求sign<sub>c</sub>{Request, o, t, c}发送给primary，其中o为对状态机的操作，t为客户端的时间戳，用于确保exactly-once语义。primary收到客户端的请求后构造出一个`pre-prepare`消息{sign<sub>p</sub>{Pre-prepare, v, n, d}, m}，其中v为当前的view，n为当前请求的序列号，d为请求消息的摘要，m为原始的请求消息。primary将`pre-prepare`消息发送给所有的backups。当backup接收到`pre-prepare`消息并验证通过后，进入prepare阶段。

### prepare

节点`i`进入prepare阶段后，会构造一个`prepare`消息sign<sub>i</sub>{Prepare, v, n, d, i}，节点会将收到的`pre-prepare`消息和`prepare`消息保存到message log中，并将`prepare`消息广播给其他节点。当且仅当i收到`2f+1`个view和n都相同的对消息m的`prepare`投票时(包括自己的投票)，我们称`prepared(m, v, n, i)`为真，然后进入commit阶段。`prepared(m, v, n, i)`为真则能够保证在没有发生view-change时，所有non-faulty节点都会对sequence number为n的消息m投prepare，即能够保证在同一个view中，所有non-faulty节点对请求顺序达成共识。

### commit

节点`i`进入commit阶段后，会构造一个`commit`消息sign<sub>i</sub>{Commit, v, n, d, i}，并将`commit`消息发送给其他节点。同时节点i从其他节点接收`commit`消息，并将验证有效的`commit`消息保存到message log中，当且仅当i收到`2f+1`个view和n都相同的对消息m的`commit`投票时(包括自己的投票)，我们称`commit-local(m, v, n, i)`为真，此时节点i可以执行请求并更新状态。即便由于网络原因导致只有部分节点commit，view-change也会保证所有non-faulty节点在后续的view中仍会以sequence number为n提交此请求(后面safety部分会详细描述)。

### checkpoint

checkpoint在PBFT主要有两个作用，一是减少存储压力，二是用于view-change(后面会详细描述)。每一次共识的每一轮投票都会将收到的有效消息保存下来，为了节省存储空间，节点可以定时去做checkpoint(比如当`n mod 100 = 0`时)。每个节点将sign<sub>i</sub>{Checkpoint, n, d, i}发送到其他节点，当接收到`2f+1`个`Checkpoint`消息之后，可以将此checkpoint落盘并删除n之前存储的共识信息，接收到的`2f+1`个消息就是这个checkpoint的有效证明。

### view-change

在共识过程中，节点会通过设置定时器来触发view-change，比如当在一定的时间内无法收到消息或没有将请求commit，节点i就会通过向其他节点发送`view-change`消息sign<sub>i</sub>{ViewChange, v+1, n, C, P, i}发起view-change流程。其中`n`是节点i**最新的**checkpoint `s`对应的sequence number，`C`是`s`的有效证明，包含了`2f+1`个签名的`checkpoint`消息；对于每个sequence number大于`n`的请求，将其对应的`2f+1`个prepare投票(如果存在的话)记为P<sub>m</sub>，P由所有的P<sub>m</sub>组成。当`v+1`的primary `pa`，接收到`2f+1`个有效的`view-change`消息之后(包括自己的)，会向其他节点发送`new-view`消息sign<sub>pa</sub>{NewView, v+1, VS, O}并进入`v+1`，`VS`是所有从其他节点接收到的有效的`view-change`消息，`O`是`pre-prepare`消息集合，计算规则如下:

1. 将`VS`中**最新的**checkpoint的sequence number记为`min-s`，将`VS`中最大的prepare投票的sequence number记为`max-s`
2. 对于每个在(min-s, max-s]的sequence number `n`，有两种情况：
   1. 在所有收到的`view-change`消息中，至少有一个与n对应的P<sub>m</sub>，则生成一个新的`pre-prepare`消息sign<sub>pa</sub>{Pre-prepare, v+1, n, d}
   2. 与n对应的P<sub>m</sub>不存在，则生成一个no-op `pre-prepare`消息sign<sub>pa</sub>{Pre-prepare, v+1, n, d<sup>null</sup>}

当backup收到来自`pa`的`new-view`消息之后，按照同样的规则验证VS和O，然后进入`v+1`，然后将所有的`pre-preprare`消息放入到message log中，然后按照正常的流程去执行每一个`pre-prepare`消息。

### safety

PBFT保证所有的non-faulty节点能够对请求的顺序达成一致，即对于一个请求`req`，如果某一个non-faulty节点在sequence number为n时提交了这个请求，那么其他的non-faulty也一定是在sequence number为n时提交了这个请求。safety的证明可以分成两个部分，一是当没有发生view-change时，在prepare阶段已经描述过了，下面主要讨论当发生view-change时，PBFT如何保证safety。

对于一个sequence number为n的请求，会出现两种可能：

1. 有non-faulty节点提交了这个请求，那么说明至少有`f+1`个non-faulty节点已经`prepared`，因此在view-change时，下一个primary至少会收到一个`view-change`消息，其中包括了对这个请求的P<sub>m</sub>，所以即便到了下一个view，这个请求提交时的sequence number仍然为n

2. 所有的non-faulty节点都没有提交这个请求，这种情况又可以分为两种：

   1. 这个请求在某些节点已经`prepared`，比如只有部分节点收到了`2f+1`的prepare投票，但下一个view的primary没有收到这些prepared节点的view-change消息，那么这个请求就不会在n被提交
   2. 这个请求在`2f+1`个节点已经`prepared`，比如有部分non-faulty节点已经到了commit阶段，但由于网络原因没有收到`2f+1`的commit投票，那么这个请求在下一个view仍然会在n被提交

   这两种情况无论出现哪一种情况都不会违反safety

### liveness

PBFT对于网络的假设是半同步网络，依赖于超时来执行view-change。为了尽量保证liveness，PBFT采取了下面的策略：

1. 在等待new-view消息时启动定时器，如果在T时间内没有收到new-view消息，则下一轮的等待时间加倍
2. 当收到`f+1`个有效的view-change消息时，为了避免当前节点太晚进入到下一个view，即便当前view-change的定时器还没有超时，也会选择其中最小的view广播view-change消息

PBFT还会保证`responsiveness`，这一点到后面讨论Tendermint和HotStuff时再具体描述。

### complexity

假设网络中有n个节点，其中最多容忍f个恶意节点。

以message来衡量，正常情况下及view-change的复杂度均为O(n<sup>2</sup>)；由于有可能遇到连续的恶意primary，恢复到正常流程最坏情况为O(n<sup>3</sup>)，有些文献也会记为O(fn<sup>2</sup>)。

以authenticator(比如签名)来衡量，正常情况下复杂度为O(n<sup>2</sup>)；由于view-change消息中对于每个请求都携带着P<sub>m</sub>，则view-change的复杂度为O(n<sup>3</sup>)，当遇到连续的恶意primary，恢复到正常流程复杂度为O(n<sup>4</sup>)，有些文献也会记为O(fn<sup>3</sup>)。

### 其他

### catch up

如果节点掉线一段时间后重连，如果想要尽快地同步到最新状态，那么checkpoint可能还不够，还需要额外的一些机制，比如当节点收到超过`f+1`个view大于当前view的任意类型的有效消息，则尝试去从其他节点的message log去同步请求以及每个请求在每个阶段的QC(quorum certificate，由`2f+1`个投票组成)。

在区块链中，请求就是block，sequence number为高度，即连续递增，同时区块链中会对每个block做checkpoint。以[sawtooth-pbft](https://github.com/hyperledger/sawtooth-pbft)的实现为例，每个高度为h的block中记录着对高度为h-1的block的commit证明，当节点收到比大于当前高度的block时[会去检查并catch up](https://github.com/hyperledger/sawtooth-pbft/blob/master/src/node.rs#L720)，文档可以参考[这里](https://github.com/hyperledger/sawtooth-rfcs/blob/master/text/0031-pbft-node-catchup.md)。

## Tendermint

TODO

## HotStuff

TODO

## Compare

TODO