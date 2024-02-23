---
layout:     post
title:      transformerXL笔记
subtitle:   https://arxiv.org/abs/1901.02860
date:       2024-01-20
author:     Placeholder
# header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - transformer
---

<!-- ## 前言
最初弄这个原因是大三上的叶南阳老师在听我的大作业汇报的时候点评到可以看一下DDPM等扩散模型，因为我选的那个论文(BIRM)涉及到了一些如变分推理的东西。然后某时刻一打开labml-nn的扩散模型，发现诶这不和那个代码里面某一段非常相似吗？然后就起了这个念头，希望按顺序弄掉labml的几个东西。 -->

## 传统transformer的问题
主要有两个问题。首先是在划分segments的时候，这些segments之间是没有联系的。也就是说一组[seqlen, bs, 1]的语料,transformer只会对seqlen那一维度做相关，而不是对bs做。另一个问题就是，划分的segments没有考虑到句子的自然边界，他们在语义上不完整。

## transformerXL的改进
改进也主要包括两部分
### Segment-level Recurrence
这一块的内容概括起来就是，在处理当前segment的时候，缓存上一个segment中**所有**layers的隐向量。这些向量**仅仅参与前向计算**。如此递归下去，使得bs维度也能进行类rnn的操作。

数学形式如下:

令$s_\tau$为第$\tau$个形状为$R^{L\times d}$的segment，$s_{\tau +1}$同理。


令$h_\tau^n$为第$n$层第$\tau$个segment的隐藏层序列，那么$h_{\tau+1}^n$如下:

$$
\begin{aligned}
&\widetilde{\mathbf{h}}_{\tau+1}^{n-1}=\bigl[\mathrm{SG}(\mathbf{h}_{\tau}^{n-1})\circ\mathbf{h}_{\tau+1}^{n-1}\bigr], \\
&\mathbf{q}_{\tau+1}^{n},\mathbf{k}_{\tau+1}^{n},\mathbf{v}_{\tau+1}^{n}=\mathbf{h}_{\tau+1}^{n-1}\mathbf{W}_{q}^{\top},\widetilde{\mathbf{h}}_{\tau+1}^{n-1}\mathbf{W}_{k}^{\top},\widetilde{\mathbf{h}}_{\tau+1}^{n-1}\mathbf{W}_{v}^{\top}, \\
&\mathbf{h} _{\tau+1}^{n}=\mathrm{TransformerLayer}\left(\mathbf{q}_{\tau+1}^{n},\mathbf{k}_{\tau+1}^{n},\mathbf{v}_{\tau+1}^{n}\right). 
\end{aligned}
$$
回忆attention层输出的形状仅仅取决于query的形状，这里仅仅修改了输入transformer的k和v。同时SG意思是阻断梯度传播，$\circ$符号意思是(在$L$维度)拼接。

但你如果仅仅看了这个公式，你可能看不懂论文下面的训练和测试的图。根据
[Transformer-XL介绍](https://zhuanlan.zhihu.com/p/84159401)的解释，图上有一个点需要注意，在当前segment中，第n层的每个隐向量的计算，都是利用下一层中包括当前位置在内的，连续前L个长度的隐向量，这是在上面的公式组中没有体现出来的，也是文中没有明说的。每一个位置的隐向量，除了自己的位置，都跟下一层中前(L-1)个位置的token存在依赖关系，而且每往下走一层，依赖关系长度会增加(L-1)。这使得在对长文本进行计算的时候，可以缓存上一个segment的隐向量的结果，不必重复计算，大幅提高计算效率。

稍后让我去labml-nn的代码那块仔细想想，代码中是否体现了这一点。

(labml-nn的实现里，RPE和论文中公式不同，属于是一种简化但是能说服自己的实现。同时这里的实现中，transformerXL是一种不同于transformer的Encoder-Decoder结构，他是直接input-output。但是由于它mem的使用是写在forward的参数里的，看不出来依赖关系的递进)

### Relative Position Encodings
transformerXL放弃绝对位置编码的主要原因在于他会对多个segment做类似rnn的操作，此时你这个多个segment的绝对位置编码其实是一样的。多个segments之间无法区分位置关系，就像论文中写的
$$\begin{aligned}
\mathbf{h}_{\tau+1}& =f(\mathbf{h}_{\tau},\mathbf{E_{s}}_{\tau+1}+\mathbf{U}_{1:L})  \\
\mathbf{h}_{\tau}& =f(\mathbf{h}_{\tau-1},\mathbf{E}_{\mathbf{s}_{\tau}}+\mathbf{U}_{1:L}), 
\end{aligned}$$
这样完全没必要输入一个$U_{1:L}$进去。

作者随后说我们没必要只在输入的时候加上一个embedding，我们完全可以在做attention score的时候加入一个相对位置embedding。这启示我们去看一下做softmax之前的结果:

$$\begin{aligned}
\mathbf{A}_{i,j}^{\mathbf{abs}}& =\underbrace{\mathbf{E}_{x_{i}}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k}\mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k}\mathbf{U}_{j}}_{(b)}  \\
&+\underbrace{\mathbf{U}_{i}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k}\mathbf{E}_{x_{j}}}_{(c)}+\underbrace{\mathbf{U}_{i}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k}\mathbf{U}_{j}}_{(d)}.
\end{aligned}$$

其中$\mathbf{E}$是embedding，$\mathbf{U}$是正弦位置编码。然后作者直接提出了他的想法:
$$\begin{aligned}
\mathbf{A}_{i,j}^{\mathrm{rel}}& =\underbrace{\mathbf{E}_{x_{i}}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k,E}\mathbf{E}_{x_{j}}}_{(a)}+\underbrace{\mathbf{E}_{x_{i}}^{\top}\mathbf{W}_{q}^{\top}\mathbf{W}_{k,R}\mathbf{R}_{i-j}}_{(b)}  \\
&+\underbrace{u^\top \mathbf{W}_{k,E}\mathbf{E}_{x_j}}_{(c)}+\underbrace{v^\top\mathbf{W}_{k,R}\mathbf{R}_{i-j}}_{(d)}.
\end{aligned}$$

这里相当于我们完全抛弃了绝对位置编码，改为相对位置编码。

1. $R_{i-j}$比较好理解，由于只用了$i$前面的段，所以$i-j\geqslant 0$, 且不用学习，理解成$i-j$都没有问题。
2. 既然是相对位置，参照那个展开式，中间就不应该保有绝对位置。所以用俩可训练参数$u,v$来代替。

3. 接着引入了两个矩阵，目的分别是输出content-based key vectors和location-based key vectors。如果你不引入的话，你最后两项相当于共享$W_k$, 这在式子上并不能看出共享有什么意义。分开反而更加突出各自的作用。(c) 确保了整体的content-bias，(d) 则确保了整体的position bias。

完整的流程公式可以直接看论文。