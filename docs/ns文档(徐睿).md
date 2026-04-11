1.1
[图片]
[图片]
[图片]
测试问题：1.原目的节点流生成时间和大小
网络平均负载，上下行链路，超额订阅定义
50%网络平均负载（对流大小cdf，泊松到达模型，服务器的匹配规则建模，贴近50%）
50%网络平均负载，计算理论每台服务器吞吐。流大小cdf计算流大小的期望，泊松分布得到服务器的发送速率，
两者相乘近似服务器的理论吞吐

先得到每台服务器的50%负载的发送速率，在根据泊松分布得到时间间隔
2.各拥塞控制算法参数设置（增速速率、减速速率、增减速间隔、有无窗口控制、ecn上下阈值）


Hpcc: 精确评估每一条链路的队列瓶颈，在终端设置精确窗口值（保留5%链路带宽，预防incast短流多打一。牺牲少量长流吞吐，保证短流性能）
dctcp:使用ecn，但计算ce包的比例α，通过ce包比例调整cwnd（窗口控制没有hpcc精确）
dcqcn:使用ecn二极信号，终端通过aimd乘性调整发送速率。（二极信号不能准确判断拥塞程度,发送速率也不能精确控制in-flight bytes）
timely：端到端的部署，不依赖交换机。通过rtt变换，控制发送速率（发送速率不能精确控制in-flight bytes）

2021 on-ramp:在终端检测和暂存突发流量，可附加于上面各种算法，对hpcc没起到什么作用




rdma网络存在的问题
1.incast拥塞导致 pfc暂停
2.pfc暂停导致拥塞扩展，pfc死锁和pfc风暴
3.pfc暂停导致带宽容量损失


流量模型
Hadoop 70%0-100KB 25%100KB-1MB 5%大于1MB 平均1KB
75%流量在机架内部
37%50B 2%50B-1KB 31% 1KB-100KB 25%100KB-1MB 5%大于1MB
[图片]
参数
Fct slowdown 实际fct被理想fct标准化后的结果，越大拥塞越严重

[图片]

2020 swift：提到hpcc在大规模incast和iops密集型负载不能发挥作用




1.5

各算法优势场景
timely  对延迟敏感应用 机器学习训练，分布式缓存
Dctcp  突发性短流与长流共存场景
过订比例
double oversubRatio = static_cast<double>(SERVER_COUNT * LEAF_SERVER_CAPACITY) / (SPINE_LEAF_CAPACITY * SPINE_COUNT * LINK_COUNT);
（服务器数量×链路容量）/（链路容量×spine数量×链路数量）
举例：8spine 12leaf 32 host leaf_spine 100G  leaf_server 25G
Oversubratio 1:1

负载计算
 double requestRate = load * LEAF_SERVER_CAPACITY * SERVER_COUNT / oversubRatio / (8 * avg_cdf (cdfTable)) / SERVER_COUNT;
 

网络负载调节的核心是控制单位时间注入网络的比特总量。
首先根据服务器网卡带宽和负载比例计算理论吞吐量（比特/秒）。
接着用理论吞吐量除以平均流大小得到流发送速率（流/秒）。
然后基于泊松过程生成流到达时间间隔，最后从CDF分布中采样生成每个流的具体大小。
整个过程将比特级负载转化为具有时间特性和大小特性的具体流序列。

总体情况

[图片]
[图片]
leaf_spine拓扑：host-leaf 25G 382条链路； leaf-spine 100G 96条链路；382台主机随机匹配
流量模型：1.server(64KB) cache(601KB) search(1.6MB)  mine(7.41MB)；负载0.3-0.9；泊松分布
2.server cache search mine（+多incast）；负载0.3-0.9；泊松分布
流表计算: 1.网卡带宽和负载度得到每台服务器发送比特速率
2.服务器发送比特速率和流平均大小得到流生成速率
3.流生成速率带入泊松分布得到随机事件时刻
4.每台服务器按照流模型生成流大小，泊松分布得到流发生时刻
5.hpcc论文中hadoop场景;四种流量模型，0.3、0.5、0.7、0.9四个负载度（16个实验）；加入多incast（16个实验）

1MB分析阈值
Server（0.3-0.9负载） <1MB dctcp
cache（0.3-0.9负载）<1MB  hpccpint ;>1MB dctcp
search（0.3-0.9负载）<1MB hpccpint ;>1MB dctcp
mine（0.3-0.9负载）<1MB timely （0.3-0.7负载） hpccpint（0.9负载） ;>1MB dctcp

Server+多incast（0.3-0.9负载） <1MB dctcp
cache+多incast（0.3-0.9负载）<1MB dctcp ;>1MB dctcp
search+多incast（0.3-0.9负载）<1MB hpccpint ;>1MB dctcp
mine+多incast（0.3-0.9负载）<1MB timely ;>1MB dctcp
100KB和1MB分析阈值
Hadoop 
Server（0.3-0.9负载）
cache（0.3-0.9负载）
search（0.3-0.9负载）
mine（0.3-0.9负载）

Server+多incast（0.3-0.9负载）
cache+多incast（0.3-0.9负载）
search+多incast（0.3-0.9负载）
mine+多incast（0.3-0.9负载）
<100KB    hpcc hpccpint
100KB-1MB  不确定
>1MB       dctcp
server（0.3-0.9负载）
[图片]
[图片]
<1MB  dctcp 
[图片]
[图片]
<1MB  dctcp 
[图片]
[图片]
<1MB  dctcp 
[图片]
[图片]
<1MB  dctcp  
cache（0.3-0.9负载）
[图片]
[图片]
<1MB  hpcc dctcp hpccpint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB hpccpint  ；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpccpint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpccpint；  >1MB dctcp    总体 dctcp
search（0.3-0.9负载）
[图片]
[图片]
<1MB  hpcc hpccpint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB   hpccpint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpcc  hpccpint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpcc  hpccpint；  >1MB dctcp    总体 dctcp
mine（0.3-0.9负载）
Mine 长流占据的流量远比其他几种模型大，长时间占用某些链路。ecmp负载均衡导致短流和大流的碰撞，可能高负载场景，短流运气好没走拥塞路径；低负载场景短流都走了拥塞路径。
高负载反而比低负载fct完成时间更短
[图片]
[图片]
<1MB  timely ；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  timely；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  timely；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpcc hpccpint；  >1MB dctcp    总体 dctcp

server+多incast（0.3-0.9负载）
[图片]
[图片]
<1MB  dctcp hpccpint；    
[图片]
[图片]
<1MB  dctcp ；  
[图片]
[图片]
<1MB  dctcp ；  
[图片]
[图片]
<1MB  dctcp ；  
cache+多incast（0.3-0.9负载）
[图片]
[图片]
<1MB  dctcp；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  dctcp hpcc pint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  dctcp hpcc pint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpcc dctcp hpcc pint；  >1MB dctcp    总体 dctcp
search+多incast（0.3-0.9负载）
[图片]
[图片]
<1MB   hpcc pint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB   hpcc pint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  dctcp hpcc pint；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB  hpcc pint；  >1MB dctcp    总体 dctcp
mine+多incast（0.3-0.9负载）
[图片]
[图片]
<1MB  timely；  >1MB dctcp    总体 timely
[图片]
[图片]
<1MB timely ；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB timely ；  >1MB dctcp    总体 dctcp
[图片]
[图片]
<1MB timely ；  >1MB dctcp    总体 dctcp
1.12
PFC（Priority-based Flow Control）的基本概念 
简单说，为了保证无损网络，交换机检测到某端口某优先级缓冲区快溢出（全局共享缓冲池），通知直连端口上游设备暂停发送。
pfc拥塞扩散原因：1.p2输入端口流量霸占输出端口缓冲区，
2.其他输入端口p3、p4的流量也会逐渐积压，向上游设备发送暂停帧，
3.上游设备不发送数据，入口缓冲区流量堆积，又将pause扩散到上游
[图片]
dcqcn通过ecn+cnp，逐流调节发送速率。


PFC（基于优先级的流量控制）是一种以太网流量控制机制，旨在实现无损网络。它的核心功能是防止交换机或网卡因缓冲区溢出导致的数据包丢失。具体来说：
- 当交换机端口的接收缓冲区使用量超过预设阈值时，会向数据发送方（上游设备）发送PAUSE帧，要求其暂停发送特定优先级的数据流。
- 发送方在收到PAUSE帧后会暂停传输，直到收到RESUME帧（缓冲区降至安全阈值以下）后才恢复发送。
- PFC支持8个优先级队列，允许对不同类型的流量（如RDMA、存储、普通TCP）进行差异化控制。
头阻塞是PFC的主要缺陷之一，源于其粗粒度的流量控制机制。PFC的操作单位是端口或端口+优先级，而非单个数据流。当多个流共享同一端口或优先级时，PAUSE帧会暂停该优先级下的所有流量，即使其中某些流并未直接导致拥塞。


1.16

总体情况

[图片]
[图片]
leaf_spine拓扑：host-leaf 25G 382条链路； leaf-spine 100G 96条链路；382台主机随机匹配
流量模型：1.server(0.64KB) cache(601KB) search(1.6MB)  mine(7.41MB)；负载0.3-0.9；泊松分布
2.server cache search mine（+多incast）；负载0.3-0.9；泊松分布
流表计算: 1.网卡带宽和负载度得到每台服务器发送比特速率
2.服务器发送比特速率和流平均大小得到流生成速率
3.流生成速率带入泊松分布得到随机事件时刻
4.每台服务器按照流模型生成流大小，泊松分布得到流发生时刻
5.hpcc论文中hadoop场景;四种流量模型，0.3、0.5、0.7、0.9四个负载度（16个实验）；加入多incast（16个实验）


问题：1. dcqcn的主要解决pfc扩散死锁问题（但是没有hpcc好，hpcc可以降低到0，加一个发送窗口就可将pause降低到0）
[图片]
2.timely的主要面向高度可预测的微秒级延迟应用（大规模机器学习，分布式内存缓存）
2.现有的场景下还需要补充什么场景（0.1负载，1.0负载，加大incast规模，周期性尺寸大小一致突发流）
3.最终实验场景，混合流量场景比较融合算法和各独立算法性能？
Server（0.3-0.9负载）


hadoop0.3负载+60打1 incast
[图片]
0-100KB  hpcc  hpccpint
100KB-1MB timely
>1MB dctcp

hadoop0.5负载
[图片]
 0-100KB  hpcc 
100KB-1MB  dctcp
>1MB dctcp
server（0.3-0.9）
[图片]
 0-100KB hpcc
100KB-1MB dctcp
>1MB 
[图片]
 0-100KB hpcc pint
100KB-1MB dcqcn
>1MB 
[图片]
 0-100KB hpcc pint
100KB-1MB dcqcn
>1MB 
[图片]
 0-100KB  hpcc pint
100KB-1MB dcqcn
>1MB 
cache（0.3-0.9）
[图片]
 0-100KB hpcc hpccpint
100KB-1MB dctcp
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dctcp
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dctcp
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dctcp
>1MB dctcp

search（0.3-0.9）
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB hpcc 
>1MB dctcp
[图片]
 0-100KB  hpcc pint
100KB-1MB hpcc pint
>1MB  dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB hpcc
>1MB  dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB hpcc
>1MB dctcp
mine（0.3-0.9）
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB hpcc
>1MB dctcp
[图片]
 0-100KB hpcc
100KB-1MB hpcc
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB timely
>1MB dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB hpcc 
>1MB dctcp

server+多incast（0.3-0.9）
[图片]
 0-100KB hpcc
100KB-1MB dcqcn
>1MB 
[图片]
 0-100KB hpcc pint
100KB-1MB dcqcn
>1MB 
[图片]
 0-100KB hpcc pint 
100KB-1MB dcqcn
>1MB 
[图片]
 0-100KB hpcc pint
100KB-1MB dcqcn
>1MB 
cache+多incast（0.3-0.9）
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB timely
>1MB  dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB dcqcn
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dctcp
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dctcp
>1MB  dctcp

search+多incast（0.3-0.9）
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB  timely
>1MB dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB timely
>1MB  dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB dcqcn
>1MB dctcp
[图片]
 0-100KB hpcc pint
100KB-1MB hpcc pint
>1MB dctcp
mine+多incast（0.3-0.9）
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB timely
>1MB dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB timely
>1MB dctcp
[图片]
 0-100KB hpcc hpcc pint
100KB-1MB timely
>1MB  dctcp
[图片]
 0-100KB hpcc 
100KB-1MB timely
>1MB  dctcp
1.19



最优算法统计
1.bandit多臂老虎机+mlp 
流场景 流负载 算法 时刻 流大小 -> FCT
训练集 （1）根据平均FCT算每种算法每种场景中的理论cost
（2）不同场景调节选择算法的概率，得到实际cost
（3）比较理论和实际cost，调节算法概率让实际cost更逼近理论cost，得到最终模型
测试集 输入模型不同的场景得到最优算法，与预测集理论最优算法对比
问题: 方向错了，做成预测推理了，实际要做的是分类决策
2. 分类决策
最优算法<->场景平均FCT最优
数据集是32个场景，大量平均FCT的样本
方案（1）32个场景全部跑几十次，得到几十组32场景的平均FCT，神经网络学习不同场景和最优算法的规律
方案（2）32个场景跑一次，一次性输入大量数据流，分割样本计算平均FCT
问题:规模大，流量大跑起来费时
mine模型4个负载度，100KB-1MB区间达到4万条流的样本，总体流数70w，总流量达4900G
3.统计学

[图片]
4模型×有无incast×4负载度=32场景
[图片]

最优算法和负载度关系不大
与流大小和流模型关系大，规律性更强
暂时无法在飞书文档外展示此内容


Dcqcn dctcp hpccpint hpcc timely
五种算法去掉hpcc 和timely
hpcc是满血版int网络内遥测，给所有数据包头部附加拥塞控制信息，额外开销大
timely的报头格式和int报头格式不兼容，需重新修改报头协议；timely不稳定，在server场景不如dcqcn

Dcqcn对  incast流、100KB-500KB server serveri效果好
dctcp对>1MB流、500KB-1MB cache cachei效果好
hpccpint对<100KB流、100KB-500KB cache mine search cachei minei searchi、500KB-1M mine minei search searchi效果好


multi-cc 在不同流模型中基于流特征切换最优算法
 multi-cc验证
4模型×有无incast×4负载度=32场景
[图片]
总体上，multi-cc 对 0-100KB,100KB-500KB, >1MB区间以及incast流，没有明显短板，但是各算法长板不能发挥到极致，达到次最优算法的水平
500KB-1MB之间为最优算法

对所有流而言，在mine minei场景中可以达到平均fct最短，其他场景平均fct接近dctcp达到次优水平

对cache search mine cachei searchi minei0.9极高负载度一定程度降低尾部时延


问题1.混合业务场景怎么理解，数据集把web server mine search混合到一起，还是web+incast这样就算混合业务了
incast流特征
1.突发频率：一秒钟十次至200次，大概10ms一次
2.持续时间2ms
3.流数量100-400间
4.25Gbps链路，是6.25MB，200个流为例，一条流32KB

1.根据原目的地址确定incast分算法
2.重新调整server mine cache search 流大小分布（对于incast流不是dcqcn和timely最优，dcqcn，dctcp，hpccpint中dctcp综合性能更好一些）
3. incast还有必要去测吗？还是基于流大小？如果确实和incast的没有关系都不用做了
有incast的数据集，无incast的数据集，是否会影响指标最好的算法
incast大小不同，incast平均大小32KB，cache和server 0.9负载 dctcp效果好，其他差不多，incast500KB，timely对incast效果最好
incast的数量不同，是否影响指标最好的算法
4. 0-100KB劣化的原因是什么,有办法调节吗
[图片]
200 平均32KB incast流（incast<100KB时dcqcn和timely不显著）
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
200 平均500KB incast流（timely和dcqcn对incast的效果比较好）
100KB-1MB
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
incast流
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]
[图片]


3.7


Cccr 融合ecn和rtt信号
Ictcp 不显示识别，接收端驱动的拥塞控制，可缓解incast
  
显式识别 incast

隐式识别 incast  cccr融合ecn和rtt信号，更全面评估网络状态
                       ictcp接收端驱动拥塞控制

暂时无法在飞书文档外展示此内容
incast流大小在0-100KB，各算法影响不大
 incast流大小在100KB-1MB区间的时候，统一用cccr或pluser算法（incast复杂场景）
1.cccr算法，单个信号无法准确评估网络状态，基于经验规则的算法依赖于参数调优，融合ecn和rtt信号，已经发生拥塞根据ecn减速，未发生拥塞根据rtt提速，引入最大的发送窗口限制，通过 BDP（带宽时延乘积）来限制在途字节数（In-flight Bytes）
2.pluser算法（专门区分incast和普通流的拥塞控制算法），放弃“基于队列长度阈值”的判断方式，转而采用队列长度的梯度（变化率）来显式识别Incast，终端采用急刹车和快恢复的方式调节窗口
3.在pluser基础上修改，使用队列长度梯度识别incast，再切换dcqcn或timely拥塞控制算法



待解决的问题，一些实验现象如何发掘其原因 1.不同大小流使用最佳拥塞控制算法，在0-100KB,>1MB景效果不是最优而是次优,在100KB-1MB是最优，如何解释
2.hpcc和dctcp（基于窗口拥塞控制队列长度保持在0）对incast的效果不如dcqcn和timely，如何解释
3.多cc算法融合 timely 包头和hpccpint包头的问题
[图片]
dctcp根据包标记的比例调节窗口，需要多个rtt才能完成，而timely只需要两个rtt就开始执行调整
[图片]
[图片]

我们的研究思路，1.各场景最优算法排查；2.理想情况下的多cc性能测试（仿真开始前就设置好对应的最优算法）；3.真实的多cc性能测试（仿真的过程中识别incast并切换算法）

DCQCN减速增速幅度，单次降速最多降到 50%（或 m_minRate），最少100mb/s；单次恢复步幅由 targetRate 增量决定（主动期 +5Mb/s，超激进期 +50Mb/s），实际 m_rate 每步大约向新的 targetRate 靠拢一半（即单步上升量约 ≤ m_rhai/2）。
dcqcn有个定时器，1500us检查一次有没有收到cnp，有cnp降速，再增速。

dctcp每个窗口都会检查一次有没有cnp，有则降速。没有cnp前加性增速。
TIMELY 用 RTT 绝对值 + RTT 梯度决定增/减速；增速为定量加法（5Mb/s→50Mb/s），降速为乘性缩放（由 beta、THigh 或 gradient 决定），所有更新按“完整 RTT 样本”节奏发生。


TIMELY 在无拥塞时会持续按 RTT 规则主动增速；
DCQCN 在这份实现里无拥塞时不单独做主动增速，主要是在收到 CNP 后降速，并在降速后通过恢复阶段再增速。
3.13
1.各场景最优算法排查
（1）无incast实验（各场景普通流最优算法）
暂时无法在飞书文档外展示此内容
0-100KB
[图片]
短流场景 dcqcn不好的原因在于增速定时器300us，远远超过短流的生命周期，短流一旦降速，速率一直压在低位
timely不好的原因在于其依赖rtt梯度，要多个rtt样本，短流的生命周期很短，样本少，梯度噪声大。单词排队抖动就会误判为拥塞，导致不必要的降速；降速后需要多阶段增速，短流通常维持不到后期。timely需要时间观察rtt趋势并收敛

DCQCN和TIMELY在短流场景不好的原因在于恢复速度时提速慢，需要长时间收敛，超过了短流的生命周期
100KB-1MB
[图片]
中流一个明显问题，search模型低负载timely最优，高负载timely因负载的变化时高时低

cache模型长流占29%，mine长流占8%，search长流占20%，

在 TIMELY 中，拥塞信号来自 RTT 绝对值和 RTT 变化率（gradient）。
长流会更持续地占用队列资源，因此会抬高并扰动同端口上的 RTT 基线。当长流与中流哈希到同一输出端口时，中流看到的 RTT 信号会混入长流造成的排队噪声，进而更容易被 TIMELY 判定为“需要降速”；而 TIMELY 的恢复又是渐进的，所以中流可能出现“降得快、回得慢”的现象。

因此：

长流占比越高，数量越多，TIMELY 对中流的干扰通常越明显；
在 低负载 下（如 search 低负载），RTT 信号较干净，TIMELY 仍可能是中流最优；
随着 负载升高，长流引入的持续排队与噪声增强，TIMELY 对中流的性能会明显劣化。

>1MB
[图片]
ALL FLOW AVG.FCT
[图片]

（2）插入200个平均500KB的incast实验
1.对各场景的incast流，timely算法效果最好

暂时无法在飞书文档外展示此内容
0-100kb
[图片]
100kb-1mb
[图片]
[图片]
加入incast场景，mine最优算法不再是timely，原因又是在于incast流瞬时的排队延迟，导致其他流发生了拥塞误判，其他流乘性降速，在瞬时拥塞快速褪去后，其他流速度不能快速恢复
>1mb
[图片]
incast流
[图片]
All flow avg.fct
[图片]

（3）插入三轮不同量级incast
0-100KB   40条incast流
100KB-1MB 50条incast流
1MB-2MB  30条incast流
1.对各场景的incast流，timely的fct有波动，但总体上是timely最好

暂时无法在飞书文档外展示此内容


(4)最优算法对比
无incast影响的最优算法设置
不受incast影响时，最优算法设置相对稳定
暂时无法在飞书文档外展示此内容
在 TIMELY 中，拥塞信号来自 RTT 绝对值和 RTT 变化率，适用于流量平滑、队列变化真实清晰的场景，mine模型的长流最长，长流数量少（cache模型长流占29%，mine长流占8%，search长流占20%），且没有incast，rtt信号干净且不会被突发噪声污染。
对RTT变化的感知是双刃剑，对incast流而言能很快察觉队列变化趋势并提前作出调整，但是这种队列突变得到的rtt变化率对其他流来说就是噪声
受200个500KB 多打一流影响的最优算法设置 
仿真时间:0.01秒（总流量:0.3负载3.6GB, 0.9负载10.8GB）incast 100MB
1.对短暂突发incast流，timely最优
TIMELY对于短暂突发incast比较好，是因为队列瞬间抬高，rtt立刻变大，timely能马上乘性减速，对这种队列尖峰，timely往往能更快压住队列长度，FCT会更好。这种情况rtt信号比较干净。
其他算法对incast流不好的原因在于，他们看到的是已经发出去的包在交换机上经历的状态；而timely看到的是刚发生的排队结果
暂时无法在飞书文档外展示此内容

一场仿真受三轮不同规模多打一影响的最优算法设置 
仿真时间0.05秒 (总流量:0.3负载18GB, 0.9负载54GB) 15ms一次 incast 80MB
1.对这种持续反复或持续时间长的incast，多数场景timely最优，部分场景会劣化为最差。多个发送端持续发送较久，或者多批 incast 连续出现，RTT 表现成连续高位，或者高低反复跳，rtt信号不是单纯的排队尖峰，而是持续的拥塞多流的互相影响。timely容易减速，刚恢复又减速，形成锯齿振荡
暂时无法在飞书文档外展示此内容

测多cc时我所设置的最优算法
暂时无法在飞书文档外展示此内容
多cc在有些场景不能达到最优的原因可能是1. 各模型中流的最优算法难以确定，incast强度大小，终端主机通信配对规则，都会影响中流的最优算法设置。
2.即使多cc最优算法设置和各场景单种算法摸排出来的最优算法一致，多cc在某些场景中仍然不是最优算法，可能误差在于
（1）单算法最优是“全网都用同一种算法”下测出来的。多CC时不同算法会互相改变队列形态（ECN比例、RTT波动、INT利用率），所以每类流的“局部最优”叠加后不一定是全局最优。
（2）多控制信息的开销同时启用多种拥塞信息ECN标记、INT标记和时间戳标记同时加到数据包中会有额外开销
（3）反馈时域不同，各算法“多久观测/决策一次、信号延迟多大”不同，hpcc反应快，dctcp反应慢，dctcp在终端获取到网络的状态时，网络状态已经因hpcc快速反应而变化了
[图片]
[图片]
（4）速率生效路径不同 ，决策后“新速率何时真正作用到发包节奏”不同，hpcc作出决策更新目标速率后会立刻把新速率推到调度器，设置下一次发包间隔，但是timely/dcqcn/dctcp直接改目标速率但是发包调度器没有立刻生效。
某个交换机出口队列有50KB，500KB的流，小流用的hpcc，中流用的dctcp，多cc此时发生拥塞，小流hpcc已经作出降速的反馈了，中流的dctcp还在往队列里发数据。单hpcc此时发生拥塞，会立刻把这个队列中50KB和500KB的流都停掉，多cc反而做不到精准的拥塞控制了，这能解释0-100KB区间多cc一定程度劣化
2. 多cc简单融合测试（仍存在误差）
三轮不同量级incast场景
0-100KB   40条incast流
100KB-1MB 50条incast流
1MB-2MB  30条incast流
根据已知流先验信息，直接对特定流设置相应的最优算法。
但在真实的数据中心环境，对incast的感知是有时间差的，真实多cc融合只能尽可能接近理想测试结果
多cc最优算法设置
（1）0-100KB区间(多cc和dctcp、hpccpint差不多)
[图片]

（2）100KB-1MB区间（多cc不是第一就是第二好）
[图片]
[图片]
（3）>1MB区间（多cc基本上第二好）
[图片]
（4）Incast流
[图片]
（5）ALL FLOW AVG.FCT（多cc维持在第二好的水平，个别情况 能达到最优）
多cc算法各场景没有特别明显的短板
[图片]
3.真实多cc融合测试（incast识别和算法切换）


1) 交换机侧 Incast 识别算法（队列梯度变化率 + 原目的地址匹配）

1.1 观测对象与键
- 键（聚合粒度）：orig_dip（原目的地址）。
- 对每个出口队列维护：
  - q(t)：队列长度（包或字节）
  - dq/dt：队列变化率（建议 EWMA 平滑）
- 对每个 orig_dip 维护：
  - S(orig_dip, W)：窗口 W 内活跃源集合（源 IP 去重）
  - Nsrc = |S|
    
1.2 判定条件
当同时满足以下条件，判定 orig_dip 进入 incast：
1. Nsrc(orig_dip, W) >= K（多源汇聚）
2. dq/dt >= θ_up（队列快速增长）
3. q(t) >= Qmin（避免低负载误判）
  

1.3 原目的地址匹配
- 每个数据包解析 orig_dip（隧道场景取内层目的地址；非隧道取 IP 目的地址）。
- 识别状态按 orig_dip 维护，确保“同一接收端聚合”被准确抓到，而不是仅按当前转发表目标口聚合。
  

---

2) Incast 信息同步到终端的方法

2.1 同步链路
基于 EIN 的端到端同步：

1. 交换机/网络中间节点给数据包打 EinTag  
2. 接收端 ReceiveUdp() 检测到 EinTag 后，在 ACK 里置 FLAG_EIN（seqh.SetEin()）  
3. 发送端 ReceiveAck() 读到 ein=1，若 EnableEinTimelySwitch=true：
  - qp->m_cc_mode = m_einTimelyTargetCc（默认 7 = TIMELY）
  - 并初始化 TIMELY 状态：m_curRate/m_lastUpdateSeq/m_incStage/rttDiff/lastRtt
    
这条链路是：交换机对incast流标记EIN -> 接收端回显 ACK -> 发送端切 CC，不需要额外控制面。


  

---
这个incast识别算法涉及到4个调参
N:队列梯度的采样数
Th：识别为incast的队列梯度阈值
K：一定时间窗口内的活跃流数
W：时间窗口（捕捉fan-in/多打一的时间区间）
N=50 Th=0.7 K=3 
W对识别的准确率影响较大，W=100ns时，真实多cc的fct（282.718）非常贴近理论多cc的fct（281.114），识别incast准确率很接近
在cache0.3负载三轮incast场景的调参测试
[图片]
[图片]


3.21
（1）无incast实验（各场景普通流最优算法）
0-100KB
[图片]
短流场景 dcqcn不好的原因在于增速定时器300us，远远超过短流的生命周期，短流一旦降速，速率一直压在低位
timely不好的原因在于其依赖rtt梯度，要多个rtt样本，短流的生命周期很短，样本少，梯度噪声大。单词排队抖动就会误判为拥塞，导致不必要的降速；降速后需要多阶段增速，短流通常维持不到后期。timely需要时间观察rtt趋势并收敛

DCQCN和TIMELY在短流场景不好的原因在于恢复速度时提速慢，需要长时间收敛，超过了短流的生命周期
100KB-1MB
暂时无法在飞书文档外展示此内容

[图片]
中流一个明显问题，search模型低负载timely最优，高负载timely因负载的变化时高时低

cache模型长流占29%，mine长流占8%，search长流占20%，

在 TIMELY 中，拥塞信号来自 RTT 绝对值和 RTT 变化率（gradient）。
长流会更持续地占用队列资源，因此会抬高并扰动同端口上的 RTT 基线。当长流与中流哈希到同一输出端口时，中流看到的 RTT 信号会混入长流造成的排队噪声，进而更容易被 TIMELY 判定为“需要降速”；而 TIMELY 的恢复又是渐进的，所以中流可能出现“降得快、回得慢”的现象。

因此：

长流占比越高，数量越多，TIMELY 对中流的干扰通常越明显；
在 低负载 下（如 search 低负载），RTT 信号较干净，TIMELY 仍可能是中流最优；
随着 负载升高，长流引入的持续排队与噪声增强，TIMELY 对中流的性能会明显劣化。

>1MB
[图片]
ALL FLOW AVG.FCT
[图片]

（2）插入200个平均500KB的incast实验
1.对各场景的incast流，timely算法效果最好

0-100KB
[图片]
100KB-1MB
[图片]
暂时无法在飞书文档外展示此内容

[图片]
加入incast场景，mine最优算法不再是timely，原因又是在于incast流瞬时的排队延迟，导致其他流发生了拥塞误判，其他流乘性降速，在瞬时拥塞快速褪去后，其他流速度不能快速恢复
>1MB
[图片]
incast流
[图片]
All flow avg.fct
[图片]

（3）插入三轮不同量级incast
0-100KB   40条incast流
100KB-1MB 50条incast流
1MB-2MB  30条incast流
1.对各场景的incast流，timely的fct有波动，但总体上是timely最好

0-100KB区间
[图片]

100KB-1MB区间
[图片]
暂时无法在飞书文档外展示此内容

[图片]
>1MB区间
[图片]
Incast流
[图片]
ALL FLOW AVG.FCT（多cc维持在第二好的水平，个别情况 能达到最优）
多cc算法各场景没有特别明显的短板
[图片]

一、最优算法对比
无incast最优算法设置
不受incast影响时，最优算法设置相对稳定
暂时无法在飞书文档外展示此内容
在 TIMELY 中，拥塞信号来自 RTT 绝对值和 RTT 变化率，适用于流量平滑、队列变化真实清晰的场景，mine模型的长流流量大，数量少（cache模型长流占29%，mine长流占8%，search长流占20%），且没有incast，rtt信号干净且不会被突发噪声污染。
对RTT变化的感知是双刃剑，对incast流而言能很快察觉队列变化趋势并提前作出调整，但是这种队列突变得到的rtt变化率对其他流来说就是噪声

0-100KB 短流，>1MB长流，以及incast流的最优算法都好解释，但是100KB-1MB中流最优算法受incast强度，流量模型，负载度影响，不好解释。
hpccpint能通过交换机提供的精确、实时的队列与链路负载信息，实现最快速度的反饋和速率调整。对于生命周期极短、对排队延迟异常敏感的短流友好
DCTCP 的 ECN 标记机制具有滤波的特性，DCTCP 能忽略掉大量由短流突发引起的、短暂无害的队列波动，专注于应对真正会导致拥塞恶化的、持续性的队列增长
一场仿真一轮多打一的最优算法设置 （200个500KB）
仿真时间:0.01秒（总流量:0.3负载3.6GB, 0.9负载10.8GB）incast总流量 100MB
结论1.对一轮突发incast流，timely最优
TIMELY算法能有效处理短暂突发性Incast流量，原因在于其独特的设计机制：它对RTT的绝对值及其变化趋势高度敏感。在Incast场景下，大量流同时到达会导致网络队列瞬间被填满，RTT随即急剧上升。TIMELY能够立即感知到这种队列尖峰，并根据RTT的剧烈变化趋势以乘性方式快速降低发送速率，从而迅速抑制队列的进一步增长。
通过这种快速反应，TIMELY能将原本在同一时间点爆发、相互剧烈竞争的Incast流，在时间上“打散”到多个稍有不同的时刻，有效缓解了瞬时拥塞的严重程度，降低了丢包风险，最终实现了更优的整体性能。
其他算法对incast流不好的原因在于，他们看到的是已经发出去的包在交换机上经历的状态；而timely看到的是刚发生的排队结果
暂时无法在飞书文档外展示此内容
表格统计数据流不含incast流

一场仿真三轮不同规模多打一的最优算法设置 
仿真时间0.05秒 (总流量:0.3负载18GB, 0.9负载54GB) 15ms一次 incast 80MB
结论2.对多轮持续反复incast，多数场景timely最优，部分场景会劣化为最差。
在多轮持续爆发的Incast场景中，网络RTT呈现持续高位震荡或频繁的高低跳变。此时，拥塞信号已非单次突发形成的短暂排队尖峰，而是多流之间持续、动态的竞争所导致的复杂干扰。
TIMELY算法基于RTT及其变化趋势进行控制的特性，在此类场景中表现出双重性：
- 在多数情况下，其对RTT变化的敏感性仍能帮助其快速响应每一轮拥塞的加剧，从而在动态竞争中获得优势。
- 但在部分场景下，持续的RTT波动会误导TIMELY做出过度反应：在拥塞稍有缓解、速率刚开始恢复时，下一轮RTT的上升又会触发其再次乘性减速。这种“减速-恢复-又减速”的循环，导致其发送速率呈现出剧烈的锯齿状振荡，无法稳定地利用带宽，从而使其性能反而劣化为最差。
暂时无法在飞书文档外展示此内容
表格统计数据流不含incast流


测多cc时我所设置的最优算法
结论3.多cc在0-100KB,大于1MB区间不是最优，100KB-1MB区间部分场景不是最优 
[图片]
hpcc逐包评估链路利用率后调速，dctcp每rtt窗口统计一次拥塞包比例后调速。
多cc中，同一个链路发生拥塞，dctcp反应比hpccpint慢，hpccpint已经降速，dctcp还在往链路发数据，加重拥塞，hpccpint继续降速，此时多cc没有单hpcc拥塞控制精准。这可以解释0-100KB区间多cc设置hpccpint但效果次于dctcp。
dctcp开始降速，hpccpint察觉到链路拥塞缓解，又开始加速往链路发数据，此时多cc没有单dctcp拥塞控制精准，这可以解释大于>1MB区间多cc设置dctcp效果次于最优。
暂时无法在飞书文档外展示此内容

中流部分场景不能达到最优的原因1. 各模型中流的最优算法难以确定，incast强度，流量模型，负载度，都会影响中流的最优算法设置。我的多cc最优算法设置还未达到各因素影响下的最优。
2.即使多cc最优算法设置和各因素影响下的最优单cc算法一致，多cc在某些场景中仍然不是最优算法，可能误差在于

挑战一：动态博弈下的“最优”失效
- 核心问题：算法的“局部最优”是在静态环境中测得的。 一旦混合，A算法的行为会改变B算法赖以生存的网络环境（如队列形态、ECN标记率、RTT波动）。B算法为适应新环境而调整，又会反过来影响A。这是一个动态博弈过程。
- 您提出的例证：DCTCP依赖稳定的ECN比例，但当 HPCC这类快速算法引入后，会剧烈扰动队列，导致ECN标记模式失准，从而使 DCTCP的“最优”性能无法实现。最终全局表现可能远差于任何单一算法的“局部最优”组合。
挑战二：控制平面的信号冲突与延迟差
- 核心问题：不同算法的控制环路时域（观测、决策、执行的频率和延迟）不同，在混合部署时会产生“时空错位”。
- 您提出的例证：
  - “快”算法（如HPCC）：观测（INT）-> 决策 -> 生效的闭环极短（微秒级）。
  - “慢”算法（如DCTCP）：观测（ECN）-> 决策 -> 生效的闭环较长（RTT级）。
- 后果：当 DCTCP的终端终于根据旧的ECN信息计算出新速率时，网络状态早已被 HPCC的快速反应所改变。DCTCP的决策基于“过时”的画面，其行动可能是错误甚至有害的，这引发了控制环路之间的不稳定振荡
挑战三：数据平面的信息过载与开销
- 核心问题：为支持多种算法，每个数据包可能需要携带多种拥塞信号（ECN位、INT元数据、高精度时间戳）。这直接增加了包头开销，降低了有效数据吞吐量。在高速网络（如100G+）中，这种开销不可忽视。
挑战四：协议栈的速率生效异步
- 核心问题：即使算法作出了“正确”的速率决策，决策何时真正作用于物理链路上的发包行为，不同算法实现也不同。
- 您提出的例证：
  - 立即生效型（如HPCC）：将新速率直接推送给发包调度器，立即改变下一个包的发送间隔。
  - 延迟生效型（如DCTCP/TIMELY）：只更新一个“目标速率”变量，发包调度器异步地、逐渐地向该目标速率靠拢。
- 后果：在混合场景下，HPCC流能“抢跑”，瞬间抢占刚释放的带宽；而 DCTCP流还在“加速”过程中。这造成了短时间尺度上的不公平性，破坏了算法间公平竞争的假设。

二、真实多cc
（1）incast识别机制
交换机侧 Incast 识别算法（队列梯度变化率 + 原目的地址匹配）
交换机对incast流标记EIN -> 接收端回显 ACK -> 发送端切 CC

这个incast识别算法涉及到4个调参
N:队列梯度的采样数
Th：识别为incast的队列梯度阈值
K：一定时间窗口内的活跃incast流数
W：时间窗口（捕捉fan-in/多打一的时间区间）
N=50 Th=0.7 K=3 
W对识别的准确率影响较大，W=100ns时，真实多cc的fct（282.718）非常贴近理论多cc的fct（281.114），识别准确率很接近
在cache0.3负载三轮incast场景的调参测试
[图片]
[图片]
（2）流大小+incast识别机制
[图片]
终端实时检测每条流已发送的字节数，超过100KB/1MB阈值，则切换算法。初始拥塞控制算法设置为hpccpint
在cache0.3负载三轮incast场景的调参测试
N=50 Th=0.7 K=3 W=100ns
多cc（短流hpccpint； 长流dctcp； 中流尝试了四种拥塞控制算法）
1.当前场景单cc时，dctcp对中流效果好；多cc时也是dctcp对中流效果好
2.仅有incast机制时，多cc的fct（282.718us）比理论多cc的fct（281.114us）多1.6us延迟
流大小识别+incast机制时，多cc的fct（282.773us）只额外增加55ns平均延迟
[图片]
[图片]
