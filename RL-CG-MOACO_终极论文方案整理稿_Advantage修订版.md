# RL-CG-MOACO 终极论文方案整理稿

## 0. 论文最终定位

本文面向**多目标敏捷卫星调度问题**，提出一种强化学习辅助的冲突图引导多目标蚁群优化算法：

$$
\boxed{\text{RL-CG-MOACO}}
$$

即：

$$
\text{Reinforcement Learning assisted Conflict-Graph-guided Multi-Objective Ant Colony Optimization}
$$

中文名称可以写为：

> **强化学习辅助的冲突图引导多目标蚁群优化算法**

最终版本固定为：

$$
\boxed{
\text{任务-卫星-窗口冲突图}
+
\text{MOACO}
+
\text{MLP-DDQN}
+
\text{Pareto 档案}
}
$$

本方案不引入偏好向量 $w$，不引入 GNN，不直接使用 HV 作为强化学习主要奖励。核心思想是：

> 将任务-卫星-窗口候选方案建模为冲突图节点，利用冲突图动作掩码保证蚂蚁构造过程的可行性，并在 MOACO 的逐节点构造阶段引入 MLP-DDQN 学习候选节点的状态相关长期价值。最终由信息素和冲突图启发式给出 ACO 基础吸引力，再由 DDQN 输出的相对优势项对节点选择概率进行状态相关校正，引导蚂蚁构造高质量 Pareto 调度方案。

---

## 1. 问题建模

### 1.1 任务集合

设任务集合为：

$$
T=\{task_1,task_2,\dots,task_m\}
$$

每个任务 $task_j$ 具有：

- 任务收益或优先级 $p_j$；
- 观测持续时间 $d_j$；
- 观测时间约束；
- 可能的姿态角需求；
- 可观测卫星集合。

### 1.2 卫星集合

设卫星集合为：

$$
S=\{sat_1,sat_2,\dots,sat_n\}
$$

每颗卫星具有：

- 可用观测时间窗口；
- 姿态机动能力；
- 任务负载；
- 可选的能量、存储、数传等资源限制。

### 1.3 时间窗口集合

对于任务 $task_j$ 和卫星 $sat_i$，可能存在一个或多个可见时间窗口：

$$
W_{j,i}=\{window_{j,i,1},window_{j,i,2},\dots\}
$$

每个窗口包含：

$$
window_{j,i,k}=(start_{j,i,k},end_{j,i,k})
$$

---

## 2. 任务-卫星-窗口候选节点编码

### 2.1 三元节点定义

将每个候选观测方案表示为一个三元节点：

$$
v_{j,i,k}=(task_j,sat_i,window_k)
$$

含义为：

> 任务 $j$ 由卫星 $i$ 在第 $k$ 个观测窗口执行。

所有候选节点构成节点集合：

$$
V=\{v_1,v_2,\dots,v_N\}
$$

### 2.2 决策变量

定义二元决策变量：

$$
x_{j,i,k}=
\begin{cases}
1, & \text{如果任务 } j \text{ 被安排给卫星 } i \text{ 在窗口 } k \text{ 执行}\\
0, & \text{否则}
\end{cases}
$$

一个完整调度方案可以表示为候选节点子集：

$$
X\subseteq V
$$

---

## 3. 多目标优化模型

本文考虑三个目标，统一写成最小化形式。

### 3.1 目标一：最小化未完成收益率

$$
f_1(X)=1-\frac{\sum_j p_jx_j}{\sum_j p_j}
$$

其中：

- $p_j$：任务 $j$ 的收益；
- $x_j=1$：表示任务 $j$ 被成功调度；
- $f_1$ 越小，表示完成的任务收益越高。

### 3.2 目标二：最小化姿态机动代价

$$
f_2(X)=\sum_i\sum_{(a,b)\in Seq_i}Tr_{a,b}^{i}
$$

其中：

- $Seq_i$：卫星 $i$ 的任务执行序列；
- $Tr_{a,b}^{i}$：卫星 $i$ 从任务 $a$ 转向任务 $b$ 的姿态转移代价或转移时间。

### 3.3 目标三：最小化卫星负载不均衡

$$
f_3(X)=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}(L_i-\bar{L})^2
}
$$

其中：

- $L_i$：卫星 $i$ 的任务负载；
- $\bar{L}$：所有卫星的平均负载；
- $n$：卫星数量。

### 3.4 最终多目标优化模型

$$
\min F(X)=\big(f_1(X),f_2(X),f_3(X)\big)
$$

---

## 4. 约束条件与冲突图建模

### 4.1 基本约束

#### 4.1.1 每个任务最多执行一次

$$
\sum_i\sum_k x_{j,i,k}\leq 1
$$

#### 4.1.2 同一卫星观测窗口不能重叠

对于同一颗卫星 $sat_i$，若两个节点 $u$ 和 $v$ 的时间窗口重叠，则不能同时选择。

#### 4.1.3 姿态转移时间约束

若卫星 $i$ 连续执行任务 $a$ 和任务 $b$，需要满足：

$$
end_a+Tr_{a,b}^{i}\leq start_b
$$

否则两个节点存在冲突。

### 4.2 冲突图定义

构建冲突图：

$$
G=(V,E)
$$

其中：

- $V$：所有任务-卫星-窗口候选节点；
- $E$：候选节点之间的冲突边。

若两个候选节点不能同时被选择，则添加边：

$$
(u,v)\in E
$$

### 4.3 冲突类型

冲突边主要来自以下情况：

1. **任务重复冲突**：两个节点对应同一个任务；
2. **时间窗口冲突**：两个节点属于同一颗卫星且时间窗口重叠；
3. **姿态转移冲突**：两个节点属于同一颗卫星，但连续执行时姿态转移时间不足；
4. **可选资源冲突**：能量、存储、数传等资源不足时也可转化为冲突或资源可行性检查。

### 4.4 可行解与独立集关系

一个调度方案 $X\subseteq V$ 可行，当且仅当其中任意两个节点之间不存在冲突边：

$$
X \text{ is feasible}
\Leftrightarrow
\forall u,v\in X,\ (u,v)\notin E
$$

也就是说：

$$
\boxed{
\text{可行调度方案 }X\text{ 对应冲突图中的一个独立集}
}
$$

---

## 5. 最终创新点设计

为避免创新点过多和重复，最终建议写成四个核心创新点。

### 5.1 创新点一：任务-卫星-窗口三元冲突图建模

本文将每个候选观测方案表示为三元节点：

$$
v_{j,i,k}=(task_j,sat_i,window_k)
$$

并将任务重复、时间窗口重叠、姿态转移不足等复杂调度约束显式转化为冲突边，构建冲突图：

$$
G=(V,E)
$$

该建模将复杂约束调度问题转化为冲突图上的可行节点选择问题，为后续可行动作掩码、蚁群构造和强化学习节点选择提供结构基础。

### 5.2 创新点二：基于冲突图动作掩码的可行构造机制

蚂蚁构造过程中，当前已选节点集合为：

$$
X_t
$$

对于候选节点 $v$，如果：

$$
N(v)\cap X_t\neq \emptyset
$$

说明节点 $v$ 与当前部分解冲突，不能被选择。

因此定义当前可行动作集合：

$$
A_t=
\{v\in V\mid v\notin X_t,\ N(v)\cap X_t=\emptyset\}
$$

其中：

- $N(v)$：节点 $v$ 在冲突图中的邻域；
- $A_t$：当前可选的无冲突节点集合。

该机制保证：

$$
X_t
$$

在构造过程中始终保持可行。

它的核心思想是：

$$
\boxed{
\text{RL 不学习硬约束，而是在冲突图掩码后的可行动作空间中学习选择策略。}
}
$$

### 5.3 创新点三：MLP-DDQN 辅助的候选节点选择策略

本文将蚂蚁逐步构造调度方案的过程建模为马尔可夫决策过程，即 MDP。

状态 $s_t$ 表示当前构造局面，动作 $a_t$ 表示选择下一个候选节点：

$$
a_t=v_{j,i,k}
$$

使用 MLP 近似 DDQN 的动作价值函数：

$$
Q_\theta(s_t,v)=MLP([g_t,h(v)])
$$

其中：

- $g_t$：当前全局构造状态特征；
- $h(v)$：候选节点特征；
- $Q_\theta(s_t,v)$：当前状态下选择节点 $v$ 的长期价值。

该策略学习的是：

$$
\boxed{
\text{在当前构造状态下，哪个候选节点更值得被蚂蚁选择。}
}
$$

### 5.4 创新点四：基于 Advantage 校正的状态感知节点选择机制

传统 MOACO 主要依赖信息素和启发式信息：

$$
\tau(v),\quad \eta(v)
$$

其中，信息素 $\tau(v)$ 反映历史优秀解中的群体搜索经验，启发式信息 $\eta(v)$ 反映候选节点的静态结构质量。二者可以给出传统 ACO 的基础吸引力：

$$
B_t(v)=\alpha\log \tau(v)+\beta\log \eta(v)
$$

然而，$\tau(v)$ 和 $\eta(v)$ 主要体现历史经验和静态偏好，难以刻画同一候选节点在不同部分解状态下的相对价值。为此，本文使用 MLP-DDQN 估计候选节点的状态相关动作价值：

$$
Q_\theta(s_t,v)
$$

但本文不直接将绝对 $Q$ 值作为额外启发式项加入节点选择概率，而是将其转化为当前可行动作集合 $A_t$ 内的相对优势项：

$$
\bar{Q}_t=
\frac{1}{|A_t|}
\sum_{u\in A_t}Q_\theta(s_t,u)
$$

$$
Adv_\theta(s_t,v)=Q_\theta(s_t,v)-\bar{Q}_t
$$

其中，$Adv_\theta(s_t,v)$ 表示候选节点 $v$ 相对于当前可行动作集合平均水平的长期选择优势。最终节点选择概率定义为：

$$
P(v|s_t)=
\frac{
\exp\left(
B_t(v)+\kappa Adv_\theta(s_t,v)
\right)
}
{
\sum_{u\in A_t}
\exp\left(
B_t(u)+\kappa Adv_\theta(s_t,u)
\right)
}
$$

等价地，也可以写成：

$$
P(v|s_t)=
\frac{
\tau(v)^\alpha\eta(v)^\beta
\exp\left(\kappa Adv_\theta(s_t,v)\right)
}
{
\sum_{u\in A_t}
\tau(u)^\alpha\eta(u)^\beta
\exp\left(\kappa Adv_\theta(s_t,u)\right)
}
$$

其中：

- $B_t(v)$：由信息素和启发式信息得到的 ACO 基础吸引力；
- $Q_\theta(s_t,v)$：DDQN 估计的状态相关长期动作价值；
- $\bar{Q}_t$：当前可行动作集合中的平均动作价值；
- $Adv_\theta(s_t,v)$：候选节点相对于当前可选节点平均水平的优势校正项；
- $\kappa$：DDQN 状态校正项的影响强度。

该机制的核心思想是：

$$
\boxed{
\text{ACO 提供基础偏好，DDQN 不直接给绝对加分，而是提供状态相关的相对优势校正。}
}
$$

因此，DDQN 并不是简单作为第三个启发式加数，而是用于判断当前构造状态下某个候选节点是否比其他可选节点更值得选择，从而将传统 MOACO 的静态节点转移规则扩展为状态感知的自适应节点选择策略。

---

## 6. MDP 建模

### 6.1 状态空间

状态 $s_t$ 由当前部分解、当前可选节点集合、卫星负载和当前目标状态等组成。

实际输入网络时，用全局状态特征向量 $g_t$ 表示：

$$
g_t=[
\widehat{Profit}_t,
\widehat{Maneuver}_t,
\widehat{LoadStd}_t,
\widehat{AvailableRatio}_t,
\widehat{AvgConflict}_t,
\widehat{IterRatio}
]
$$

各项含义如下。

| 特征 | 含义 |
|---|---|
| $\widehat{Profit}_t$ | 当前已获得收益比例 |
| $\widehat{Maneuver}_t$ | 当前累计姿态机动代价 |
| $\widehat{LoadStd}_t$ | 当前卫星负载不均衡程度 |
| $\widehat{AvailableRatio}_t$ | 当前剩余可选节点比例 |
| $\widehat{AvgConflict}_t$ | 当前可选节点平均冲突度 |
| $\widehat{IterRatio}$ | 当前迭代进度 |

### 6.2 动作空间

动作是选择一个当前可行的候选节点：

$$
a_t=v\in A_t
$$

其中：

$$
A_t=
\{v\in V\mid v\notin X_t,\ N(v)\cap X_t=\emptyset\}
$$

### 6.3 状态转移

选择节点 $v_t$ 后：

$$
X_{t+1}=X_t\cup\{v_t\}
$$

同时更新可选节点集合：

$$
A_{t+1}=A_t-\{v_t\}-N(v_t)
$$

该转移由冲突图确定，因此基本是确定性的。

### 6.4 奖励函数

奖励采用：

$$
\boxed{
\text{Warm-up 估计 }T_{ref}
+
\text{缩放即时构造奖励}
+
\text{Pareto 档案终局奖励}
}
$$

详细设计见第 8 节。

### 6.5 折扣因子

建议取：

$$
\gamma=0.9
$$

因为选择一个节点不仅影响当前收益，还会影响后续仍然可选的节点集合。

---

## 7. MLP-DDQN 网络输入设计

### 7.1 动作价值函数

采用 MLP 近似动作价值函数：

$$
Q_\theta(s_t,v)=MLP([g_t,h(v)])
$$

其中输入由两部分拼接而成：

1. 全局状态特征 $g_t$；
2. 候选节点特征 $h(v)$。

### 7.2 候选节点特征

最终固定使用以下节点特征：

$$
h(v)=
[
\hat{p}(v),
\hat{R}(v),
\hat{C}(v),
\hat{M}(v),
\hat{L}(s_i),
\hat{\tau}(v),
\hat{\eta}(v),
BlockPenalty(v)
]
$$

| 特征 | 含义 |
|---|---|
| $\hat{p}(v)$ | 节点对应任务收益归一化值 |
| $\hat{R}(v)$ | 窗口稀缺度 |
| $\hat{C}(v)$ | 节点冲突度 |
| $\hat{M}(v)$ | 插入该节点带来的机动代价 |
| $\hat{L}(s_i)$ | 节点所属卫星当前负载 |
| $\hat{\tau}(v)$ | 信息素值 |
| $\hat{\eta}(v)$ | 冲突图启发式值 |
| $BlockPenalty(v)$ | 选择该节点后对后续可选空间的屏蔽代价 |

### 7.3 网络结构建议

MLP 可以采用如下结构：

```text
Input: [g_t, h(v)]
Linear(input_dim, 128)
ReLU
Linear(128, 128)
ReLU
Linear(128, 64)
ReLU
Linear(64, 1)
Output: Q_theta(s_t,v)
```

### 7.4 不使用 GNN 的原因

最终方案使用 MLP，而不是 GNN，原因是：

1. 本文已经通过冲突度、稀缺度、BlockPenalty、启发式值等显式特征表达了冲突图结构；
2. MLP 更容易实现、训练和解释；
3. 本文创新重点不是复杂神经网络结构，而是 DDQN 如何嵌入 MOACO 的逐节点构造过程；
4. MLP 可以降低模型复杂度，避免论文主线被网络结构分散。

---

## 8. 最终奖励函数设计

### 8.1 奖励设计原则

最终奖励不直接使用 HV 作为主要训练奖励。

原因是：

1. HV 是整个 Pareto 档案的指标，不是单个动作的直接反馈；
2. HV 奖励过于稀疏；
3. HV 很难分配到每一步节点选择动作；
4. HV 对参考点敏感；
5. 早期和后期 HV 改善幅度差异较大，容易导致训练不稳定。

因此最终采用：

$$
\boxed{
\text{缩放即时构造奖励}+
\text{Pareto 档案终局奖励}
}
$$

HV 主要作为实验评价指标，而不是主要训练奖励。

### 8.2 单步构造评分

先定义单步构造评分 $c_t$：

$$
c_t=
\frac{1}{3}
\left(
\Delta \widehat{Profit}
-
\Delta \widehat{Maneuver}
-
\Delta \widehat{Load}
\right)
-
0.2\cdot BlockPenalty(v)
$$

其中：

- $\Delta \widehat{Profit}$：归一化收益提升；
- $\Delta \widehat{Maneuver}$：归一化机动代价增加；
- $\Delta \widehat{Load}$：归一化负载不均衡变化；
- $BlockPenalty(v)$：冲突屏蔽惩罚。

### 8.3 收益归一化

$$
\Delta \widehat{Profit}
=
\frac{p(v)}{p_{\max}}
$$

其中：

$$
p_{\max}=\max_{v\in V}p(v)
$$

### 8.4 机动代价归一化

$$
\Delta \widehat{Maneuver}
=
\frac{\Delta Maneuver(v)}{Tr_{\max}+\epsilon}
$$

其中：

$$
Tr_{\max}=\max_{i,a,b}Tr_{a,b}^{i}
$$

如果没有真实姿态转移矩阵，则用候选任务对之间估计得到的最大姿态转移代价作为 $Tr_{\max}$。

### 8.5 负载变化归一化

当前负载标准差为：

$$
LoadStd_t
$$

选择节点后变为：

$$
LoadStd_{t+1}
$$

负载变化为：

$$
\Delta Load=LoadStd_{t+1}-LoadStd_t
$$

归一化为：

$$
\Delta \widehat{Load}
=
clip\left(
\frac{LoadStd_{t+1}-LoadStd_t}{LoadStd_{\max}+\epsilon},
-1,
1
\right)
$$

如果 $\Delta \widehat{Load}>0$，说明负载不均衡变差；如果 $\Delta \widehat{Load}<0$，说明负载均衡变好。由于奖励中使用：

$$
-\Delta \widehat{Load}
$$

因此负载均衡改善会得到正向奖励。

### 8.6 冲突屏蔽惩罚

记任务 $j$ 在当前候选集中的可用节点集合为

$$
A_t^j=\{u\in A_t\mid task(u)=j\}.
$$

选择节点 $v$ 后被完全屏蔽的未执行任务集合为

$$
B_t(v)=
\left\{
j\ne task(v)\mid
A_t^j\ne\varnothing,
A_t^j\subseteq N(v)
\right\}.
$$

于是冲突屏蔽惩罚定义为

$$
BlockPenalty(v)=
\frac{
\sum_{j\in B_t(v)}p_j
}
{
\sum_{j:A_t^j\ne\varnothing}p_j
}
$$

含义是：

> 当前节点 $v$ 虽然可以选，但如果选了它，会使多少后续高收益任务失去全部可用窗口。

同一任务的多个卫星或时间窗口候选只计算一次收益。被选任务自身的替代窗口不计入惩罚；其他任务若仍保留至少一个不冲突窗口，也不视为被屏蔽。由于：

$$
B_t(v)\subseteq\{j\mid A_t^j\ne\varnothing\}
$$

所以：

$$
BlockPenalty(v)\in[0,1]
$$

该项天然归一化。

### 8.7 单步评分裁剪

为防止异常值影响训练，对 $c_t$ 进行裁剪：

$$
c_t\leftarrow clip(c_t,-1,1)
$$

### 8.8 Warm-up 阶段估计参考构造长度

如果一只蚂蚁需要经过多步节点选择才能构造完整解，则未缩放的即时奖励累计值可能远大于终局奖励。

例如，一个解需要 40 步构造，如果每步即时奖励最大接近 1，则累计即时奖励可能接近 40，而终局奖励只有 1。这样会导致 Pareto 档案终局奖励影响过弱。

因此，本文在 Warm-up 阶段估计参考构造长度 $T_{ref}$。

Warm-up 阶段只使用 $\tau+\eta$ 构造解，不让 DDQN 影响选择。每只蚂蚁构造完整解后，记录其构造步数：

$$
T_e=|X_e|
$$

假设 Warm-up 阶段共生成 $N$ 个解，则：

$$
T_{ref}=
\frac{1}{N}
\sum_{e=1}^{N}T_e
$$

也可以取整数：

$$
T_{ref}=round\left(
\frac{1}{N}
\sum_{e=1}^{N}T_e
\right)
$$

### 8.9 缩放即时奖励

最终单步即时奖励为：

$$
r_t=
\frac{c_t}{T_{ref}}
$$

这样整条路径上的累计即时奖励大约为：

$$
\sum_{t=1}^{T}r_t
\approx
O(1)
$$

与终局奖励 $+1$ 保持相近数量级。

### 8.10 Pareto 档案终局奖励

一只蚂蚁构造完整解 $X$ 后，更新 Pareto 档案。如果该解进入 Pareto 档案，则给终局奖励：

$$
r_{final}=
\begin{cases}
+1, & X \text{ enters Pareto archive}\\
0, & \text{otherwise}
\end{cases}
$$

终局奖励只加在 episode 的最后一步，而不是加到每一步。

### 8.11 最终训练奖励

最终训练奖励为：

$$
R_t=
\begin{cases}
\frac{clip(c_t,-1,1)}{T_{ref}}, & t<T\\[8pt]
\frac{clip(c_t,-1,1)}{T_{ref}}+r_{final}, & t=T
\end{cases}
$$

其中：

$$
r_{final}=
\begin{cases}
+1, & X \text{ enters Pareto archive}\\
0, & \text{otherwise}
\end{cases}
$$

该奖励机制可以命名为：

> **构造过程感知与 Pareto 档案反馈相结合的混合奖励机制**

英文可写为：

> **Construction-aware and Pareto-archive-based hybrid reward**

---

## 9. DDQN 在线训练机制

### 9.1 训练方式

本文采用在线训练方式，不提前预训练模型。

每只蚂蚁构造一个完整调度方案的过程视为一个 episode。

每一步存储 transition：

$$
(s_t,v_t,R_t,s_{t+1})
$$

### 9.2 Double DQN 更新

对于下一状态 $s_{t+1}$，先用当前网络选择动作：

$$
v^*=
\arg\max_{u\in A_{t+1}}
Q_\theta(s_{t+1},u)
$$

再用目标网络估计该动作的价值：

$$
y_t=
R_t+\gamma Q_{\theta^-}(s_{t+1},v^*)
$$

损失函数为：

$$
Loss=
\left(
 y_t-Q_\theta(s_t,v_t)
\right)^2
$$

定期同步目标网络：

$$
\theta^-\leftarrow\theta
$$

### 9.3 Replay Buffer

经验回放池中存储：

$$
(s_t,v_t,R_t,s_{t+1},A_{t+1})
$$

其中 $A_{t+1}$ 用于计算下一状态下的最大 Q 值。

---

## 10. 300 代训练安排

如果总迭代数为 300，最终采用如下安排。

| 阶段 | 代数 | 主要作用 | DDQN 是否参与选择 | 说明 |
|---|---:|---|---|---|
| Warm-up | 1–30 | 统计 $T_{ref}$，收集原始轨迹 | 否 | 只用 $\tau+\eta$ 构造解 |
| 轻度 RL | 31–90 | DDQN 开始训练并轻度参与 | 是 | $\kappa:0.1\rightarrow0.4$ |
| 强 RL | 91–240 | DDQN 正式辅助构造 | 是 | $\kappa:0.4\rightarrow1.0$ |
| 收敛训练 | 241–280 | 降低学习率，稳定策略 | 是 | $\kappa:1.0\rightarrow1.2$ |
| 冻结输出 | 281–300 | 冻结 DDQN，只使用网络辅助构造 | 是 | 不再更新网络参数 |

### 10.1 Warm-up 阶段

第 1–30 代：

- 不使用 DDQN 参与节点选择；
- 使用 $\tau+\eta$ 构造解；
- 记录每个 episode 的构造长度 $T_e$；
- 计算 $T_{ref}$；
- 可以保存原始轨迹 $(s_t,v_t,c_t,s_{t+1})$。

Warm-up 结束后：

$$
T_{ref}=round\left(\frac{1}{N}\sum_{e=1}^{N}T_e\right)
$$

并将原始轨迹中的 $c_t$ 转化为正式奖励：

$$
R_t=\frac{clip(c_t,-1,1)}{T_{ref}}
$$

### 10.2 RL 参与阶段

第 31 代之后，DDQN 开始参与节点选择。

对于当前可行动作集合 $A_t$，先计算每个候选节点的动作价值：

$$
Q_\theta(s_t,v),\quad v\in A_t
$$

然后计算当前可行动作集合内的平均动作价值：

$$
\bar{Q}_t=
\frac{1}{|A_t|}
\sum_{u\in A_t}Q_\theta(s_t,u)
$$

并得到相对优势校正项：

$$
Adv_\theta(s_t,v)=Q_\theta(s_t,v)-\bar{Q}_t
$$

节点选择概率为：

$$
P(v|s_t)=
\frac{
\tau(v)^\alpha\eta(v)^\beta
\exp\left(\kappa Adv_\theta(s_t,v)\right)
}
{
\sum_{u\in A_t}
\tau(u)^\alpha\eta(u)^\beta
\exp\left(\kappa Adv_\theta(s_t,u)\right)
}
$$

其中 $\kappa$ 随迭代逐渐增大，用于控制 DDQN 状态校正项对节点选择的影响强度。

### 10.3 推荐参数

| 参数 | 建议值 |
|---|---:|
| 总迭代数 | 300 |
| 蚂蚁数量 | 30–50 |
| $\alpha$ | 1 |
| $\beta$ | 2 |
| $\rho$ | 0.1 |
| $\gamma$ | 0.9 |
| 学习率 | $1e^{-3}$，后期降到 $5e^{-4}$ 或 $1e^{-4}$ |
| Batch size | 64 或 128 |
| Replay Buffer 容量 | 10000–50000 |
| Target network 同步 | 每 10 或 20 代 |
| BlockPenalty 系数 | 0.2 |
| $\kappa$ | 0 到 1.2 阶段式增长 |
| $\epsilon$-greedy 探索率 | 0.30 衰减到 0.02–0.05 |

---

## 11. 信息素更新机制

### 11.1 信息素挥发

每轮迭代后执行信息素挥发：

$$
\tau(v)\leftarrow(1-\rho)\tau(v)
$$

### 11.2 Pareto 档案信息素增强

对 Pareto 档案中的非支配解进行信息素增强：

$$
\tau(v)\leftarrow \tau(v)+
\sum_{X\in Archive,\ v\in X}
Q\cdot\omega(X)\cdot\phi(v)
$$

其中：

- $\omega(X)$：解 $X$ 的 Pareto 贡献；
- $\phi(v)$：节点贡献；
- $Q$：信息素增强系数。

为了降低复杂度，最终版本可以先采用简化形式：

$$
\omega(X)=1
$$

即只要解在 Pareto 档案中，就对其包含的节点进行信息素强化。

---

## 12. 轻量级局部搜索

加入 DDQN 后，局部搜索作为辅助增强模块，而不是最核心创新点。

### 12.1 快速插入

对于未调度任务对应的候选节点 $v$，如果：

$$
N(v)\cap X=\emptyset
$$

则可以直接插入：

$$
X\leftarrow X\cup\{v\}
$$

### 12.2 有限替换

如果节点 $v$ 无法直接插入，则计算其冲突集合：

$$
ConflictSet(v)=N(v)\cap X
$$

若删除少量冲突节点并插入 $v$ 后可以改善解，则执行替换。

替换增益可以定义为：

$$
MGain=
\Delta \widehat{Profit}
-
\Delta \widehat{Maneuver}
-
\Delta \widehat{Load}
$$

为避免局部搜索过重，建议：

- 只对收益较高的前 $K$ 个未调度任务尝试；
- 每只蚂蚁替换次数限制在 5–10 次；
- 局部搜索作为消融实验模块。

---

## 13. 算法整体流程

### 13.1 文字流程

1. 读取任务、卫星、时间窗口和姿态转移数据；
2. 生成任务-卫星-窗口候选节点集合 $V$；
3. 根据约束构建冲突图 $G=(V,E)$；
4. 计算节点冲突度、稀缺度、启发式值和 BlockPenalty；
5. 初始化信息素、Pareto 档案、MLP-DDQN 网络和经验回放池；
6. 进入 Warm-up 阶段，使用 $\tau+\eta$ 构造解，统计 $T_{ref}$；
7. Warm-up 后开始 DDQN 在线训练；
8. 每只蚂蚁从空解开始构造调度方案；
9. 每一步根据冲突图动作掩码得到当前可行动作集合 $A_t$；
10. 对每个候选节点计算 $Q_\theta(s_t,v)$；
11. 融合 $\tau(v)$、$\eta(v)$ 和 $Q_\theta(s_t,v)$ 得到选择概率；
12. 选择节点并更新当前解和可行动作集合；
13. 计算缩放后的即时奖励并保存经验；
14. 一只蚂蚁构造完成后执行轻量级替换-插入局部搜索；
15. 评价三目标函数；
16. 更新 Pareto 档案；
17. 若解进入 Pareto 档案，则在最后一步给予终局奖励；
18. 使用 Replay Buffer 更新 DDQN；
19. 根据 Pareto 档案更新信息素；
20. 达到最大迭代次数后输出 Pareto 档案。

### 13.2 伪代码

```text
Algorithm RL-CG-MOACO

Input:
    Task set T
    Satellite set S
    Observation windows W
    Maximum iteration MaxIter
    Number of ants N_ant

Output:
    Pareto archive Archive

1.  Generate task-satellite-window candidate nodes V
2.  Construct conflict graph G=(V,E)
3.  Compute graph features C(v), R(v), eta(v), BlockPenalty(v)
4.  Initialize pheromone tau(v)=tau0
5.  Initialize Pareto archive Archive=empty
6.  Initialize MLP-DDQN network Q_theta and target network Q_theta-
7.  Initialize Replay Buffer

8.  Warm-up phase:
9.      for iter = 1 to 30 do
10.         for each ant do
11.             Construct solution using tau and eta only
12.             Record episode length T_e
13.             Store raw trajectory if needed
14.         end for
15.     end for
16.     Compute T_ref = round(mean(T_e))
17.     Convert raw construction scores c_t into scaled rewards c_t / T_ref

18. for iter = 31 to MaxIter do
19.     Population = empty

20.     for each ant do
21.         X = empty
22.         A = V
23.         EpisodeTransitions = empty

24.         while A is not empty do
25.             Build global feature g_t

26.             for each node v in A do
27.                 Extract node feature h(v)
28.                 Compute Q_theta(s_t,v)
29.             end for

30.             Compute Q_mean = mean_{u in A} Q_theta(s_t,u)

31.             for each node v in A do
32.                 Compute Adv_theta(s_t,v) = Q_theta(s_t,v) - Q_mean
33.                 Compute selection probability:
                       P(v|s_t) proportional to
                       tau(v)^alpha * eta(v)^beta * exp(kappa * Adv_theta(s_t,v))
34.             end for

35.             Select node v_t according to P(v|s_t)
36.             Add v_t into X
37.             Update feasible action set A = A - {v_t} - N(v_t)
38.             Compute construction score c_t
39.             Compute scaled immediate reward r_t = clip(c_t,-1,1) / T_ref
40.             Store transition temporarily
41.         end while

42.         Apply lightweight replacement-insertion local search
43.         Evaluate objective vector F(X)
44.         Update Pareto archive
45.         if X enters Archive then
46.             Add terminal reward r_final=+1 to the last transition
47.         else
48.             Add terminal reward r_final=0 to the last transition
49.         end if
50.         Store episode transitions into Replay Buffer
51.         Add X into Population
52.     end for

53.     Train DDQN using Replay Buffer
54.     Update target network periodically
55.     Evaporate pheromone
56.     Reinforce pheromone according to Pareto archive
57. end for

58. return Archive
```

---

## 14. 实验设计建议

### 14.1 对比算法

建议包括：

1. NSGA-II；
2. MOEA/D；
3. MOPSO；
4. 普通 MOACO；
5. CG-MOACO；
6. RL-MOACO；
7. RL-CG-MOACO。

### 14.2 消融实验

| 版本 | 目的 |
|---|---|
| MOACO | 基础多目标蚁群算法 |
| CG-MOACO | 验证冲突图和动作掩码作用 |
| RL-MOACO | 验证 DDQN 对普通 MOACO 的作用 |
| RL-CG-MOACO | 完整方法 |
| RL-CG-MOACO without BlockPenalty | 验证冲突屏蔽惩罚 |
| RL-CG-MOACO without final reward | 验证 Pareto 档案终局奖励 |
| RL-CG-MOACO without local search | 验证局部搜索贡献 |
| RL-CG-MOACO without DDQN | 验证强化学习模块贡献 |

### 14.3 评价指标

#### 多目标指标

- HV；
- IGD；
- Spread；
- Spacing；
- 非支配解数量。

#### 调度指标

- 任务完成率；
- 总收益；
- 姿态机动代价；
- 负载均衡指标；
- 可行解比例；
- CPU 时间。

#### 机制分析指标

- 每代 Pareto 档案大小；
- HV 收敛曲线；
- IGD 收敛曲线；
- 平均可选节点数量；
- 平均构造长度；
- Warm-up 估计得到的 $T_{ref}$；
- BlockPenalty 对节点选择的影响；
- DDQN 参与前后的节点选择分布变化。

---

## 15. 论文结构建议

### 15.1 Introduction

主要说明：

- 敏捷卫星调度的重要性；
- 多目标优化需求；
- 复杂约束导致可行解构造困难；
- 传统 MOACO 依赖固定启发式；
- 强化学习可以学习状态相关的构造策略；
- 本文提出 RL-CG-MOACO。

### 15.2 Related Work

建议分为：

1. Agile Earth Observation Satellite Scheduling；
2. Multi-objective Satellite Scheduling；
3. RL-enhanced Metaheuristics for Scheduling；
4. RL-enhanced ACO and Combinatorial Optimization。

### 15.3 Problem Formulation

包括：

- 任务集合；
- 卫星集合；
- 观测窗口；
- 决策变量；
- 目标函数；
- 约束条件；
- 冲突图建模。

### 15.4 Proposed Method

包括：

1. 三元冲突图构建；
2. 冲突图动作掩码；
3. MLP-DDQN 节点选择；
4. 信息素-启发式基础吸引力与 DDQN Advantage 校正；
5. Warm-up 估计 $T_{ref}$ 的混合奖励；
6. Pareto 档案更新；
7. 轻量局部搜索；
8. 算法伪代码。

### 15.5 Experiments

包括：

- 数据集；
- 对比算法；
- 参数设置；
- 多目标评价指标；
- 消融实验；
- 收敛性分析；
- Pareto 前沿可视化；
- 机制分析；
- 参数敏感性分析。

### 15.6 Conclusion

总结：

- 提出了任务-卫星-窗口冲突图；
- 设计了冲突图动作掩码；
- 将 MLP-DDQN 嵌入 MOACO 的逐节点构造过程；
- 设计了 Warm-up 估计 $T_{ref}$ 的混合奖励机制；
- 提升了多目标敏捷卫星调度的 Pareto 解集质量。

---

## 16. 最终一句话总结

本文最终核心内容可以概括为：

$$
\boxed{
\text{面向多目标敏捷卫星调度，构建任务-卫星-窗口冲突图，并利用 MLP-DDQN 在 MOACO 的逐节点构造过程中学习候选节点选择价值；同时通过冲突图动作掩码保证可行性，通过 Warm-up 估计 }T_{ref}\text{ 的缩放即时奖励和 Pareto 档案终局奖励训练 DDQN，最终引导蚂蚁构造高质量非支配调度方案。}
}
$$
