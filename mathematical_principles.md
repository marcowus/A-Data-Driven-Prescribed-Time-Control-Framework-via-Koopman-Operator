# 基于Koopman算子与预定时间自适应反步法的范德波振荡器控制

本文档详细阐述了针对二阶范德波（Van der Pol）振荡器系统的控制原理。项目结合了数据驱动的Koopman算子建模与非线性预定时间自适应反步控制（Prescribed-Time Adaptive Backstepping Control），旨在实现系统状态在预定时间 $T_p$ 内的收敛，并对未知参数进行自适应估计。

## 1. 系统模型 (System Model)

被控对象为经典的二阶范德波振荡器，其动力学方程如下：

$$
\begin{cases}
\dot{x}_1 = x_2 \\
\dot{x}_2 = -x_1 + \theta^* (1 - x_1^2) x_2 + u + d(t)
\end{cases}
$$

其中：
- $x = [x_1, x_2]^T \in \mathbb{R}^2$ 为系统状态（位置与速度）。
- $u \in \mathbb{R}$ 为控制输入。
- $\theta^*$ 为未知的系统参数（真实值）。
- $d(t)$ 为外部扰动，假设为 $d(t) = 0.1 \sin(\pi t)$。
- 非线性项定义为 $\phi(x) = (1 - x_1^2) x_2$。

## 2. Koopman算子数据驱动建模 (Data-Driven Modeling via Koopman Operator)

为了辅助控制或系统分析，我们引入Koopman算子理论将非线性系统提升至高维线性空间。

### 2.1 提升函数 (Lifting Functions)
定义观测函数（提升函数）向量 $\psi(x): \mathbb{R}^2 \to \mathbb{R}^{N_{obs}}$，将低维非线性状态映射到高维特征空间。代码中采用的基函数包括多项式及三角函数：

$$
\psi(x) = [x_1, x_2, x_1^2, x_2^2, x_1 x_2, \sin(x_1), \cos(x_1), \dots]^T
$$

### 2.2 扩展动态模式分解 (EDMD)
Koopman算子 $\mathcal{K}$ 描述了观测函数随时间的演化。对于离散时间系统，我们寻找一个有限维近似矩阵 $A_{aug}$，使得：

$$
\psi(x_{k+1}) \approx A_{aug} \begin{bmatrix} \psi(x_k) \\ u_k \end{bmatrix}
$$

通过收集轨迹数据 $\{ (x_k, u_k, x_{k+1}) \}_{k=1}^M$，构建数据矩阵：
$$
\Psi_X = [\psi(x_1), \dots, \psi(x_M)]^T, \quad
\Psi_Y = [\psi(x_2), \dots, \psi(x_{M+1})]^T, \quad
U = [u_1, \dots, u_M]^T
$$
令 $\Phi = [\Psi_X, U]$，通过吉洪诺夫正则化（Ridge Regression）求解最优近似矩阵：

$$
A_{aug}^T = (\Phi^T \Phi + \lambda I)^{-1} \Phi^T \Psi_Y
$$

该模型可用于预测系统状态导数或未来轨迹。

## 3. 预定时间自适应反步控制 (Prescribed-Time Adaptive Backstepping Control)

为了确保系统状态在用户设定的时间 $T_p$ 内收敛到平衡点，我们设计了基于时变增益的反步控制器。

### 3.1 预定时间缩放函数 (Prescribed-Time Scaling Function)
引入时变增益函数 $\mu(t)$，当 $t \to T_p$ 时，$\mu(t) \to \infty$，迫使误差在 $T_p$ 前收敛至零。定义如下：

$$
\mu(t) = \frac{1}{T_p - t}, \quad t \in [0, T_p)
$$

### 3.2 控制器设计 (Controller Design)

采用反步法（Backstepping）设计控制律。

**第一步 (Step 1):**
定义位置误差 $z_1 = x_1$。选取虚拟控制量 $\alpha_1$ 以镇定 $z_1$：

$$
\alpha_1 = -\sigma_1 \mu(t) z_1 = -\frac{\sigma_1}{T_p - t} z_1
$$

其中 $\sigma_1 > 0$ 为设计参数。

**第二步 (Step 2):**
定义速度误差 $z_2 = x_2 - \alpha_1$。
对 $z_2$ 求导：
$$
\dot{z}_2 = \dot{x}_2 - \dot{\alpha}_1 = -x_1 + \theta^* \phi(x) + u + d(t) - \dot{\alpha}_1
$$

为了消除未知参数 $\theta^*$ 的影响，引入估计值 $\hat{\theta}$。设计控制律 $u$ 如下：

$$
u = \underbrace{- z_1}_{\text{抵消耦合项}} \underbrace{- \sigma_2 \mu(t) z_2}_{\text{预定时间镇定}} + \underbrace{x_1 - \hat{\theta} \phi(x) + \dot{\alpha}_1}_{\text{模型补偿}}
$$

代入代码中的具体实现：
$$
u = -z_1 - \frac{\sigma_2}{T_p - t} z_2 + x_1 - \hat{\theta} (1-x_1^2)x_2 + \dot{\alpha}_{1,approx}
$$

> 注意：$\dot{\alpha}_1$ 包含对时间的偏导项，随着 $t \to T_p$ 快速增长，提供极强的控制作用以压制误差。

### 3.3 自适应律 (Adaptive Law)
为了估计未知参数 $\theta^*$，采用基于Lyapunov设计的自适应律：

$$
\dot{\hat{\theta}} = \gamma z_2 \phi(x) = \gamma z_2 (1-x_1^2)x_2
$$

其中 $\gamma > 0$ 为学习率。

## 4. 稳定性分析与切换策略 (Stability & Switching)

### 4.1 稳定性原理
构造Lyapunov函数 $V = \frac{1}{2}z_1^2 + \frac{1}{2}z_2^2 + \frac{1}{2\gamma}(\hat{\theta} - \theta^*)^2$。
通过上述控制律的设计，可以证明在 $t < T_p$ 期间，跟踪误差 $z_1, z_2$ 的收敛速度受 $\mu(t)$ 主导，从而实现预定时间收敛。

### 4.2 切换机制 (Switching Mechanism)
由于增益函数 $\mu(t)$ 在 $t = T_p$ 处奇异（趋于无穷大），实际工程中需要在接近 $T_p$ 时切换控制策略。
代码中引入切换因子 $\alpha_{switch} = 0.95$。
- **当 $t < 0.95 T_p$ 时**：启用预定时间自适应控制器。
- **当 $t \ge 0.95 T_p$ 时**：切换至线性控制器（LQR/PD控制）维持系统稳定：
  $$
  u_{linear} = -K x = -[k_1 x_1 + k_2 x_2]
  $$

## 5. 结论

该工作结合了非线性控制理论与数据驱动方法：
1.  **高精度控制**：通过预定时间控制实现了状态在 $T_p=5s$ 内的快速收敛。
2.  **鲁棒性**：自适应律有效处理了参数不确定性。
3.  **约束满足**：尽管未显式使用对数障碍函数，高增益特性使得系统状态保持在安全范围内（如图所示的 IBLF 效果）。
