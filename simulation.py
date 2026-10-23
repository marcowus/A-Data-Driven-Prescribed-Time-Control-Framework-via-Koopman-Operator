import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 离心压缩机数字孪生参数与动力学引擎
# ==========================================
class CentrifugalCompressorTwin:
    def __init__(self, dt=0.001, total_time=15.0):
        self.dt = dt
        self.t_span = np.arange(0, total_time, dt)
        self.N_steps = len(self.t_span)

        # 机械参数: 额定转速 3000 RPM -> f_r = 50 Hz
        self.f_r = 50.0
        self.omega_r = 2 * np.pi * self.f_r
        self.Z_gear = 18  # 齿轮齿数 (啮合频率 f_m = 900 Hz)
        self.omega_m = self.Z_gear * self.omega_r

        # 轴承转子动力学固有参数
        self.omega_n = 2 * np.pi * 52.0  # 一阶临界转速
        self.zeta = 0.05                 # 机械阻尼比
        self.beta_nl = 1.5e6             # Duffing 非线性刚度

        # Moore-Greitzer 压缩气动模型参数
        self.l_c = 4.0                   # 压缩系统有效管长
        self.B = 0.65                    # Greitzer B参数 (决定喘振失稳裕度)
        self.H = 0.18                    # 压升特性半高
        self.W = 0.25                    # 流量特性半宽
        self.Psi_c0 = 0.30               # 零流量压升截距
        self.sigma_stall = 3.5           # 失速涡发展速率
        self.gamma_T0 = 0.65             # 节流阀基准开度系数

        # 湍流混合噪声强度 (保障健康态的强混合性)
        self.noise_aero = 0.03
        self.noise_mech = 0.08

    def _psi_c(self, phi):
        """压缩机静压升特性曲线 (立方多项式)"""
        val = (phi / self.W) - 1.0
        return self.Psi_c0 + self.H * (1.0 + 1.5 * val - 0.5 * (val ** 3))

    def simulate(self, fault_mode='NORMAL'):
        """
        SDE 求解器: 仿真五类动力学工况
        fault_mode: 'NORMAL', 'MISALIGNMENT', 'STICTION', 'STALL', 'GEAR'
        """
        # 初始化状态
        phi = 0.40   # 初始无量纲流量 (处于稳定右翼区)
        psi = self._psi_c(phi) # 初始压升
        R = 0.0      # 失速涡胞振幅平方
        theta_v = 1.0 # 节流阀相对开度
        x, vx = 0.0, 0.0 # 转子径向振动位移与速度

        # 存储观测信号: 监测轴承径向振动传感器与排气压力脉动
        vib_sensor = np.zeros(self.N_steps)
        press_sensor = np.zeros(self.N_steps)

        for k in range(self.N_steps):
            t = self.t_span[k]

            # --- 工况开关与外激动力学分支 ---
            F_ext = 0.0
            stiction_offset = 0.0

            if fault_mode == 'NORMAL':
                # 健康工况: 唯有湍流耗散与微观噪声
                phi_bias = 0.0

            elif fault_mode == 'MISALIGNMENT':
                # 故障一: 转子动平衡/不对中 (出现 1x, 2x, 3x 刚性等差锁相谐波)
                F_ext = (0.25 * np.cos(self.omega_r * t) +
                         0.20 * np.cos(2 * self.omega_r * t + np.pi/4) +
                         0.15 * np.cos(3 * self.omega_r * t + np.pi/3))
                phi_bias = 0.0

            elif fault_mode == 'STICTION':
                # 故障二: 节流阀/防喘阀卡涩 (死区与积分滞后诱发约 2.5 Hz 的低频极限环)
                stiction_offset = 0.12 * np.sin(2 * np.pi * 2.5 * t)
                phi_bias = 0.0

            elif fault_mode == 'STALL':
                # 故障三: 旋转失速/喘振前兆 (进气节流降低, 流量推向失稳区, 涡胞自发形成)
                phi_bias = -0.18 # 压入特性曲线峰值左侧不稳区

            elif fault_mode == 'GEAR':
                # 故障四: 齿轮早期点蚀 (啮合高频受到轴频的双侧调频/调幅多波混频)
                AM = 1.0 + 0.35 * np.cos(self.omega_r * t)
                FM = 0.25 * np.sin(self.omega_r * t)
                F_ext = 0.12 * AM * np.cos(self.omega_m * t + FM)
                phi_bias = 0.0

            # --- 1. 气动流场动力学积分 (Moore-Greitzer) ---
            effective_phi = phi + phi_bias
            throttle_v = max(0.1, theta_v + stiction_offset)
            Phi_T = self.gamma_T0 * throttle_v * np.sqrt(max(0.01, psi))

            dphi = (1.0 / self.l_c) * (self._psi_c(effective_phi) - psi - 0.75 * R * effective_phi)
            dpsi = (1.0 / (4.0 * (self.B**2) * self.l_c)) * (effective_phi - Phi_T)
            dR   = self.sigma_stall * R * (1.0 - effective_phi**2 - (1.0/6.0)*R)

            # Euler-Maruyama 随机注入
            phi += dphi * self.dt + self.noise_aero * np.sqrt(self.dt) * np.random.randn()
            psi += dpsi * self.dt + self.noise_aero * np.sqrt(self.dt) * np.random.randn()
            R   += dR * self.dt + 0.005 * np.sqrt(self.dt) * np.random.randn()
            R    = max(0.0, R) # 振幅平方非负

            # --- 2. 机械转子动力学积分 ---
            # 流固耦合力: 失速气动力与流体脉动施加在轴系上
            F_aero = 0.15 * np.sqrt(R) * np.cos(0.43 * self.omega_r * t) # 旋转失速频率约为 0.43x 转频

            ax = (- 2.0 * self.zeta * self.omega_n * vx
                  - (self.omega_n**2) * x
                  - self.beta_nl * (x**3)
                  + F_aero + F_ext)

            vx += ax * self.dt + self.noise_mech * np.sqrt(self.dt) * np.random.randn()
            x  += vx * self.dt

            # 记录传感器值
            vib_sensor[k] = x
            press_sensor[k] = psi

        return vib_sensor, press_sensor

# ==========================================
# 2. 多重遍历平均 (A_N^(n)) 算法核心实现
# ==========================================
def compute_multiple_ergodic_average(signal, n_slots=3, lag_step=2, max_N=3000):
    """
    计算基于等差采样的多阶遍历残差 A_N^(n)
    A_N(x) = (1/N) * sum_{k=1}^N prod_{j=1}^n f(T^{j*k} x)

    signal: 标量时序数据 (单条物理轨迹)
    n_slots: 阶数 n (如 2: 二阶自相关, 3: 三重等差干涉, 4: 四重等差干涉)
    lag_step: 基础步长比例
    max_N: 累加时间平均窗口长度 N
    """
    # 步骤 1: 严格去均值与尺度归一化 (确保 E_mu[f] = 0)
    f = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)

    # 确保信号长度足够提取 n*k 的等差步长
    req_len = n_slots * max_N * lag_step + 10
    if len(signal) < req_len:
        max_N = (len(signal) - 10) // (n_slots * lag_step)

    # 步骤 2: 构建等差乘积流 prod_{j=1}^n f(t + j*k*step)
    k_indices = np.arange(1, max_N + 1)
    product_series = np.ones(max_N)

    for j in range(1, n_slots + 1):
        delays = j * k_indices * lag_step
        product_series *= f[delays]

    # 步骤 3: 累积时间平均历程 A_N (对应 N 从 1 到 max_N 的收敛曲线)
    A_N_curve = np.cumsum(product_series) / k_indices
    return k_indices, A_N_curve

# ==========================================
# 3. 运行多工况数字孪生仿真并验证收敛性
# ==========================================
print("Running centrifugal compressor digital twin full-condition SDE simulation...")
twin = CentrifugalCompressorTwin(dt=0.0005, total_time=16.0)

scenarios = {
    'NORMAL': ('Normal Condition (Strong Mixing)', 'vib'),
    'MISALIGNMENT': ('Condition 1: Rotor Misalignment/Unbalance (Kronecker factor)', 'vib'),
    'STICTION': ('Condition 2: Anti-surge Valve Stiction (Limit cycle)', 'press'),
    'STALL': ('Condition 3: Rotating Stall Precursor (Quasi-periodic 2-torus)', 'press'),
    'GEAR': ('Condition 4: Early Gear Micro-pitting (High-order phase coupling)', 'vib')
}

sim_results = {}
for key, (desc, sensor_type) in scenarios.items():
    vib, press = twin.simulate(fault_mode=key)
    sig = vib if sensor_type == 'vib' else press
    sim_results[key] = sig

# ==========================================
# 4. 可视化分析: 验证定理 1.1 的极限分化行为
# ==========================================
# Use a non-interactive backend for saving the figure
import matplotlib
matplotlib.use('Agg')
# Set a font that supports Chinese characters, assuming typical Linux environment
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(5, 2, figsize=(15, 14))
plt.subplots_adjust(hspace=0.45, wspace=0.25)

for idx, (key, (title, sensor_type)) in enumerate(scenarios.items()):
    sig = sim_results[key]

    # 时序波形截取 (展示局部 1.0 秒信号)
    t_plot = twin.t_span[:2000]
    axes[idx, 0].plot(t_plot, sig[:2000], color='navy', lw=1.2)
    axes[idx, 0].set_title(f"Waveform: {title}", fontsize=11, fontweight='bold')
    axes[idx, 0].set_xlabel("Time t (s)")
    axes[idx, 0].set_ylabel("Sensor Output")
    axes[idx, 0].grid(True, alpha=0.3)

    # 计算并绘制 n=2 与 n=3 阶多重遍历残差随 N 的演变
    k_ax, A_N_2 = compute_multiple_ergodic_average(sig, n_slots=2, lag_step=2, max_N=2500)
    _,    A_N_3 = compute_multiple_ergodic_average(sig, n_slots=3, lag_step=2, max_N=2500)

    axes[idx, 1].plot(k_ax, A_N_2, label='2nd-order Ergodic Mean $A_N^{(2)}$', color='forestgreen', lw=1.5)
    axes[idx, 1].plot(k_ax, A_N_3, label='3rd-order Ergodic Mean $A_N^{(3)}$', color='crimson', lw=1.5)
    axes[idx, 1].axhline(0.0, color='black', linestyle='--', alpha=0.7, lw=1.0)

    axes[idx, 1].set_title(f"Ergodic Residual Convergence $A_N^{{(n)}}$ ({key})", fontsize=11, fontweight='bold')
    axes[idx, 1].set_xlabel("Average Sample Length N")
    axes[idx, 1].set_ylabel("Empirical Value")
    # axes[idx, 1].set_ylim(-0.35, 0.35) # Let it auto-scale to see full divergence
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('ergodic_simulation.png', dpi=300)
print("Simulation completed. Plot saved to 'ergodic_simulation.png'")
