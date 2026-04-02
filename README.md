# 2D 连续扫地机器人验证方案（MoE-PPO）

## 目标
做一个 2D 连续平面环境，用于验证扫地 / 避障 / 回充三任务上的 PPO / MoE-PPO。  
环境支持连续位置、速度控制、局部可见、遮挡感知。 :contentReference[oaicite:0]{index=0}

## 技术栈
- Python 3.10
- PyTorch  
- Gymnasium：环境接口
- Pymunk：2D 物理与碰撞
- Ray casting：局部视野 / 遮挡 / 简单激光雷达
- Stable-Baselines3：先跑 PPO baseline
- 自写 MoE policy：后续替换 SB3 默认策略网络 :contentReference[oaicite:1]{index=1}

## 环境设计

接口与reward仿照gym

### 状态
- 机器人：`x, y, theta, v, omega, battery`
- 局部观测：多条射线距离、可见污渍、可见障碍、充电桩相对方向
- 地图：连续平面；清洁覆盖可单独用细网格 mask 记录

### 动作
- 建议：`[v_cmd, omega_cmd]`
- 也可先简化为 `ax, ay`

### 奖励
- 清扫到新区域：正奖励
- 碰撞：负奖励
- 每步时间成本：小负奖励
- 成功回充：正奖励
- 电量耗尽：大负奖励
- 全区域清洁完成：终局大奖励

## 实验顺序
1. PPO + 连续平面 + 全局观测
2. PPO + 局部视野 + 遮挡
3. PPO + 动态障碍
4. MoE-PPO 替换普通 PPO
5. 比较成功率、碰撞率、清洁覆盖率、回充成功率、专家使用分布