# Lab5: Value-based Reinforcement Learning (DQN)

## Context
NYCU Spring 2026 DLP Lab5，截止日 **2026/05/05 (Tue.) 23:59**。目標是實作 DQN 及其進階版本，分三個 Task 完成 CartPole 和 Pong 兩個環境。主要實作檔案為 `dqn.py`（助教提供模板，含多處 TODO），評估腳本 `test_model.py` 已完整可用。

---

## ⚠️ 重要警告：這些事情會扣分或直接 0 分

### 直接 0 分
- 5/21 23:59 後提交 → **直接 0 分，不接受**

### 嚴重扣分
| 狀況 | 扣分 |
|------|------|
| 沒有提交有效 demo 影片 | **model snapshots 全部不計分**（Task 1+2+3 的 50% 全沒） |
| 檔案命名或資料夾結構錯誤 | **-5 分** |
| 助教無法重現結果（沒有提供執行指令） | **-5 分** |
| 每少一張評估截圖 | **-3 分**（從 Report 60% 中扣） |
| 5/5 後 ~ 5/21 前提交 | 最終成績 **× 0.7**（打七折） |

### 程式碼限制
- **不可更動 `dqn.py` 中模組的結構**（DQN, PrioritizedReplayBuffer, AtariPreprocessor, DQNAgent 這四個 class 的架構不能改）

---

## 配分結構

| 項目 | 分數 |
|------|------|
| **Report（60%）** | |
| - Introduction | 5% |
| - Implementation 說明（Tasks 1-3 的概念與實作細節） | 20% |
| - Analysis & Discussion（訓練曲線 + 樣本效率分析 + ablation study） | 25% |
| - 額外訓練策略 bonus | 最多 +10% |
| **Model Performance（50%）** | |
| - Task 1 CartPole snapshot | 15%：`min(avg_score, 480) / 480 × 15%`，滿分需平均 > 480 |
| - Task 2 Pong snapshot | 20%：`(min(avg_score, 19) + 21) / 40 × 20%`，滿分需平均 > 19 |
| - Task 3 Enhanced Pong | 20%：依達到 score 19 的時間點（越早越高） |
| - Demo 影片 | 未提交則 snapshot 全不計分 |

**Task 3 分數對照表：**
| 達到 score 19 的步數 | 600k | 1M | 1.5M | 2M | 2.5M | >2.5M |
|---------------------|------|-----|------|-----|------|-------|
| 得分比例 | 20% | 17% | 15% | 13% | 11% | 8% |

---

## 提交格式（嚴格遵守，錯誤 -5 分）

### ZIP 檔名
```
LAB5_{StudentID}_{YourName}.zip
```

### ZIP 內部結構
```
LAB5_StudentID.zip
|-- LAB5_StudentID_Code/         ← 只放 source code
|   |-- dqn.py
|   |-- test_model.py
|   |-- requirements.txt
|   |-- (其他 .py 或 .sh)
|-- LAB5_StudentID.pdf           ← 報告（單一 PDF）
|-- LAB5_StudentID.mp4           ← Demo 影片（5-6 分鐘）
|-- LAB5_StudentID_task1.pt      ← Task 1 最佳模型
|-- LAB5_StudentID_task2.pt      ← Task 2 最佳模型
|-- LAB5_StudentID_task3_600000.pt
|-- LAB5_StudentID_task3_1000000.pt
|-- LAB5_StudentID_task3_1500000.pt
|-- LAB5_StudentID_task3_2000000.pt
|-- LAB5_StudentID_task3_2500000.pt
|-- LAB5_StudentID_task3_best.pt ← 任何步數達到 score 19 的快照
```

> **注意**：report、影片、模型檔案都放在 ZIP 根目錄，**不可額外包一層資料夾**

---

## 評估協定（必須做到）

1. 提供可直接執行的指令或 `.sh` 腳本給助教重現結果
   - 範例：`python test_model_task3.py --model_path LAB5_StudentID_task3_2500000.pt`
   - Evaluation seeds: **0 to 19**（20 個 seed）
2. 提交 `requirements.txt`
3. Report 中每個 Task 都要有**評估截圖**（顯示 20 seeds 的逐一結果和平均）

---

## 專案目標

| Task | 環境 | 關鍵技術 | 狀態 |
|------|------|---------|------|
| Task 1 | CartPole | 基礎 DQN + Replay Buffer | 待實作 |
| Task 2 | Pong (Atari) | CNN 架構 + frame stacking | 待實作 |
| Task 3 | Pong (Atari) | Double DQN / PER / Multi-step | 待實作 |

---

## 逐步完成計畫

### Step 1：Task 1 — CartPole DQN（`dqn.py`）

**缺少的部分（需實作）：**

1. **定義 Q-Network 架構**（`dqn.py` 約第 36-49 行）
   ```python
   self.network = nn.Sequential(
       nn.Linear(state_dim, 128), nn.ReLU(),
       nn.Linear(128, 128),       nn.ReLU(),
       nn.Linear(128, action_dim)
   )
   ```

2. **初始化 Replay Buffer**（`DQNAgent.__init__` 中）
   ```python
   self.memory = deque(maxlen=self.buffer_size)
   ```

3. **實作 `train()` 的 batch sampling**（約第 247-252 行）
   ```python
   batch = random.sample(self.memory, self.batch_size)
   states, actions, rewards, next_states, dones = zip(*batch)
   ```

4. **實作 Bellman loss 與反向傳播**（約第 263-268 行）
   ```python
   q_values = self.q_network(states).gather(1, actions)
   with torch.no_grad():
       next_q = self.target_network(next_states).max(1)[0]
       target = rewards + gamma * next_q * (1 - dones)
   loss = F.mse_loss(q_values.squeeze(), target)
   self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
   ```

**驗證**：CartPole 平均 reward 需達 **480+**（滿分門檻），wandb 有訓練曲線。

---

### Step 2：Task 2 — Atari Pong CNN DQN

1. **修改 Q-Network 為 CNN 架構**
   ```python
   # CNN for Atari: input (batch, 4, 84, 84)
   self.conv = nn.Sequential(
       nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
       nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
       nn.Conv2d(64, 64, 3, 1), nn.ReLU()
   )
   self.fc = nn.Sequential(
       nn.Linear(64*7*7, 512), nn.ReLU(),
       nn.Linear(512, action_dim)
   )
   ```

2. `AtariPreprocessor` 已實作（grayscale + resize 84x84 + frame stacking）

3. Replay buffer 記憶體注意：Atari 建議 50k~500k，依 GPU 記憶體調整

**訓練時間警告**：Pong 約需 **1M steps**，RTX3090 約跑 **20 小時**，請提早開始！

**驗證**：用 `test_model.py` 跑 20 seeds，平均 reward 需 > 19（滿分門檻）。

---

### Step 3：Task 3 — Enhanced DQN

#### 3a. Double DQN（必做）
```python
next_actions = self.q_network(next_states).argmax(1, keepdim=True)
next_q = self.target_network(next_states).gather(1, next_actions)
```

#### 3b. Prioritized Experience Replay（必做，骨架已在 `dqn.py`）
- 實作 `add(transition, priority)`：priority = |TD error| + ε
- 實作 `sample(batch_size)`：按 P(i) = p_i^α / Σp_k^α 抽樣，回傳 IS weights
- 實作 `update_priorities(indices, td_errors)`

#### 3c. Multi-step Returns（建議做，report 需說明）
```
R_t^(n) = Σ γ^k · r_{t+k} + γ^n · max_a Q(s_{t+n}, a)
```

**Task 3 特別注意**：
- Report 中**必須標記**模型首次達到 score 19 的 timestep
- 助教會**重新跑你的快照**驗證，如無法佐證則依重跑結果計分
- 需提交 `task3_best.pt`（任何步數達到 score 19 的那個快照）

**模型快照**：每隔里程碑存一份：600k / 1M / 1.5M / 2M / 2.5M steps

---

### Step 4：評估與提交準備

- [ ] Task 1-3 各跑 **20 seeds**（seed 0~19），截圖 average reward
- [ ] 確認 x 軸是 **environment steps**（不是 episodes）的訓練曲線
- [ ] 提供 ablation study（各技術分開比較）
- [ ] 錄製 **5-6 分鐘 MP4**，英文講解（需助教預先批准才能用中文）
- [ ] 撰寫 report（含 Bellman error 公式、Double DQN 改動、PER 實作、wandb 說明）
- [ ] 準備 `requirements.txt`
- [ ] 提供可重現的執行指令
- [ ] 按照嚴格格式打包 ZIP

---

## 關鍵檔案

| 檔案 | 狀態 | 說明 |
|------|------|------|
| `dqn.py` | ~60% 完成 | 主要實作檔，多處 TODO（助教模板，不可改結構） |
| `test_model.py` | 完整 | 評估腳本，直接可用（助教提供） |
| `Spring2026_DLP_RL_HW1.pdf` | 參考 | 完整規格說明 |

## 套件版本需求
- Python >= 3.8
- gymnasium 1.1.1
- ale-py >= 0.10.0
- opencv-python
- torch
- wandb
