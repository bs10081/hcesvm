# CE-SVM (Cost-Effective SVM) 數學模型

## 概述

CE-SVM (Cost-Effective Support Vector Machine) 是一個結合特徵選擇與準確率優化的二元分類模型。此模型基於標準 SVM，加入了：

1. **L1 範數正則化**（促進稀疏解）
2. **特徵選擇機制**（透過成本預算約束）
3. **多層準確率約束**（三層二元指標）
4. **類別平衡優化**（正負類準確率項）

**原始來源**：Parkinsons CE-SVM LINGO 模型 (`CEAS_SVM1_SL_Par.lg4`)

---

## 符號定義

### 資料集參數

| 符號 | 說明 | 維度 |
|------|------|------|
| $n$ | 訓練樣本總數 | - |
| $d$ | 特徵總數 | - |
| $\mathbf{x}_i \in \mathbb{R}^d$ | 第 $i$ 個樣本的特徵向量 | $n \times d$ |
| $y_i \in \{+1, -1\}$ | 第 $i$ 個樣本的類別標籤 | $n$ |
| $\text{cost}_j$ | 第 $j$ 個特徵的成本 | $d$ |
| $n_{+}$ | 正類 (+1) 樣本數 | - |
| $n_{-}$ | 負類 (-1) 樣本數 | - |

### 超參數

| 符號 | 說明 | LINGO 變數 | 典型值 |
|------|------|-----------|--------|
| $C$ | 鬆弛變數懲罰係數 | `C_hyper` | 1.0 |
| $B$ | 特徵選擇預算上限 | `budget` | 75 |
| $\epsilon$ | 準確率約束的小量 | `e` | 0.0001 |
| $M$ | Big-M 常數 | `M` | 1000 |

---

## 決策變數

### 主要變數

| 變數 | 維度 | 類型 | 說明 | LINGO 變數 |
|------|------|------|------|-----------|
| $w^+_j$ | $d$ | 連續 $\geq 0$ | 權重正部分 | `w_plus(j)` |
| $w^-_j$ | $d$ | 連續 $\geq 0$ | 權重負部分 | `w_minus(j)` |
| $b$ | 1 | 連續（自由） | 截距 | `b` |
| $\xi_i$ | $n$ | 連續 $\geq 0$ | 鬆弛變數 | `ksi(i)` |

**權重恢復**：
$$
w_j = w^+_j - w^-_j, \quad \forall j \in \{1, \ldots, d\}
$$

**L1 範數**：
$$
\|w\|_1 = \sum_{j=1}^{d} (w^+_j + w^-_j)
$$

### 特徵選擇變數

| 變數 | 維度 | 類型 | 說明 | LINGO 變數 |
|------|------|------|------|-----------|
| $v_j$ | $d$ | 二元 $\{0, 1\}$ | 特徵 $j$ 是否被選擇 | `v(j)` |

### 準確率相關變數（三層指標）

| 變數 | 維度 | 類型 | 說明 | LINGO 變數 |
|------|------|------|------|-----------|
| $\alpha_i$ | $n$ | 二元 $\{0, 1\}$ | 第一層指標（鬆弛變數啟用） | `ai(i)` |
| $\beta_i$ | $n$ | 二元 $\{0, 1\}$ | 第二層指標（準確率門檻） | `bi(i)` |
| $\rho_i$ | $n$ | 二元 $\{0, 1\}$ | 第三層指標（更嚴格門檻） | `ri(i)` |
| $l_+$ | 1 | 連續 $\in [0, 1]$ | 正類準確率下界 | `l_p` |
| $l_-$ | 1 | 連續 $\in [0, 1]$ | 負類準確率下界 | `l_n` |

**層次關係**：
$$
\alpha_i \geq \beta_i \geq \rho_i, \quad \forall i
$$

---

## 優化模型

### 目標函數

$$
\begin{aligned}
\min \quad & \|w\|_1 + C \cdot \sum_{i=1}^{n} (\alpha_i + \beta_i + \rho_i) - l_+ - l_- \\
= \quad & \sum_{j=1}^{d} (w^+_j + w^-_j) + C \cdot \sum_{i=1}^{n} (\alpha_i + \beta_i + \rho_i) - l_+ - l_-
\end{aligned}
$$

**目標項解釋：**
1. **$\|w\|_1$**：L1 正則化，促進稀疏解（特徵選擇）
2. **$C \cdot \sum (\alpha + \beta + \rho)$**：誤分類懲罰（透過三層指標間接控制）
3. **$-l_+ - l_-$**：最大化正負類準確率

---

## 約束條件

### 1. SVM 分離約束

對所有樣本 $i \in \{1, \ldots, n\}$：

$$
y_i \left( \sum_{j=1}^{d} (w^+_j - w^-_j) \cdot x_{ij} + b \right) \geq 1 - \xi_i
$$

**意義**：樣本 $i$ 必須在正確的超平面側，允許鬆弛 $\xi_i$。

### 2. Big-M 約束（三層）

對所有樣本 $i$：

$$
\begin{aligned}
\xi_i &\leq M \cdot \alpha_i \\
\xi_i &\leq 1 + M \cdot \beta_i \\
\xi_i &\leq 2 + M \cdot \rho_i
\end{aligned}
$$

**意義**：
- 若 $\alpha_i = 0$，則 $\xi_i = 0$（樣本完美分類）
- 若 $\beta_i = 0$，則 $\xi_i \leq 1$（樣本在邊界內）
- 若 $\rho_i = 0$，則 $\xi_i \leq 2$（樣本在較寬鬆範圍內）

### 3. 層次指標約束

$$
\alpha_i \geq \beta_i \geq \rho_i, \quad \forall i
$$

**意義**：指標遞減，形成三層結構。

### 4. 準確率下界約束

$$
\xi_i \geq (1 + \epsilon) \cdot (1 - (1 - \beta_i)), \quad \forall i
$$

簡化為：
$$
\xi_i \geq (1 + \epsilon) \cdot \beta_i
$$

**意義**：若 $\beta_i = 1$（誤分類），則 $\xi_i$ 必須至少為 $1 + \epsilon$。

### 5. 正類準確率約束

$$
\sum_{i: y_i = +1} (1 - \beta_i) \geq l_+ \cdot n_+
$$

**意義**：正確分類的正類樣本數 $\geq l_+ \times n_+$（準確率下界）。

### 6. 負類準確率約束

$$
\sum_{i: y_i = -1} (1 - \beta_i) \geq l_- \cdot n_-
$$

**意義**：正確分類的負類樣本數 $\geq l_- \times n_-$（準確率下界）。

### 7. 特徵選擇預算約束

$$
\sum_{j=1}^{d} \text{cost}_j \cdot v_j \leq B
$$

**意義**：選擇的特徵總成本不超過預算 $B$。

### 8. 特徵啟用約束

對所有特徵 $j$：

$$
\begin{aligned}
w^+_j + w^-_j &\leq 1000 \cdot v_j \\
w^+_j + w^-_j &\geq 0.0000001 \cdot v_j
\end{aligned}
$$

**意義**：
- 若 $v_j = 0$（特徵未選擇），則 $w^+_j = w^-_j = 0$（權重為零）
- 若 $v_j = 1$（特徵被選擇），則權重可以非零

### 9. 變數邊界約束

$$
\begin{aligned}
w^+_j, w^-_j &\geq 0, \quad \forall j \\
\xi_i &\geq 0, \quad \forall i \\
\alpha_i, \beta_i, \rho_i &\in \{0, 1\}, \quad \forall i \\
v_j &\in \{0, 1\}, \quad \forall j \\
b &\in \mathbb{R} \text{ (free)}
\end{aligned}
$$

---

## LINGO 程式碼對應

### Sets 定義

```lingo
Sets:
    set1/1..n/: y, ksi, predict, ai, bi, ri;        ! 樣本集合
    set2/1..d/: w, w_plus, w_minus, cost, v;       ! 特徵集合
    set3(set1, set2): x;                            ! 特徵矩陣
Endsets
```

### Data 載入

```lingo
Data:
    y, x, cost = @ole('path/to/excel.xlsx');
    C_hyper = 1;
    budget = 75;
    e = 0.0001;
    M = 1000;
Enddata
```

### 目標函數

```lingo
Min = obj;
obj = @sum(set2(j): w_plus(j) + w_minus(j))
    + C_hyper * @sum(set1(i): ai(i) + bi(i) + ri(i))
    - l_p - l_n;
```

### 主要約束

```lingo
! SVM 分離約束
@for(set1(i):
    y(i) * (@sum(set2(j): (w_plus(j) - w_minus(j)) * x(i,j)) + b) >= 1 - ksi(i);
);

! Big-M 和層次約束
@for(set1(i):
    ksi(i) <= M * ai(i);
    ksi(i) <= 1 + M * bi(i);
    ksi(i) <= 2 + M * ri(i);
    ai(i) >= bi(i);
    bi(i) >= ri(i);
    ksi(i) >= (1 + e) * (1 - (1 - bi(i)));
);

! 準確率約束
@sum(set1(i): (1 - bi(i)) * (1 - y(i))) >= l_n * @sum(set1(i): 1 - y(i));
@sum(set1(i): (1 - bi(i)) * (1 + y(i))) >= l_p * @sum(set1(i): 1 + y(i));

! 特徵選擇約束
lc = @sum(set2(j): cost(j) * v(j));
lc <= budget;

@for(set2(j):
    w_plus(j) + w_minus(j) <= 1000 * v(j);
    w_plus(j) + w_minus(j) >= 0.0000001 * v(j);
    w_plus(j) >= 0;
    w_minus(j) >= 0;
    @bin(v(j));
);

! 指標變數
@for(set1(i):
    ksi(i) >= 0;
    @bin(ai(i));
    @bin(bi(i));
    @bin(ri(i));
);

@free(b);
```

---

## 與標準 SVM 的差異

| 面向 | 標準 SVM | CE-SVM |
|------|---------|--------|
| **正則化** | L2 範數 ($\frac{1}{2}\|w\|^2$) | L1 範數 ($\|w\|_1$) |
| **特徵選擇** | 無 | 有（成本預算約束） |
| **準確率優化** | 無 | 有（正負類準確率下界） |
| **鬆弛懲罰** | 線性 ($C \sum \xi$) | 透過三層指標 ($C \sum (\alpha + \beta + \rho)$) |
| **目標方向** | 最小化誤差 | 最小化誤差 **+ 最大化準確率** |

---

## Gurobi 實作注意事項

1. **L1 範數線性化**：使用 $w = w^+ - w^-$ 分解
2. **Big-M 值選擇**：$M = 1000$ 在 LINGO 中有效，Gurobi 中可能需調整
3. **特徵啟用約束**：上界 $1000$ 和下界 $0.0000001$ 可以調整為適應不同尺度的資料
4. **準確率變數**：$l_+, l_-$ 是連續變數，需要在目標函數中最大化（負號）
5. **三層指標**：$\alpha \geq \beta \geq \rho$ 可能導致模型複雜度增加，需要良好的預解決 (presolve)

---

## 模型特性

### 優點

1. **稀疏解**：L1 範數促進權重稀疏，自動特徵選擇
2. **成本意識**：透過預算約束控制特徵選擇成本
3. **類別平衡**：正負類準確率項改善不平衡資料的表現
4. **多層控制**：三層指標提供細粒度的誤分類控制

### 缺點

1. **模型複雜度**：相比標準 SVM，變數數量增加（$3n + 3d + 2$）
2. **求解時間**：MILP 問題，可能需要較長求解時間
3. **超參數敏感**：$C$, $B$, $M$ 等參數需要仔細調整

---

## 應用場景

1. **醫療診斷**：如 Parkinson's 病預測，特徵獲取有成本
2. **不平衡分類**：需要同時優化正負類準確率
3. **高維資料**：需要特徵選擇以避免過擬合
4. **成本敏感學習**：不同特徵有不同的獲取成本

---

## 參考文獻

- 原始 LINGO 模型：`CEAS_SVM1_SL_Par.lg4`
- Parkinsons CE-SVM 資料集：`Parkinsons_CESVM.xlsx`
- 作者：NCNU (國立暨南國際大學) 資訊管理學系
