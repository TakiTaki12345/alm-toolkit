# alm-toolkit

A modular Python toolkit for asset-liability management (ALM) and fixed-income analytics: yield curve construction, bond pricing, duration/convexity risk measures, and liability-driven immunization.

一個模組化的 Python 資產負債管理（ALM）與固定收益分析工具組：殖利率曲線建構、債券定價、存續期間／凸度風險指標，以及負債驅動的免疫化。

> **Status / 現狀:** Work in progress. The toolkit is being built incrementally, one verified module at a time, starting from a clean, application-agnostic core.
>
> 持續開發中。本工具組以漸進方式建構，從乾淨、與應用無關的核心出發，每次完成並驗證一個模組。

---

## Architecture / 架構

The codebase follows a **src layout** with a strict one-way dependency rule: `applications/` depends on `core/`, but `core/` never depends on `applications/`. The core provides application-agnostic primitives (curves, cash flows, pricing, risk); applications (such as immunization) are built on top.

程式碼採用 **src layout**，並遵守嚴格的單向依賴原則：`applications/` 依賴 `core/`，但 `core/` 永不依賴 `applications/`。核心提供與應用無關的基礎元件（曲線、現金流、定價、風險），應用層（如免疫化）則建構於其上。


---

## Installation / 安裝

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running tests / 執行測試

```bash
pytest
```

---

## Design Highlights / 設計亮點

A record of the engineering decisions behind this toolkit — the reasoning, not just the result.

本工具組背後工程決策的紀錄——重點在推理過程，而非只是結果。

### 1. Continuous compounding for clean derivatives / 以連續複利換取乾淨的微分

Discount factors use continuous compounding, `DF(t) = exp(-r·t)`, rather than discrete compounding `1/(1+r)^t`. The payoff is mathematical: the derivative of `exp(-r·t)` stays clean, which makes the duration and convexity formulas (built later) far simpler than the discrete equivalent.

折現因子採用連續複利 `DF(t) = exp(-r·t)`，而非離散複利 `1/(1+r)^t`。回報在於數學上的簡潔：`exp(-r·t)` 的微分形式乾淨，使得之後建構的存續期間與凸度公式遠比離散版本簡單。

### 2. Encapsulated interpolation for future extensibility / 封裝插值以利日後擴充

Interpolation and extrapolation live behind the `zero_rate` method. The MVP uses linear interpolation with flat extrapolation (a conservative market convention); upgrading to spline interpolation later requires changing only this one method, because pricing and risk code talk to the curve solely through `zero_rate` and `discount_factor`.

插值與外推被封裝在 `zero_rate` 方法之後。MVP 使用線性插值搭配平坦外推（保守的市場慣例）；日後升級為樣條插值只需修改這一個方法，因為定價與風險程式碼僅透過 `zero_rate` 與 `discount_factor` 與曲線溝通。

### 3. "Forgiving on input, strict on errors" API design / 「對輸入寬容、對錯誤嚴格」的 API 設計

`CashFlow` silently sorts unordered (time, amount) pairs because out-of-order input is a matter of convenience, not a mistake. `YieldCurve`, by contrast, *rejects* non-increasing tenors, because disordered tenors signal a genuine data error and would silently corrupt interpolation. The same library is forgiving where disorder is harmless and strict where it is dangerous.

`CashFlow` 會默默排序亂序的（時間, 金額）配對，因為順序隨意只是方便性問題、而非錯誤。相對地，`YieldCurve` 會*拒絕*非遞增的期限，因為亂序的期限代表真正的資料錯誤、且會悄悄破壞插值。同一套程式庫，在亂序無害處寬容、在亂序危險處嚴格。

### 4. High cohesion via object orientation / 以物件導向達成高內聚

`CashFlow` binds data (times, amounts) and operations (present value, and later IRR, duration) into one object. Internally it stores parallel NumPy arrays for vectorized computation, while exposing a clean object interface — combining the performance of arrays with the ergonomics of objects.

`CashFlow` 將資料（時間、金額）與操作（現值，以及之後的 IRR、存續期間）綁進單一物件。內部以平行的 NumPy 陣列儲存以利向量化運算，對外則提供乾淨的物件介面——兼得陣列的效能與物件的易用性。

### 5. Explicit numerical tolerances over exact equality / 以明確的數值容差取代精確相等

Tests assert results within explicit tolerances rather than exact floating-point equality, because IEEE 754 arithmetic makes exact equality both unattainable and the wrong target (`0.1 + 0.2 != 0.3`). Correctness in financial computation is established by bounding error within a justified tolerance — which is itself a core task of model validation.

測試以明確的容差驗證結果，而非精確的浮點相等，因為 IEEE 754 運算使得精確相等既不可達成、也是錯誤的目標（`0.1 + 0.2 != 0.3`）。金融計算的正確性，是透過將誤差限制在有依據的容差內來建立的——而這本身正是模型驗證的核心工作之一。

### 6. Cross-validating analytic formulas with numerical methods / 以數值方法交叉驗證解析公式

Duration and convexity are implemented as closed-form analytic formulas, then independently verified against numerical differentiation of the present-value function (central differences). If the analytic formula were wrong, the numerical check would catch it. Validating a model against an independent method built on different principles is a core technique of model validation.

存續期間與凸度以解析公式（封閉形式）實作，再以現值函數的數值微分（中央差分）獨立驗證。若解析公式有誤，數值檢驗會抓出來。用一個基於不同原理的獨立方法來驗證模型，正是模型驗證的核心技術。

### 7. Separation of concerns: data, risk, and application / 關注點分離：資料、風險與應用

The codebase keeps three concerns apart. `CashFlow` represents data — a stream of dated payments. `risk.py` provides analysis — duration and convexity as functions of a yield, kept separate because they depend on an external parameter (the yield level chosen by the analyst) rather than being intrinsic to the cash flow. The immunization application sits on top, consuming both. A lightweight `Bond` wrapper carries a cash flow and a yield, exposing PV, duration, and convexity, so the immunization solver can be written in the language of bonds rather than raw numbers — and its core stays a clean 2×2 linear system.

程式碼將三種關注點分離。`CashFlow` 代表資料——一串帶日期的給付。`risk.py` 提供分析——以殖利率為自變數的存續期間與凸度，之所以獨立，是因為它們依賴一個外部參數（分析者選定的利率水準），而非現金流自帶的屬性。免疫化應用層位於其上，同時取用兩者。一個輕量的 `Bond` 包裝持有現金流與殖利率，對外提供現值、存續期間與凸度，使免疫求解器能以「債券」而非裸數字的語言來書寫——其核心因而維持為乾淨的 2×2 線性系統。

### 8. Immunization as solve-then-verify, not just solve / 免疫化是「求解後驗證」，而非只求解

The two-bond immunizer does not stop at solving the 2×2 system for the portfolio weights. It also reports whether the solution is *feasible* (both weights non-negative — equivalent to the liability's duration lying between the two bonds') and whether the *convexity condition* holds (asset convexity ≥ liability convexity, per Redington). A solver that silently returns a portfolio requiring short positions, or one that fails the convexity test, would be worse than useless in practice; the result object surfaces these diagnostics explicitly.

兩債券免疫器不止於解出 2×2 系統的組合權重。它還會回報解是否*可行*（兩權重皆非負——等價於負債存續期間落在兩債券之間），以及*凸度條件*是否成立（依 Redington 理論，資產凸度 ≥ 負債凸度）。一個默默回傳「需要放空」組合、或未通過凸度檢驗的求解器，在實務上比沒用更糟；結果物件將這些診斷明確呈現出來。

---

## Development Note / 開發說明

Built with the help of Claude (Anthropic) as a learning and architecture aid.

本專案在 Claude（Anthropic）的協助下開發，作為學習與架構上的輔助。

---

## License / 授權

MIT