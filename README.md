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

---

## Development Note / 開發說明

Built with the help of Claude (Anthropic) as a learning and architecture aid.

本專案在 Claude（Anthropic）的協助下開發，作為學習與架構上的輔助。

---

## License / 授權

MIT