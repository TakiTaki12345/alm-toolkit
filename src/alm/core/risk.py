# Interest rate risk measures: duration and convexity.
from __future__ import annotations

import numpy as np
from alm.core.cashflow import CashFlow

"""
為什麼這次用獨立函數，而不是像 present_value 那樣做成 CashFlow 的方法？ 
這是個刻意的設計取捨，present_value 放進 CashFlow 是因為「一串現金流的現值」是現金流最核心、最內在的性質。
但 duration/convexity 是風險分析的視角，而且它們吃一個外部參數 y（用哪個利率水準算敏感度，是分析者的選擇，不是現金流自帶的屬性）。
把它們放在獨立的 risk.py 模組，能讓「資料表達（cashflow）」和「風險分析（risk）」這兩種關注點分開——這是關注點分離。
未來 effective_duration 也會放這裡，risk 模組會長成一個完整的風險分析工具箱。
這不是唯一正確答案（也有人會把它們做成 CashFlow 的方法），但這個切法在「擴充性」上更乾淨。
"""

def present_value_at_yield(cashflow: CashFlow, y: float) -> float:
    # PV(y) = sum_i CF_i * exp(-y * t_i)
    # CashFlow comes from lm.core.cashflow

    t = cashflow.times
    cf = cashflow.amounts
    return float(np.sum(cf * np.exp(-y * t)))


def macaulay_duration(cashflow: CashFlow, y: float) -> float:
    # D_mac = sum_i t_i * (CF_i * exp(-y * t_i)) / PV

    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Macaulay duration is undefined when PV is zero")
    return float(np.sum(t * pv_weights) / pv)


def modified_duration(cashflow: CashFlow, y: float) -> float:
    # D_mod = -(1/PV) * dPV/dy

    # Under continuous compounding, modified duration equals Macaulay
    return macaulay_duration(cashflow, y)


def convexity(cashflow: CashFlow, y: float) -> float:
    # C = (1/PV) * d2PV/dy2 = sum_i t_i^2 * (CF_i * exp(-y * t_i)) / PV

    t = cashflow.times
    cf = cashflow.amounts
    pv_weights = cf * np.exp(-y * t)
    pv = np.sum(pv_weights)
    if pv == 0:
        raise ValueError("Convexity is undefined when PV is zero")
    return float(np.sum(t**2 * pv_weights) / pv)