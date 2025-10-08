# -*- coding: utf-8 -*-
"""
Questão 3 — Cálculo Numérico
Critério de parada: ERRO RELATIVO < 1% para Bisseção, Newton e Secante.
Gera um PDF com: tabelas de iteração, gráficos de convergência e resumo comparativo.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import wrap

# =========================
# 1) PARÂMETROS DO PROBLEMA
# =========================
# Troque estes valores pelos seus medidos, se quiser:
g = 9.80665    # m/s^2
L = 1.00       # m
T = 1.80       # s

REL_TOL = 1e-2   # 1% de erro relativo
KMAX   = 100     # limite de iterações (segurança)

# Constante do modelo f(θ) = sin(θ) - C = 0
C = (T / (2*np.pi)) * np.sqrt(g / L)

# Checagem de existência de solução real (|C| <= 1):
if abs(C) > 1:
    raise ValueError(f"|C|={abs(C):.6f} > 1 → sem solução real para sin(θ)=C. Ajuste T, L ou g.")

# Raiz de referência para medir erro absoluto (apenas para análise/gráficos):
theta_star = float(np.arcsin(C))   # solução principal em [0, π/2]

# ==================
# 2) FUNÇÕES AUXILIARES
# ==================
def trunc4(x: float) -> float:
    """Trunca para 4 casas decimais (simula arredondamento forte por iteração)."""
    return np.trunc(x * 1e4) / 1e4

def f_factory(C, rounding=None, dtype=np.float64):
    """
    Cria f(θ) = sin(θ) - C com:
      - dtype: float64 (padrão) ou float32 (precisão reduzida),
      - rounding: função de truncamento opcional (p.ex., trunc4) aplicada a cada iteração.
    """
    C_ = dtype(C)
    def f(theta):
        t = dtype(theta)
        if rounding is not None:
            t = dtype(rounding(float(t)))
        val = dtype(np.sin(t) - C_)
        if rounding is not None:
            val = dtype(rounding(float(val)))
        return val
    return f

def df_factory(rounding=None, dtype=np.float64):
    """Cria df(θ) = cos(θ) com as mesmas opções de dtype e truncamento da f."""
    def df(theta):
        t = dtype(theta)
        if rounding is not None:
            t = dtype(rounding(float(t)))
        val = dtype(np.cos(t))
        if rounding is not None:
            val = dtype(rounding(float(val)))
        return val
    return df

def rel_err(new: float, old: float, eps: float = 1e-12) -> float:
    """
    Erro relativo entre dois sucessivos: |new - old| / max(|new|, eps).
    Usamos max(|new|, eps) para evitar divisão por zero.
    """
    denom = max(abs(new), eps)
    return abs(new - old) / denom

# =========================================
# 3) MÉTODOS NUMÉRICOS (PARADA: ER < 1%)
# =========================================
def bisection_rel(f, a, b, rel_tol=REL_TOL, kmax=KMAX, theta_star=None):
    """
    Bisseção com parada por erro relativo entre m_k e m_{k-1}.
    Pré-condição: f(a)*f(b) < 0.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisseção: f(a)*f(b) > 0 no intervalo inicial.")

    rows = []
    m_prev = None

    for k in range(kmax):
        m = 0.5 * (a + b)
        fm = f(m)
        rerr = np.nan if m_prev is None else rel_err(m, m_prev)
        aerr = abs(m - theta_star) if theta_star is not None else np.nan

        rows.append(dict(k=k, theta=float(m), f=float(fm),
                         rel_err=float(rerr), abs_err=float(aerr),
                         interval=float(b - a)))

        # critério de parada
        if m_prev is not None and rerr < rel_tol:
            break

        # próxima iteração (mantém mudança de sinal)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

        m_prev = m

    return pd.DataFrame(rows)

def newton_rel(f, df, x0, rel_tol=REL_TOL, kmax=KMAX, theta_star=None):
    """
    Newton–Raphson com parada por erro relativo entre x_{k+1} e x_k.
    Atenção: se df(x) ≈ 0, o método pode ficar instável.
    """
    rows = []
    x = float(x0)

    for k in range(kmax):
        fx, dfx = float(f(x)), float(df(x))
        if dfx == 0.0:
            # registra e para para evitar divisão por zero
            rows.append(dict(k=k, theta=float(x), f=float(fx), df=float(dfx),
                             rel_err=np.nan,
                             abs_err=(abs(x - theta_star) if theta_star is not None else np.nan),
                             delta=np.nan))
            break

        x1 = x - fx / dfx
        rerr = rel_err(x1, x)
        aerr = abs(x1 - theta_star) if theta_star is not None else np.nan

        rows.append(dict(k=k, theta=float(x1), f=float(fx), df=float(dfx),
                         rel_err=float(rerr), abs_err=float(aerr),
                         delta=float(x1 - x)))

        if rerr < rel_tol:
            x = x1
            break

        x = x1

    return pd.DataFrame(rows)

def secant_rel(f, x0, x1, rel_tol=REL_TOL, kmax=KMAX, theta_star=None):
    """
    Secante com parada por erro relativo entre x_{k+1} e x_k.
    Não usa derivada; cuidado com denominador pequeno.
    """
    rows = []
    x_prev = float(x0)
    x      = float(x1)
    f_prev = float(f(x_prev))
    fx     = float(f(x))

    for k in range(kmax):
        denom = fx - f_prev
        if denom == 0.0:
            rows.append(dict(k=k, theta=float(x), f=float(fx), f_prev=float(f_prev),
                             rel_err=np.nan,
                             abs_err=(abs(x - theta_star) if theta_star is not None else np.nan),
                             delta=np.nan))
            break

        x_next = x - fx * (x - x_prev) / denom
        rerr   = rel_err(x_next, x)
        aerr   = abs(x_next - theta_star) if theta_star is not None else np.nan

        rows.append(dict(k=k, theta=float(x_next), f=float(fx), f_prev=float(f_prev),
                         rel_err=float(rerr), abs_err=float(aerr),
                         delta=float(x_next - x)))

        if rerr < rel_tol:
            x = x_next
            break

        x_prev, f_prev = x, fx
        x, fx = x_next, float(f(x_next))

    return pd.DataFrame(rows)

# =================
# 4) EXECUÇÃO
# =================
def run_all():
    # Precisões/simulações:
    f64  = f_factory(C, dtype=np.float64); df64  = df_factory(dtype=np.float64)
    f32  = f_factory(C, dtype=np.float32); df32  = df_factory(dtype=np.float32)
    ftr  = f_factory(C, rounding=trunc4, dtype=np.float64); dftr = df_factory(rounding=trunc4, dtype=np.float64)

    # Intervalo inicial e palpites:
    a0, b0 = 0.0, float(np.pi / 2)  # mudança de sinal garantida para 0 < C < 1
    x0_newton = 0.7
    x0_sec, x1_sec = 0.0, float(np.pi / 2)

    # Rodadas (parada por erro relativo < 1%)
    bis64 = bisection_rel(f64, a0, b0, theta_star=theta_star)
    new64 = newton_rel   (f64, df64, x0_newton, theta_star=theta_star)
    sec64 = secant_rel   (f64, x0_sec, x1_sec, theta_star=theta_star)

    bis32 = bisection_rel(f32, a0, b0, theta_star=theta_star)
    new32 = newton_rel   (f32, df32, x0_newton, theta_star=theta_star)
    sec32 = secant_rel   (f32, x0_sec, x1_sec, theta_star=theta_star)

    bis_tr = bisection_rel(ftr, a0, b0, theta_star=theta_star)
    new_tr = newton_rel   (ftr, dftr, x0_newton, theta_star=theta_star)
    sec_tr = secant_rel   (ftr, x0_sec, x1_sec, theta_star=theta_star)

    return {
        "bis64": bis64, "new64": new64, "sec64": sec64,
        "bis32": bis32, "new32": new32, "sec32": sec32,
        "bis_tr": bis_tr, "new_tr": new_tr, "sec_tr": sec_tr
    }

def summary_table(dfs: dict) -> pd.DataFrame:
    """Monta uma tabela resumo a partir da última linha de cada método."""
    def last_row(df): return df.iloc[-1].to_dict()
    rows = []
    mapping = [
        ("Bisseção (float64)", dfs["bis64"]),
        ("Newton (float64)"  , dfs["new64"]),
        ("Secante (float64)" , dfs["sec64"]),
        ("Bisseção (float32)", dfs["bis32"]),
        ("Newton (float32)"  , dfs["new32"]),
        ("Secante (float32)" , dfs["sec32"]),
        ("Bisseção (trunc4)" , dfs["bis_tr"]),
        ("Newton (trunc4)"   , dfs["new_tr"]),
        ("Secante (trunc4)"  , dfs["sec_tr"]),
    ]
    for name, dfm in mapping:
        lr = last_row(dfm)
        rows.append({
            "Método": name,
            "Iterações (k_final)": int(lr["k"]),
            "θ_aprox (rad)": lr["theta"],
            "Erro relativo (último)": lr.get("rel_err", np.nan),
            "Erro abs vs θ*": lr.get("abs_err", np.nan),
        })
    return pd.DataFrame(rows)

# =================
# 5) GERAÇÃO DO PDF
# =================
def add_table_page(pp, df: pd.DataFrame, title: str, max_first=20, max_last=5):
    """Adiciona uma página com uma tabela (corta meio-termo se for muito grande)."""
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.95, title, ha="center", fontsize=14)
    show_df = df.copy()
    if len(show_df) > (max_first + max_last):
        head, tail = show_df.head(max_first), show_df.tail(max_last)
        show_df = pd.concat([head,
                             pd.DataFrame([["..."] * len(head.columns)], columns=head.columns),
                             tail], ignore_index=True)
    ax = fig.add_subplot(111); ax.axis("off")
    tbl = ax.table(cellText=show_df.values, colLabels=show_df.columns, loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
    pp.savefig(fig); plt.close(fig)

def add_error_plot(pp, df: pd.DataFrame, title: str):
    """Adiciona um gráfico |θ_k − θ*| vs k (escala log)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_subplot(111)
    ax.plot(df["k"], df["abs_err"], marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("Iteração k")
    ax.set_ylabel("|θ_k − θ*| (log)")
    ax.set_title(title)
    pp.savefig(fig); plt.close(fig)

def build_pdf(dfs: dict, df_sum: pd.DataFrame, out_path="Q3_Pendulo_Guizo_rel1pct.pdf"):
    pp = PdfPages(out_path)

    # Capa
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.92, "Questão 3 — Parada por Erro Relativo < 1%", ha="center", fontsize=18)
    fig.text(0.5, 0.86, "Bisseção · Newton–Raphson · Secante", ha="center", fontsize=13)
    cap = [
        "Equação: f(θ) = sin(θ) − C,  C = (T/2π)·√(g/L)",
        f"Parâmetros: T={T:.3f} s, L={L:.2f} m, g={g:.5f} m/s²",
        f"C = {C:.6f} | θ* (ref) = {theta_star:.8f} rad = {np.degrees(theta_star):.6f}°",
        "Critério de parada: |θ_{k+1}−θ_k| / |θ_{k+1}| < 1%",
    ]
    y = 0.74
    for line in cap:
        fig.text(0.5, y, line, ha="center", fontsize=11); y -= 0.04
    pp.savefig(fig); plt.close(fig)

    # Página de modelo/hipóteses
    fig = plt.figure(figsize=(8.27, 11.69))
    txt = (
        "Objetivo: resolver sin(θ) = C para estimar a amplitude inicial θ.\n"
        "Métodos: bisseção (robusto), Newton (rápido — requer df=cos), secante (sem derivada).\n"
        "Parada: erro relativo < 1% ou k_max. Também registramos |θ−θ*| apenas para análise.\n"
        "Simulações de precisão: float64, float32 e truncamento em 4 casas decimais por iteração."
    )
    fig.text(0.08, 0.92, "Modelo e hipóteses", fontsize=14)
    fig.text(0.08, 0.88, "\n".join(wrap(txt, 100)), fontsize=10, va="top")
    pp.savefig(fig); plt.close(fig)

    # Tabelas (float64)
    add_table_page(pp, dfs["bis64"], "Iterações — Bisseção (float64) — critério: ER < 1%")
    add_table_page(pp, dfs["new64"], "Iterações — Newton (float64) — critério: ER < 1%")
    add_table_page(pp, dfs["sec64"], "Iterações — Secante (float64) — critério: ER < 1%")

    # Gráficos de erro absoluto (apenas informativo)
    add_error_plot(pp, dfs["bis64"], "Convergência: |θ−θ*| — Bisseção (float64)")
    add_error_plot(pp, dfs["new64"], "Convergência: |θ−θ*| — Newton (float64)")
    add_error_plot(pp, dfs["sec64"], "Convergência: |θ−θ*| — Secante (float64)")

    # Resumo comparativo e demais precisões
    add_table_page(pp, df_sum, "Resumo — Métodos × Precisões (parada por ER < 1%)")
    for key, title in [
        ("bis32", "Iterações — Bisseção (float32)"),
        ("new32", "Iterações — Newton (float32)"),
        ("sec32", "Iterações — Secante (float32)"),
        ("bis_tr", "Iterações — Bisseção (truncamento 4 casas)"),
        ("new_tr", "Iterações — Newton (truncamento 4 casas)"),
        ("sec_tr", "Iterações — Secante (truncamento 4 casas)"),
    ]:
        add_table_page(pp, dfs[key], f"{title} — critério: ER < 1%")

    pp.close()
    print(f"PDF gerado: {out_path}")

# =================
# 6) MAIN
# =================
if __name__ == "__main__":
    dfs = run_all()
    df_sum = summary_table(dfs)
    # Salva resumo em CSV (opcional)
    df_sum.to_csv("resumo_rel1pct.csv", index=False)
    # Gera PDF completo
    build_pdf(dfs, df_sum, out_path="Q3_Pendulo_Guizo_rel1pct.pdf")
    # Printa um resumo no terminal
    print(df_sum)
