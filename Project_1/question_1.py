import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Função escolhida
def f(x):
    return np.cos(x) - x


# Precisão e truncamento
def trunc(x, decimals=4):
    """Trunca para um número fixo de casas decimais."""
    fator = 10 ** decimals
    return math.trunc(x * fator) / fator


def aplicar_precisao(x, modo):
    """Aplica o modo de precisão desejado."""
    if modo == "float64":
        return np.float64(x)
    elif modo == "float32":
        return np.float32(x)
    elif modo == "trunc4":
        return np.float64(trunc(float(x), 4))
    else:
        raise ValueError("Modo inválido: use float64, float32 ou trunc4.")


# Método da Bisseção
def bissecao(f, a, b, tol_x, tol_f, maxit, modo, raiz_real=None):
    aplicar = lambda x: aplicar_precisao(x, modo)
    a, b = aplicar(a), aplicar(b)
    fa, fb = aplicar(f(a)), aplicar(f(b))
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) têm o mesmo sinal.")

    dados = []
    for k in range(1, maxit + 1):
        m = aplicar((a + b) / 2)
        fm = aplicar(f(m))
        erro = abs(float(m) - raiz_real) if raiz_real else np.nan
        dados.append([k, float(m), float(fm), erro])
        if abs(fm) <= tol_f or abs(b - a) / 2 <= tol_x:
            break
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return pd.DataFrame(dados, columns=["Iteração", "x", "f(x)", "Erro"])


# Método da Falsa Posição
def falsa_posicao(f, a, b, tol_x, tol_f, maxit, modo, raiz_real=None):
    aplicar = lambda x: aplicar_precisao(x, modo)
    a, b = aplicar(a), aplicar(b)
    fa, fb = aplicar(f(a)), aplicar(f(b))
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) têm o mesmo sinal.")

    dados = []
    for k in range(1, maxit + 1):
        denom = aplicar(fb - fa)
        if denom == 0:
            x = aplicar((a + b) / 2)
        else:
            x = aplicar(a - fa * (b - a) / denom)
        fx = aplicar(f(x))
        erro = abs(float(x) - raiz_real) if raiz_real else np.nan
        dados.append([k, float(x), float(fx), erro])
        if abs(fx) <= tol_f or abs(b - a) <= tol_x:
            break
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    return pd.DataFrame(dados, columns=["Iteração", "x", "f(x)", "Erro"])


# Executa para os 3 modos de precisão e gera gráficos e tabelas
def executar(metodo, nome_metodo, f, a, b, raiz_real):
    configuracoes = [
        ("float64", 1e-12, 1e-12),
        ("float32", 1e-6, 1e-6),
        ("trunc4", 1e-4, 1e-4),
    ]
    resultados = []

    for modo, tolx, tolf in configuracoes:
        df = metodo(f, a, b, tolx, tolf, 50, modo, raiz_real)
        resultados.append((modo, df))

        # Gráfico de f(x) com os pontos
        x_vals = np.linspace(a, b, 300)
        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, f(x_vals), 'b-', label='f(x)')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.scatter(df["x"], df["f(x)"], color='blue', marker='x', label='f aprox')
        plt.scatter(df["x"], np.zeros_like(df["x"]), color='red', label='raiz aprox')
        plt.title(f"{nome_metodo} – {modo}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"grafico_funcao_{nome_metodo}_{modo}.png")
        plt.close()

        # Gráfico da convergência (erro)
        plt.figure(figsize=(7, 5))
        plt.semilogy(df["Iteração"], df["Erro"], marker='o')
        plt.title(f"Convergência do erro – {nome_metodo} ({modo})")
        plt.xlabel("Iteração")
        plt.ylabel("|xₖ - raiz exata|")
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.savefig(f"convergencia_{nome_metodo}_{modo}.png")
        plt.close()

    return resultados


a, b = 0, 1

# raiz real com alta precisão
raiz_real = 0.7390851332151607

# Métodos
result_bis = executar(bissecao, "Bissecao", f, a, b, raiz_real)
result_fp = executar(falsa_posicao, "FalsaPosicao", f, a, b, raiz_real)

# Compara os resultados
print("\nResumo final:")
print("---------------------------------------------------------------")
for nome_metodo, resultados in [("Bisseção", result_bis), ("Falsa Posição", result_fp)]:
    print(f"\nMétodo: {nome_metodo}")
    for modo, df in resultados:
        ultima = df.iloc[-1]
        print(
            f"  {modo:<8} → iterações: {len(df):2d},  raiz ≈ {ultima['x']:.10f},  |f(x)| = {abs(ultima['f(x)']):.2e},  erro = {ultima['Erro']:.2e}")
        df.to_csv(f"tabela_{nome_metodo}_{modo}.csv", index=False)
