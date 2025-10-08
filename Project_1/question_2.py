import numpy as np
import matplotlib.pyplot as plt
import time

# --- Definição das Funções e Derivadas ---

# Função 1: f(x) = x^3 - 7x + 6
def f1(x):
    return x**3 - 7*x + 6

def f1_prime(x):
    return 3*x**2 - 7

# Função de iteração para o Ponto Fixo de f1(x)
# f(x) = 0 => x^3 - 7x + 6 = 0 => 7x = x^3 + 6 => x = (x^3 + 6) / 7
# Checando a convergência perto da raiz x=2: |phi'(2)| = |3*(2^2)/7| = 12/7 > 1. Não vai convergir.
# Vamos usar outra forma: x = sqrt(7x - 6) que foca na raiz x=2 e x=1.
# Ou melhor: x = (7x-6)^(1/3)
def phi1(x):
    # Usaremos uma forma que converge para a raiz x=2, partindo de x > 1.5
    # x^3 = 7x - 6 => x = (7x - 6)^(1/3)
    # A derivada de phi1 é (1/3)*(7x-6)^(-2/3)*7. Para x=2, |phi'(2)|=7/3*(8)^(-2/3)=7/12 < 1. Converge!
    if (7*x - 6) < 0:
        return 0 # Evita erro de domínio
    return (7*x - 6)**(1/3)


# Função 2: f(x) = ln(x + 1) + x - 2
def f2(x):
    return np.log(x + 1) + x - 2

def f2_prime(x):
    return 1/(x + 1) + 1

# Função de iteração para o Ponto Fixo de f2(x)
# f(x) = 0 => x = 2 - ln(x + 1)
# Checando a convergência perto da raiz x=1.55: |phi'(1.55)| = |-1/(1.55+1)| approx 0.39 < 1. Converge!
def phi2(x):
    return 2 - np.log(x + 1)


# --- Implementação dos Métodos Numéricos ---

def ponto_fixo(phi, x0, tol=1e-8, max_iter=100):
    """Método do Ponto Fixo"""
    start_time = time.perf_counter_ns()
    x = x0
    erros = []
    iteracoes = 0
    for i in range(max_iter):
        iteracoes += 1
        x_novo = phi(x)
        erro = abs(x_novo - x)
        erros.append(erro)
        if erro < tol:
            break
        x = x_novo
    end_time = time.perf_counter_ns()
    return x, erros, iteracoes, (end_time - start_time) / 1000 # em microssegundos

def newton_raphson(f, f_prime, x0, tol=1e-8, max_iter=100):
    """Método de Newton-Raphson"""
    start_time = time.perf_counter_ns()
    x = x0
    erros = []
    iteracoes = 0
    for i in range(max_iter):
        iteracoes += 1
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-12: # Evita divisão por zero
            print("Derivada próxima de zero!")
            break
        x_novo = x - fx / fpx
        erro = abs(x_novo - x)
        erros.append(erro)
        if erro < tol:
            break
        x = x_novo
    end_time = time.perf_counter_ns()
    return x, erros, iteracoes, (end_time - start_time) / 1000

def secante(f, x0, x1, tol=1e-8, max_iter=100):
    """Método da Secante"""
    start_time = time.perf_counter_ns()
    erros = []
    iteracoes = 0
    for i in range(max_iter):
        iteracoes += 1
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1 - fx0) < 1e-12: # Evita divisão por zero
            break
        x_novo = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        erro = abs(x_novo - x1)
        erros.append(erro)
        if erro < tol:
            break
        x0, x1 = x1, x_novo
    end_time = time.perf_counter_ns()
    return x1, erros, iteracoes, (end_time - start_time) / 1000


# --- Execução e Coleta de Dados ---

TOLERANCIA = 1e-9
MAX_ITERACOES = 50

# Raízes verdadeiras para cálculo do erro final
raiz_f1_verdadeira = 2.0
raiz_f2_verdadeira = 1.557145598997603

resultados = {}

# Testes para f1(x) = x^3 - 7x + 6 (Raiz = 2)
x0_f1 = 2.5 # Chute inicial
x1_f1 = 2.4 # Chute secundário para Secante

resultados['Ponto Fixo_f1'] = ponto_fixo(phi1, x0_f1, TOLERANCIA, MAX_ITERACOES)
resultados['Newton_f1'] = newton_raphson(f1, f1_prime, x0_f1, TOLERANCIA, MAX_ITERACOES)
resultados['Secante_f1'] = secante(f1, x0_f1, x1_f1, TOLERANCIA, MAX_ITERACOES)

# Testes para f2(x) = ln(x+1) + x - 2 (Raiz ~ 1.557)
x0_f2 = 1.5 # Chute inicial
x1_f2 = 1.4 # Chute secundário para Secante

resultados['Ponto Fixo_f2'] = ponto_fixo(phi2, x0_f2, TOLERANCIA, MAX_ITERACOES)
resultados['Newton_f2'] = newton_raphson(f2, f2_prime, x0_f2, TOLERANCIA, MAX_ITERACOES)
resultados['Secante_f2'] = secante(f2, x0_f2, x1_f2, TOLERANCIA, MAX_ITERACOES)


# --- Apresentação dos Resultados ---

# 1. Tabela Comparativa
print("--------------------------------------------------------------------------------------")
print("                          TABELA COMPARATIVA DE DESEMPENHO")
print("--------------------------------------------------------------------------------------")
print(f"{'Função':<10} | {'Método':<15} | {'Raiz Encontrada':<22} | {'Iterações':<10} | {'Tempo (µs)':<12} | {'Erro Final':<15}")
print("--------------------------------------------------------------------------------------")

# Função 1
raiz, _, it, t = resultados['Ponto Fixo_f1']
print(f"{'f1(x)':<10} | {'Ponto Fixo':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f1_verdadeira):<15.2e}")
raiz, _, it, t = resultados['Newton_f1']
print(f"{'f1(x)':<10} | {'Newton-Raphson':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f1_verdadeira):<15.2e}")
raiz, _, it, t = resultados['Secante_f1']
print(f"{'f1(x)':<10} | {'Secante':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f1_verdadeira):<15.2e}")
print("--------------------------------------------------------------------------------------")

# Função 2
raiz, _, it, t = resultados['Ponto Fixo_f2']
print(f"{'f2(x)':<10} | {'Ponto Fixo':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f2_verdadeira):<15.2e}")
raiz, _, it, t = resultados['Newton_f2']
print(f"{'f2(x)':<10} | {'Newton-Raphson':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f2_verdadeira):<15.2e}")
raiz, _, it, t = resultados['Secante_f2']
print(f"{'f2(x)':<10} | {'Secante':<15} | {raiz:<22.15f} | {it:<10} | {t:<12.2f} | {abs(raiz-raiz_f2_verdadeira):<15.2e}")
print("--------------------------------------------------------------------------------------")


# 2. Gráficos de Convergência
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico para f1(x)
axs[0].plot(np.log10(resultados['Ponto Fixo_f1'][1]), '-o', label='Ponto Fixo')
axs[0].plot(np.log10(resultados['Newton_f1'][1]), '-o', label='Newton-Raphson')
axs[0].plot(np.log10(resultados['Secante_f1'][1]), '-o', label='Secante')
axs[0].set_title('Convergência para $f_1(x) = x^3 - 7x + 6$')
axs[0].set_xlabel('Iteração')
axs[0].set_ylabel('Log10(Erro Absoluto)')
axs[0].legend()
axs[0].grid(True)

# Gráfico para f2(x)
axs[1].plot(np.log10(resultados['Ponto Fixo_f2'][1]), '-o', label='Ponto Fixo')
axs[1].plot(np.log10(resultados['Newton_f2'][1]), '-o', label='Newton-Raphson')
axs[1].plot(np.log10(resultados['Secante_f2'][1]), '-o', label='Secante')
axs[1].set_title('Convergência para $f_2(x) = \ln(x+1) + x - 2$')
axs[1].set_xlabel('Iteração')
axs[1].set_ylabel('Log10(Erro Absoluto)')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()