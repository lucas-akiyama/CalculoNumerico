import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Definição das Funções Físicas ---

def force_spring(x, k, alpha):
    """Lei de força da mola: F(x) = kx + alpha*x^3"""
    return k * x + alpha * (x**3)

def energy_analytical(d, k, alpha):
    """Solução Exata da integral: E = 1/2*k*d^2 + 1/4*alpha*d^4"""
    return 0.5 * k * (d**2) + 0.25 * alpha * (d**4)

# --- 2. Implementação dos Métodos Numéricos (Newton-Cotes) ---

def trapezoidal_rule(func, a, b, n, k, alpha):
    """Regra dos Trapézios Composta"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x, k, alpha)
    
    # Soma: h * (y0/2 + y1 + y2 + ... + yn/2)
    soma = (y[0] + y[-1]) / 2.0 + np.sum(y[1:-1])
    return soma * h

def simpson13_rule(func, a, b, n, k, alpha):
    """Regra 1/3 de Simpson Composta (requer n par)"""
    if n % 2 != 0: n += 1 # Ajuste forçado para n par
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x, k, alpha)
    
    # Pesos: 1, 4, 2, 4, ..., 2, 4, 1
    soma = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return (h / 3.0) * soma

def booles_rule(func, a, b, n, k, alpha):
    """
    Regra de Boole (Newton-Cotes ordem superior/grau 4).
    Requer que o número de subintervalos seja múltiplo de 4.
    """
    # Ajustar n para ser múltiplo de 4
    while n % 4 != 0: n += 1
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x, k, alpha)
    
    # A regra de Boole simples para 5 pontos (4 intervalos) tem pesos:
    # (2h/45) * (7y0 + 32y1 + 12y2 + 32y3 + 7y4)
    # Implementação composta simplificada iterando a cada 4 passos
    integral = 0
    for i in range(0, n, 4):
        # Índices locais
        y_loc = y[i:i+5]
        integral += (2 * h / 45) * (7*y_loc[0] + 32*y_loc[1] + 12*y_loc[2] + 32*y_loc[3] + 7*y_loc[4])
        
    return integral

# --- 3. Configuração do Problema ---

# Parâmetros Físicos
d_max = 0.5   # Deslocamento máximo (m)
k_const = 1000.0 # Rigidez linear (N/m)
alpha_vals = [0, 500] # Caso 1: Linear (alpha=0), Caso 2: Não Linear (alpha=500)
N_intervals = 12 # Número de subintervalos (múltiplo de 2, 3 e 4 para facilitar)

results = []

print(f"{'='*80}")
print(f"ANÁLISE DE ENERGIA DA MOLA (N={N_intervals} subintervalos)")
print(f"{'='*80}")

for alpha in alpha_vals:
    tipo = "Linear (α=0)" if alpha == 0 else "Não Linear (α=500)"
    
    # 1. Solução Exata
    E_exact = energy_analytical(d_max, k_const, alpha)
    
    # 2. Métodos Numéricos
    E_trap = trapezoidal_rule(force_spring, 0, d_max, N_intervals, k_const, alpha)
    E_simp = simpson13_rule(force_spring, 0, d_max, N_intervals, k_const, alpha)
    E_boole = booles_rule(force_spring, 0, d_max, N_intervals, k_const, alpha)
    
    # Armazenar resultados
    results.append({
        "Cenario": tipo,
        "Metodo": "Analítico (Exato)",
        "Energia (J)": E_exact,
        "Erro Absoluto": 0.0
    })
    results.append({"Cenario": tipo, "Metodo": "Trapézios", "Energia (J)": E_trap, "Erro Absoluto": abs(E_exact - E_trap)})
    results.append({"Cenario": tipo, "Metodo": "Simpson 1/3", "Energia (J)": E_simp, "Erro Absoluto": abs(E_exact - E_simp)})
    results.append({"Cenario": tipo, "Metodo": "Boole (Ordem Sup.)", "Energia (J)": E_boole, "Erro Absoluto": abs(E_exact - E_boole)})

# Criar DataFrame para exibição
df_results = pd.DataFrame(results)

# Exibir tabela formatada
print(df_results[['Cenario', 'Metodo', 'Energia (J)', 'Erro Absoluto']].to_markdown(index=False, floatfmt=".10f"))

# --- 4. Visualização Gráfica ---

x_plot = np.linspace(0, d_max, 100)
y_linear = force_spring(x_plot, k_const, 0)
y_nlinear = force_spring(x_plot, k_const, 500)

plt.figure(figsize=(10, 6))
 
# (Tag demonstrativa, o código gera o gráfico real abaixo)

plt.plot(x_plot, y_linear, 'b--', label=f'Linear (k={k_const}, α=0)')
plt.fill_between(x_plot, y_linear, alpha=0.1, color='blue')

plt.plot(x_plot, y_nlinear, 'r-', label=f'Não Linear (k={k_const}, α=500)')
plt.fill_between(x_plot, y_nlinear, alpha=0.1, color='red')

plt.title('Força vs Deslocamento: A Energia é a Área sob a Curva')
plt.xlabel('Deslocamento x (m)')
plt.ylabel('Força F(x) (N)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nDISCUSSÃO:")
print("1. Validação (Cenário Linear): Todos os métodos convergem para o valor analítico.")
print("2. Cenário Não Linear: Observe que Simpson e Boole têm erro praticamente zero.")
print("   Isso ocorre porque F(x) é um polinómio de grau 3.")
print("   A regra de Simpson integra exato polinómios até grau 3.")
print("   Portanto, para este problema específico, Simpson é tão bom quanto a solução analítica.")
