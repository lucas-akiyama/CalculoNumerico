# Distância percorrida por um carro.
# 1. dados de velocidade tabelados
# 2. Calcule a distância percorrida usando os métodos dos trapézios, Simpson
# e Newton-Cotes
# 3. Compare os resultados entre os métodos e discuta como a escolha do número
# influencia a precisão

from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Definição dos métodos
def metodo_trapezios(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Calcula a aproximação da integral definida usando a Regra dos Trapézios Composta.

    Parâmetros:
        f (Callable): Função a ser integrada.
        a (float): Limite inferior.
        b (float): Limite superior.
        n (int): Número de subintervalos.

    Retorno:
        float: Aproximação da integral.
    """
    if n <= 0:
        raise ValueError("O número de subintervalos 'n' deve ser maior que zero.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))


def metodo_simpson_13(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Calcula a aproximação usando a Regra de Simpson 1/3 (Newton-Cotes n=2).
    Requisito: 'n' deve ser par.
    """
    if n % 2 != 0:
        raise ValueError("Para o Método de Simpson 1/3, 'n' deve ser um número PAR.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    soma_impares = np.sum(y[1:-1:2])
    soma_pares = np.sum(y[2:-1:2])
    return (h / 3) * (y[0] + 4 * soma_impares + 2 * soma_pares + y[-1])


def metodo_simpson_38(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    Calcula a aproximação usando a Regra de Simpson 3/8 (Newton-Cotes n=3).
    Requisito: 'n' deve ser múltiplo de 3.
    """
    if n % 3 != 0:
        raise ValueError("Para o Método de Simpson 3/8, 'n' deve ser múltiplo de 3.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    soma_mult_3 = np.sum(y[3:-1:3])
    soma_interna_total = np.sum(y[1:-1])
    soma_resto = soma_interna_total - soma_mult_3
    return (3 * h / 8) * (y[0] + 3 * soma_resto + 2 * soma_mult_3 + y[-1])


def metodo_newton_cotes_grau_4(f: Callable[[float], float], a: float, b: float,
                               n: int) -> float:
    """
    Calcula a aproximação da integral definida usando a Regra de Boole
    (Newton-Cotes fechada de grau n=4).

    A fórmula base utiliza os pesos [7, 32, 12, 32, 7] multiplicados por 2h/45.

    Requisito:
        O número de subintervalos 'n' deve ser múltiplo de 4.

    Parâmetros:
        f (Callable): Função a ser integrada.
        a (float): Limite inferior.
        b (float): Limite superior.
        n (int): Número de subintervalos.

    Retorno:
        float: Aproximação da integral.
    """
    if n % 4 != 0:
        raise ValueError("Para a Regra de Boole (grau 4), 'n' deve ser múltiplo de 4.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    soma_impares = np.sum(y[1:-1:2])
    soma_mult_2_nao_4 = np.sum(y[2:-1:4])
    soma_mult_4 = np.sum(y[4:-1:4])
    coeficiente_global = (2 * h) / 45
    soma_ponderada = (7 * (y[0] + y[-1]) +
                      32 * soma_impares +
                      12 * soma_mult_2_nao_4 +
                      14 * soma_mult_4)
    return coeficiente_global * soma_ponderada


# Funções auxiliáres
def plotar_grafico_velocidade(f: Callable[[float], float], a: float, b: float, n: int):
    """
    Gera um gráfico visualizando a função, os pontos de discretização e a área integrada.

    Parâmetros:
        f (Callable): A função v(t).
        a (float): Tempo inicial.
        b (float): Tempo final.
        n (int): Número de subintervalos (pontos da tabela).
    """

    # 1. Dados para a Curva Suave (Alta resolução para visualização)
    # Usamos 200 pontos para dar a aparência de continuidade perfeita
    t_continuo = np.linspace(a, b, 200)
    v_continuo = f(t_continuo)

    # 2. Dados Discretos (Exatamente os pontos usados nos métodos numéricos)
    t_discreto = np.linspace(a, b, n + 1)
    v_discreto = f(t_discreto)

    # Configuração do Plot
    plt.figure(figsize=(10, 6))

    # Plot da curva contínua
    plt.plot(t_continuo, v_continuo, label=r'Função $v(t)$ (Contínua)',
             color='navy', linewidth=2, zorder=1)
    # Plot da área sob a curva (O significado da Integral)
    plt.fill_between(t_continuo, v_continuo, alpha=0.2, color='skyblue',
                     label='Deslocamento ($\int v(t) dt$)')

    # Plot dos pontos discretos (Nós de integração)
    plt.scatter(t_discreto, v_discreto, color='red', zorder=2, s=50,
                label=f'Pontos Tabelados (n={n})')

    # Plot das linhas verticais para visualizar os trapézios/faixas
    for t_val in t_discreto:
        plt.axvline(x=t_val, color='gray', linestyle=':', alpha=0.5, ymax=0.95)

    # Configurações Estéticas (Linguagem Acadêmica)
    plt.title(f"Perfil de Velocidade e Discretização (n={n})", fontsize=14,
              fontweight='bold')
    plt.xlabel("Tempo $t$ [s]", fontsize=12)
    plt.ylabel("Velocidade $v(t)$ [m/s]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', frameon=True, shadow=True)

    # Ajuste de limites para melhor visualização
    plt.xlim(a, b)
    plt.ylim(0, max(v_continuo) * 1.1)

    plt.tight_layout()
    plt.show()


def plotar_comparacao_erros(f_velocidade: Callable,
                            f_posicao_analitica: Callable,
                            a: float, b: float, n: int):
    """
    Gera um gráfico de barras comparando o Erro Absoluto (em escala logarítmica)
    de cada método de integração numérica.

    Parâmetros:
        f_velocidade: Função v(t) a ser integrada.
        f_posicao_analitica: Função s(t) primitiva para cálculo do erro exato.
        a, b: Intervalo de integração.
        n: Número de subintervalos (deve ser múltiplo de 4 e 3 para funcionar com todos).
    """

    # 1. Cálculo do Valor Exato (Referência)
    valor_exato = f_posicao_analitica(b) - f_posicao_analitica(a)

    # 2. Execução dos Métodos Numéricos
    # (Assumindo que as funções foram definidas nas etapas anteriores)
    try:
        val_trap = metodo_trapezios(f_velocidade, a, b, n)
        val_s13 = metodo_simpson_13(f_velocidade, a, b, n)
        val_s38 = metodo_simpson_38(f_velocidade, a, b, n)
        val_boole = metodo_newton_cotes_grau_4(f_velocidade, a, b, n)
    except NameError:
        print("Erro: As funções dos métodos numéricos precisam estar definidas na memória.")
        return
    except ValueError as e:
        print(f"Erro de validação de 'n': {e}")
        return

    # 3. Cálculo dos Erros Absolutos
    # Adicionamos um epsilon minúsculo para evitar log(0) caso o erro seja zero perfeito
    epsilon = 1e-16
    erros = [
        abs(val_trap - valor_exato) + epsilon,
        abs(val_s13 - valor_exato) + epsilon,
        abs(val_s38 - valor_exato) + epsilon,
        abs(val_boole - valor_exato) + epsilon
    ]

    nomes_metodos = ['Trapézios\n(Ordem 1)', 'Simpson 1/3\n(Ordem 2)',
                     'Simpson 3/8\n(Ordem 3)', 'Boole\n(Ordem 4)']

    # Cores acadêmicas distintas
    cores = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']  # Vermelho, Azul, Roxo, Verde

    # 4. Configuração do Gráfico
    plt.figure(figsize=(10, 6))

    barras = plt.bar(nomes_metodos, erros, color=cores, alpha=0.8, edgecolor='black')

    # Configuração da Escala Logarítmica (CRUCIAL para esta comparação)
    plt.yscale('log')

    # Títulos e Rótulos
    plt.title(f"Comparativo de Erro Absoluto por Método (n={n})", fontsize=14,
              fontweight='bold')
    plt.ylabel("Erro Absoluto (Escala Log) $|I_{num} - I_{exato}|$", fontsize=12)
    plt.xlabel("Método de Integração (Newton-Cotes)", fontsize=12)

    # Grid focado no eixo Y para facilitar leitura da escala log
    plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)

    # 5. Anotação dos valores sobre as barras (Notação Científica)
    for barra, erro in zip(barras, erros):
        height = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2., height * 1.2,
                 f'{erro:.1e}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


# Problema
def velocidade_carro(t):
    """
    v(t) = 30 * (1 - e^(-0.5t)) + 2 * sin(t)
    Retorna velocidade em m/s.
    """
    return 30 * (1 - np.exp(-0.5 * t)) + 2 * np.sin(t)

def posicao_analitica(t):
    """
    Integral indefinida (primitiva) de v(t):
    s(t) = 30t + 60e^(-0.5t) - 2cos(t) + C
    Considerando s(0) = 0 para achar C:
    0 = 0 + 60 - 2 + C => C = -58
    """
    termo_linear = 30 * t
    termo_exp = 60 * np.exp(-0.5 * t)
    termo_trig = -2 * np.cos(t)
    constante_integracao = -58
    return termo_linear + termo_exp + termo_trig + constante_integracao


def main():
    # Parâmetros
    t_inicial = 0.0
    t_final = 12.0
    n_subintervalos = 96  # Divisível por 2, 3 e 4

    # Gerar Tabela de Valores (Discretização)
    tempos = np.linspace(t_inicial, t_final, n_subintervalos + 1)
    velocidades = velocidade_carro(tempos)

    df_tabela = pd.DataFrame({
        'Tempo (s)': tempos,
        'Velocidade (m/s)': velocidades
    })

    print("### Tabela de Dados Coletados (Velocidade x Tempo) ###")
    print(df_tabela.to_string(index=False))
    print("-" * 40)

    # Cálculo Analítico (Valor "Real")
    deslocamento_real = posicao_analitica(t_final) - posicao_analitica(t_inicial)
    # Cálculos Numéricos
    res_trap = metodo_trapezios(velocidade_carro, t_inicial, t_final, n_subintervalos)
    res_simp13 = metodo_simpson_13(velocidade_carro, t_inicial, t_final, n_subintervalos)
    res_simp38 = metodo_simpson_38(velocidade_carro, t_inicial, t_final, n_subintervalos)
    res_boole = metodo_newton_cotes_grau_4(velocidade_carro, t_inicial, t_final,
                                           n_subintervalos)

    # Exibição dos Resultados e Erros
    resultados = {
        "Método": ["Analítico (Exato)", "Trapézios (n=1)", "Simpson 1/3 (n=2)",
                   "Simpson 3/8 (n=3)", "Boole (n=4)"],
        "Deslocamento (m)": [deslocamento_real, res_trap, res_simp13, res_simp38,
                             res_boole],
        "Erro Absoluto": [0.0, abs(res_trap - deslocamento_real),
                          abs(res_simp13 - deslocamento_real),
                          abs(res_simp38 - deslocamento_real),
                          abs(res_boole - deslocamento_real)]
    }

    df_resultados = pd.DataFrame(resultados)
    print("\n### Resultados da Integração Numérica ###")
    print(df_resultados.to_string(index=False, float_format="%.6f"))

    print("\nGerando gráfico")
    plotar_grafico_velocidade(velocidade_carro, t_inicial, t_final, n_subintervalos)

    print("Gerando gráfico comparativo de erros...")
    plotar_comparacao_erros(velocidade_carro, posicao_analitica, t_inicial, t_final, n_subintervalos)


main()
