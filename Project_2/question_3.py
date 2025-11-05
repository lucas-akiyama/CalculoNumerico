import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# --- 1. Carregar os Dados do Excel ---

# Por favor, substitua pelo caminho correto do seu arquivo
file_path = 'src/projecoes_2024_tab4_indicadores.xlsx' 

try:
    df = pd.read_excel(
        file_path, 
        header=6,
        usecols=['ANO', 'POP_T', 'LOCAL']  # Carrega apenas as colunas que nos interessam
    )

    print("--- Dados carregados do Excel com sucesso ---")
    #print(df.head()) # Mostra as 5 primeiras linhas

    # --- 3. Preparar os Vetores para o Cálculo Numérico ---

    # Vamos usar os dados de 2000 a 2024 para "treinar" o modelo
    df_treino = df[(df['ANO'] <= 2024) & (df['LOCAL'] == 'Brasil')].copy()

    # Normalizar o tempo (t=0 será o ano 2000)
    # Isso é CRUCIAL para a estabilidade numérica dos polinômios (NORMALIZAÇÃO)
    df_treino['t'] = df_treino['ANO'] - 2000

    # Criar os arrays NumPy que usaremos
    t = df_treino['t'].to_numpy(dtype=float)
    
    # Usar a população em milhões evita números gigantescos nos cálculos
    P = df_treino['POP_T'].to_numpy(dtype=float) / 1_000_000 

    #print("\n--- Arrays NumPy prontos para o ajuste ---")    #Essas linhas foram adicionadas apenas para verificar se as bases foram tratadas e carregadas corretamente, para verificar, apenas descmoente.
    #print("Vetor t (anos normalizados):")                    #Essas linhas foram adicionadas apenas para verificar se as bases foram tratadas e carregadas corretamente, para verificar, apenas descmoente.
    #print(t)                                                 #Essas linhas foram adicionadas apenas para verificar se as bases foram tratadas e carregadas corretamente, para verificar, apenas descmoente.
    #print("\nVetor P (população em milhões):")               #Essas linhas foram adicionadas apenas para verificar se as bases foram tratadas e carregadas corretamente, para verificar, apenas descmoente.
    #print(P)                                                 #Essas linhas foram adicionadas apenas para verificar se as bases foram tratadas e carregadas corretamente, para verificar, apenas descmoente.


except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado.")
    print("Por favor, verifique se o nome e o caminho do arquivo estão corretos.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
    print("Verifique se o parâmetro 'header=6' corresponde ao seu arquivo.")


# --- 2. Ajuste Polinomial (Grau 2) ---
# Modelo: P(t) = a0 + a1*t + a2*t^2

print("\n--- 2. Ajuste Polinomial (Grau 2) ---")
grau = 2
    
# Montar a Matriz A (colunas: t^0, t^1, t^2)
A_poly = np.column_stack([t**i for i in range(grau + 1)])
    
# Resolver o sistema (A^T A) * a = (A^T P) usando a função pronta do numpy
# Isso nos dá o vetor de coeficientes 'a' (a0, a1, a2)
a_coeffs_poly, _, _, _ = np.linalg.lstsq(A_poly, P, rcond=None)
    
a0_poly, a1_poly, a2_poly = a_coeffs_poly
    
print(f"Coeficientes: a0={a0_poly:.4f}, a1={a1_poly:.4f}, a2={a2_poly:.4f}")
    
# Calcular valores previstos, resíduos e EQM
P_pred_poly = A_poly @ a_coeffs_poly  # P_pred = a0 + a1*t + a2*t^2
residuos_poly = P - P_pred_poly
eqm_poly = np.mean(residuos_poly**2)
    
print(f"Erro Quadrático Médio (EQM) Polinomial: {eqm_poly:.6f}")


# --- 3. Ajuste Exponencial (com Linearização) ---
# Modelo: P(t) = a * e^(b*t)
    
print("\n--- 3. Ajuste Exponencial (Linearização) ---")
    
    # Passo de Linearização:
    # ln(P) = ln(a * e^(b*t))
    # ln(P) = ln(a) + ln(e^(b*t))
    # ln(P) = ln(a) + b*t
    #
    # Isso é uma reta! Y = A0 + A1*t
    # Onde: Y = ln(P), A0 = ln(a), A1 = b
    
Y = np.log(P) # Nosso novo vetor "y" é o log da população
    
    # Montar a Matriz A para a reta (grau 1)
A_exp_linear = np.column_stack([t**i for i in range(1 + 1)]) # Colunas: t^0, t^1
    
    # Resolver o sistema linear para encontrar A0 e A1
A_coeffs_linear, _, _, _ = np.linalg.lstsq(A_exp_linear, Y, rcond=None)
    
A0, A1 = A_coeffs_linear
    
print(f"Coeficientes linearizados: A0(ln(a))={A0:.4f}, A1(b)={A1:.4f}")

    # "Desfazer" a linearização para encontrar 'a' e 'b'
a_exp = np.exp(A0)
b_exp = A1
    
print(f"Coeficientes Originais: a={a_exp:.4f}, b={b_exp:.4f}")

    # Calcular valores previstos (usando o modelo original!), resíduos e EQM
P_pred_exp = a_exp * np.exp(b_exp * t)
residuos_exp = P - P_pred_exp
eqm_exp = np.mean(residuos_exp**2)

print(f"Erro Quadrático Médio (EQM) Exponencial: {eqm_exp:.6f}")
    

# --- 4. Comparação e Gráficos ---
    
print("\n--- 4. Tabela de Comparação ---")
print(f"Modelo         | EQM      | Coeficientes")
print(f"-----------------|----------|---------------------------------")
print(f"Polinomial (G2)  | {eqm_poly:<8.6f} | a0={a0_poly:.2f}, a1={a1_poly:.2f}, a2={a2_poly:.2f}")
print(f"Exponencial      | {eqm_exp:<8.6f} | a={a_exp:.2f}, b={b_exp:.2f}")

    # Gerar pontos para as curvas ficarem suaves no gráfico
t_curva = np.linspace(min(t), max(t), 200)
P_curva_poly = a0_poly + a1_poly*t_curva + a2_poly*t_curva**2
P_curva_exp = a_exp * np.exp(b_exp * t_curva)
    
plt.figure(figsize=(12, 8))
plt.plot(t, P, 'o', label='Dados Originais IBGE (em milhões)', color='blue')
plt.plot(t_curva, P_curva_poly, label=f'Ajuste Polinomial (EQM: {eqm_poly:.4f})', color='red', linestyle='--')
plt.plot(t_curva, P_curva_exp, label=f'Ajuste Exponencial (EQM: {eqm_exp:.4f})', color='green')
    
plt.title('Comparação de Ajustes de MMQ - População Brasil (2000-2024)')
plt.xlabel('Anos desde 2000 (t)')
plt.ylabel('População (em milhões)')
plt.legend()
plt.grid(True)
plt.show()
