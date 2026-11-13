# -*- coding: utf-8 -*-

# %% [markdown]
# # Redes de Hopfield — implementação do zero em Python
#
# **Disciplina:** Redes Neurais  
# **Tema:** memória associativa com a rede de Hopfield discreta (neurônios bipolares ±1)
#
# Neste notebook você vai:
#
# 1. implementar a **regra de Hebb** para armazenar padrões na matriz de pesos **W**;
# 2. implementar a **recuperação assíncrona**, neurônio a neurônio, e acompanhar a **função de energia**;
# 3. reproduzir **exatamente** o exemplo numérico dos slides (N = 6);
# 4. recuperar letras 10 × 10 corrompidas por ruído;
# 5. observar **estados espúrios**, a diferença entre atualização **síncrona e assíncrona** e o limite de **capacidade** (≈ 0,138·N).
#
# **Referências:** Hopfield (1982, PNAS 79:2554); Haykin (2009, cap. 13) / Haykin (2001, cap. 14, ed. em português);
# Hertz, Krogh e Palmer (1991, cap. 2); Fausett (1994, cap. 3); Hinton — *Neural Networks for Machine Learning* (Coursera), aula 11.
#
# > Execute as células **em ordem**, de cima para baixo (menu *Ambiente de execução → Executar tudo*).

# %%
import numpy as np                  # NumPy: vetores, matrizes e produto escalar
import matplotlib.pyplot as plt     # Matplotlib: gráficos e visualização dos padrões
from matplotlib.colors import ListedColormap  # permite criar um mapa de cores com só duas cores

rng = np.random.default_rng(42)     # gerador de números aleatórios com semente fixa → resultados reproduzíveis
np.set_printoptions(linewidth=120)  # imprime matrizes largas sem quebrar a linha

# %% [markdown]
# ## 1. As funções da rede
#
# | Etapa | Fórmula |
# |---|---|
# | Armazenamento (Hebb) | $W = \frac{1}{N}\sum_{\mu=1}^{p} \xi^{\mu}\,(\xi^{\mu})^{T}$, com $w_{ii}=0$ |
# | Campo local do neurônio *i* | $h_i = \sum_{j} w_{ij}\, s_j$ |
# | Atualização | $s_i \leftarrow \operatorname{sgn}(h_i)$  (se $h_i = 0$, o neurônio mantém o estado) |
# | Energia | $E(s) = -\frac{1}{2}\sum_i\sum_j w_{ij}\, s_i\, s_j$ |

# %%
def treinar_hebb(padroes, normalizar=True):
    # Calcula a matriz de pesos W pela regra de Hebb (aprendizado em um único passo, sem iterações).
    # padroes: matriz p × N, uma linha por padrão, com valores +1 ou −1.
    P = np.asarray(padroes, dtype=float)   # converte a lista de padrões em matriz de números reais
    p, N = P.shape                         # p = quantidade de padrões; N = quantidade de neurônios
    W = P.T @ P                            # P.T @ P é igual à soma dos produtos externos Σ ξ·ξᵀ (matriz N × N)
    if normalizar:                         # o fator 1/N é o da formulação clássica (Hertz et al., 1991)
        W = W / N                          # dividir por N não muda o SINAL de h, só a escala dos pesos
    np.fill_diagonal(W, 0)                 # wᵢᵢ = 0: nenhum neurônio se conecta a si mesmo
    return W                               # W é simétrica (wᵢⱼ = wⱼᵢ): condição para a energia nunca subir

# %%
def energia(W, s):
    # Energia de Hopfield com limiar θ = 0:  E(s) = −½ · sᵀ W s
    return -0.5 * s @ W @ s                # s @ W @ s soma wᵢⱼ·sᵢ·sⱼ para todos os pares (i, j)


def sinal(h, estado_atual, tol=1e-12):
    # Função sinal com regra de desempate.
    # tol: margem para erros de arredondamento (com o fator 1/N, um "zero" pode virar 1e-17).
    if h > tol:                            # campo local positivo → neurônio vai para +1
        return 1
    if h < -tol:                           # campo local negativo → neurônio vai para −1
        return -1
    return estado_atual                    # h = 0 → mantém o estado atual (evita trocas sem motivo)

# %%
def recuperar_assincrono(W, sonda, max_varreduras=100, rng=None, guardar_energia=True):
    # Recuperação ASSÍNCRONA: um neurônio é atualizado por vez, sempre usando o estado mais recente.
    # Devolve: estado final, energia após cada atualização e número de varreduras executadas.
    # guardar_energia=False deixa a função bem mais rápida (útil nos experimentos com muitas repetições).
    s = np.array(sonda, dtype=int).copy()  # copia a sonda para não alterar o vetor original
    N = len(s)                             # número de neurônios
    energias = [energia(W, s)]             # lista que guarda a energia; começa com a energia da sonda
    for varredura in range(1, max_varreduras + 1):   # uma varredura = todos os N neurônios visitados uma vez
        mudou = False                      # indica se algum neurônio trocou de estado nesta varredura
        ordem = rng.permutation(N) if rng is not None else range(N)  # ordem aleatória (rng) ou fixa 0, 1, …, N−1
        for i in ordem:                    # percorre os neurônios na ordem escolhida
            h = W[i] @ s                   # campo local: hᵢ = Σⱼ wᵢⱼ·sⱼ (produto escalar da linha i com s)
            novo = sinal(h, s[i])          # aplica a função sinal ao campo local
            if novo != s[i]:               # o neurônio discorda do campo local?
                s[i] = novo                # atualiza SOMENTE o neurônio i
                mudou = True               # registra que houve mudança
            if guardar_energia:
                energias.append(energia(W, s))  # guarda a energia após esta atualização (nunca aumenta)
        if not mudou:                      # varredura inteira sem mudanças → ponto fixo (estado estável)
            break                          # encerra: a rede convergiu
    return s, energias, varredura          # estado recuperado, histórico de energia, nº de varreduras

# %%
def recuperar_sincrono(W, sonda, max_iter=20):
    # Recuperação SÍNCRONA: todos os neurônios são atualizados ao mesmo tempo, com base no estado anterior.
    s = np.array(sonda, dtype=int).copy()  # copia a sonda
    estados = [s.copy()]                   # histórico de estados, começando pela sonda
    for t in range(max_iter):              # repete até convergir, oscilar ou atingir o limite
        h = W @ s                          # calcula os campos locais de TODOS os neurônios de uma vez
        novo = np.where(h > 0, 1, np.where(h < 0, -1, s))  # sinal com desempate, aplicado ao vetor inteiro
        estados.append(novo.copy())        # guarda o novo estado
        if np.array_equal(novo, s):        # estado não mudou → ponto fixo
            return novo, estados, "ponto fixo"
        if len(estados) >= 3 and np.array_equal(novo, estados[-3]):  # voltou ao estado de dois passos atrás
            return novo, estados, "ciclo de período 2"               # oscilação entre dois estados
        s = novo                           # o novo estado passa a ser o estado atual
    return s, estados, "limite de iterações"

# %%
# Mapa de cores: −1 → cinza-claro, +1 → azul
CMAP = ListedColormap(["#F4F4F4", "#0F62FE"])

def mostrar(ax, s, titulo, lado=10):
    # Desenha um vetor de ±1 como imagem lado × lado.
    ax.imshow(np.asarray(s).reshape(lado, lado), cmap=CMAP, vmin=-1, vmax=1)  # remonta o vetor como grade
    ax.set_title(titulo, fontsize=10)      # título acima da imagem
    ax.set_xticks([]); ax.set_yticks([])   # remove os números dos eixos


def adicionar_ruido(padrao, fracao, rng):
    # Inverte o sinal de uma fração dos bits escolhidos ao acaso (ruído "sal e pimenta").
    s = np.array(padrao, dtype=int).copy() # copia o padrão
    n_trocas = int(round(fracao * len(s))) # quantos bits serão invertidos
    idx = rng.choice(len(s), n_trocas, replace=False)  # sorteia posições distintas
    s[idx] *= -1                           # inverte +1 ↔ −1 nessas posições
    return s


def sobreposicao(a, b):
    # Overlap m = (1/N)·Σ aᵢ·bᵢ : vale 1 se iguais, −1 se opostos, ≈ 0 se não correlacionados.
    return float(np.asarray(a) @ np.asarray(b)) / len(a)

# %% [markdown]
# ## 2. Exemplo dos slides (N = 6) — a conta feita à mão
#
# Dois padrões armazenados:  ξ¹ = [+1, +1, +1, −1, −1, −1]  e  ξ² = [+1, −1, +1, −1, +1, −1].
#
# Para as contas ficarem com números inteiros, usamos **W = ξ¹ξ¹ᵀ + ξ²ξ²ᵀ** sem o fator 1/N
# (o fator não altera o sinal de *h*, portanto não altera nenhuma decisão da rede).

# %%
xi1 = np.array([+1, +1, +1, -1, -1, -1])    # padrão 1
xi2 = np.array([+1, -1, +1, -1, +1, -1])    # padrão 2

print("ξ¹·ξ¹ᵀ =\n", np.outer(xi1, xi1))        # produto externo do padrão 1 (matriz 6 × 6)
print("\nξ²·ξ²ᵀ =\n", np.outer(xi2, xi2))      # produto externo do padrão 2
W6 = treinar_hebb([xi1, xi2], normalizar=False) # soma dos produtos externos com diagonal zerada
print("\nW (diagonal zerada) =\n", W6.astype(int))  # mostra a matriz de pesos com inteiros
print("\nW é simétrica?", np.array_equal(W6, W6.T))  # confere a simetria wᵢⱼ = wⱼᵢ

# %%
def recuperar_passo_a_passo(W, sonda):
    # Mesma lógica de recuperar_assincrono, mas imprimindo cada conta (ordem fixa 1, 2, …, N).
    s = np.array(sonda, dtype=int).copy()       # estado atual começa na sonda
    N = len(s)
    print(f"Estado inicial s = {s}   E = {energia(W, s):.0f}\n")
    varredura = 0
    while True:
        varredura += 1                          # conta as varreduras
        mudou = False
        print(f"--- Varredura {varredura} ---")
        for i in range(N):                      # neurônios 1 a N (índices 0 a N−1)
            termos = " + ".join(f"({int(W[i, j])})({s[j]:+d})" for j in range(N) if j != i)  # monta a soma
            h = W[i] @ s                        # campo local do neurônio i
            novo = sinal(h, s[i])               # decisão do neurônio
            acao = "INVERTE" if novo != s[i] else "mantém"   # texto explicativo
            E_antes = energia(W, s)             # energia antes da atualização
            s[i] = novo                         # aplica a atualização
            print(f"h{i+1} = {termos} = {h:+.0f}  →  sgn = {novo:+d}  ({acao})   E: {E_antes:.0f} → {energia(W, s):.0f}")
            mudou = mudou or acao == "INVERTE"  # registra se houve alguma inversão
        print()
        if not mudou:                           # nenhuma inversão na varredura → estado estável
            print(f"Nenhum neurônio mudou: estado estável s = {s}   E = {energia(W, s):.0f}")
            return s


sonda = np.array([+1, +1, -1, -1, -1, -1])      # ξ¹ com o neurônio 3 corrompido (+1 virou −1)
final = recuperar_passo_a_passo(W6, sonda)      # executa e imprime o passo a passo
print("Recuperou ξ¹?", np.array_equal(final, xi1))

# %% [markdown]
# ## 3. Memória associativa com letras 10 × 10 (N = 100 neurônios)
#
# Cada letra vira um vetor de 100 posições: `#` → +1 e `.` → −1.

# %%
LETRAS = {                                    # desenhos 10 × 10; cada string é uma linha da imagem
    "A": ["....##....", "...####...", "..##..##..", ".##....##.", ".##....##.",
          ".########.", ".########.", ".##....##.", ".##....##.", ".##....##."],
    "T": ["##########", "##########", "....##....", "....##....", "....##....",
          "....##....", "....##....", "....##....", "....##....", "....##...."],
    "X": ["##......##", ".##....##.", "..##..##..", "...####...", "....##....",
          "....##....", "...####...", "..##..##..", ".##....##.", "##......##"],
    "L": ["##........", "##........", "##........", "##........", "##........",
          "##........", "##........", "##........", "##########", "##########"],
    "C": ["..######..", ".########.", "##......##", "##........", "##........",
          "##........", "##........", "##......##", ".########.", "..######.."],
    "O": ["..######..", ".########.", "##......##", "##......##", "##......##",
          "##......##", "##......##", "##......##", ".########.", "..######.."],
}

def letra_para_vetor(linhas):
    # Converte o desenho em vetor bipolar: '#' → +1, '.' → −1.
    return np.array([1 if c == "#" else -1 for linha in linhas for c in linha])

V = {k: letra_para_vetor(v) for k, v in LETRAS.items()}  # dicionário letra → vetor de 100 posições

memorias = ["A", "T", "X", "L"]               # letras que serão armazenadas
P = np.array([V[k] for k in memorias])        # matriz p × N (4 × 100)
W = treinar_hebb(P)                           # treina a rede: um único cálculo, sem épocas

fig, axs = plt.subplots(1, len(memorias), figsize=(8, 2.2))
for ax, k in zip(axs, memorias):
    mostrar(ax, V[k], f"memória '{k}'")       # desenha cada padrão armazenado
plt.tight_layout(); plt.show()

print("Carga α = p/N =", len(memorias) / P.shape[1], " (limite teórico ≈ 0,138)")

# %%
fracao_ruido = 0.30                           # 30% dos pixels serão invertidos
fig, axs = plt.subplots(len(memorias), 3, figsize=(6, 2.1 * len(memorias)))
for linha, k in enumerate(memorias):
    ruidoso = adicionar_ruido(V[k], fracao_ruido, rng)             # cria a entrada corrompida
    recuperado, E_hist, n_varr = recuperar_assincrono(W, ruidoso, rng=rng)  # deixa a rede evoluir
    mostrar(axs[linha, 0], V[k], f"original '{k}'")
    mostrar(axs[linha, 1], ruidoso, f"{int(fracao_ruido*100)}% de ruído  m={sobreposicao(ruidoso, V[k]):.2f}")
    mostrar(axs[linha, 2], recuperado, f"recuperado  m={sobreposicao(recuperado, V[k]):.2f}")
plt.tight_layout(); plt.show()

# %%
# Energia ao longo das atualizações: a curva NUNCA sobe (teorema de convergência de Hopfield).
ruidoso = adicionar_ruido(V["X"], 0.30, rng)                  # letra X com 30% de ruído
recuperado, E_hist, n_varr = recuperar_assincrono(W, ruidoso, rng=rng)
print("Recuperou 'X'?", np.array_equal(recuperado, V["X"]), f"| E inicial = {E_hist[0]:.2f} | E final = {E_hist[-1]:.2f}")

plt.figure(figsize=(7, 3))
plt.plot(E_hist, color="#0F62FE")                              # energia após cada atualização
plt.axhline(energia(W, V["X"]), ls="--", color="#197838", label="E da memória 'X'")  # nível da memória
plt.xlabel("número de atualizações de neurônios"); plt.ylabel("energia E")
plt.title(f"Convergência em {n_varr} varredura(s)"); plt.legend(); plt.grid(alpha=0.3)
plt.show()

print("A energia aumentou em algum passo?", np.any(np.diff(E_hist) > 1e-12))  # deve imprimir False

# %%
# Recuperação a partir de informação PARCIAL: apagamos a metade inferior da letra (pixels = −1).
parcial = V["A"].copy().reshape(10, 10)       # remonta a letra como grade 10 × 10
parcial[5:, :] = -1                           # linhas 5 a 9 viram fundo (informação perdida)
parcial = parcial.ravel()                     # volta a ser vetor de 100 posições
recuperado, _, _ = recuperar_assincrono(W, parcial, rng=rng)

fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.4))
mostrar(axs[0], parcial, "metade apagada")
mostrar(axs[1], recuperado, "recuperado")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Atualização síncrona × assíncrona
#
# No modo **síncrono** todos os neurônios mudam juntos. A energia pode não diminuir e a rede pode
# **oscilar entre dois estados** (Bruck, 1990). No modo **assíncrono** a convergência é garantida
# quando W é simétrica e wᵢᵢ = 0.

# %%
W2 = treinar_hebb([[+1, +1]], normalizar=False)  # rede mínima: 2 neurônios que "querem" ter o mesmo sinal
sonda2 = np.array([+1, -1])                      # começa com os dois neurônios discordando

final_s, estados_s, motivo = recuperar_sincrono(W2, sonda2)
print("Síncrono :", [e.tolist() for e in estados_s], "→", motivo)   # alterna [+1,−1] ↔ [−1,+1]

final_a, _, _ = recuperar_assincrono(W2, sonda2)
print("Assíncrono:", final_a.tolist(), "→ ponto fixo")               # para em um estado estável

# %% [markdown]
# ## 5. Estados espúrios
#
# A rede também cria mínimos de energia que **não foram armazenados**:
#
# * **estados reversos** (−ξ): têm exatamente a mesma energia de ξ, pois E(−s) = E(s);
# * **estados de mistura**, como sgn(ξ¹ + ξ² + ξ³).

# %%
# (a) Estado reverso: −A também é estável
reverso = -V["A"]
final, _, _ = recuperar_assincrono(W, reverso, rng=rng)
print("−A é estável?", np.array_equal(final, reverso), f"| E(A) = {energia(W, V['A']):.2f} | E(−A) = {energia(W, reverso):.2f}")

# (b) Estados de mistura com padrões aleatórios (N = 200, p = 3)
Pr = rng.choice([-1, 1], size=(3, 200))       # três padrões aleatórios independentes
Wr = treinar_hebb(Pr)
mistura = np.sign(Pr.sum(axis=0)).astype(int) # sgn(ξ¹ + ξ² + ξ³): soma de 3 valores ±1 nunca é zero
final, _, _ = recuperar_assincrono(Wr, mistura, rng=rng)
print("Mistura é estável?", np.array_equal(final, mistura),
      "| overlaps com ξ¹, ξ², ξ³:", [round(sobreposicao(final, p), 2) for p in Pr])  # ≈ 0,5 com cada um

# %%
# (c) Para onde a rede das letras vai a partir de estados totalmente aleatórios?
contagem = {"memória": 0, "reverso": 0, "espúrio": 0}
exemplos_espurios = []                         # guarda alguns estados espúrios para desenhar
for tentativa in range(300):
    s0 = rng.choice([-1, 1], size=100)         # estado inicial aleatório
    s, _, _ = recuperar_assincrono(W, s0, rng=rng, guardar_energia=False)
    if any(np.array_equal(s, p) for p in P):
        contagem["memória"] += 1               # caiu em uma letra armazenada
    elif any(np.array_equal(s, -p) for p in P):
        contagem["reverso"] += 1               # caiu no negativo de uma letra
    else:
        contagem["espúrio"] += 1               # caiu em um mínimo que ninguém armazenou
        if len(exemplos_espurios) < 4:
            exemplos_espurios.append(s)
print(contagem)

if exemplos_espurios:
    fig, axs = plt.subplots(1, len(exemplos_espurios), figsize=(2 * len(exemplos_espurios), 2.2), squeeze=False)
    for ax, s in zip(axs[0], exemplos_espurios):
        mostrar(ax, s, "espúrio")
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 6. Capacidade de armazenamento
#
# Com padrões aleatórios, a recuperação quase perfeita se mantém até **α = p/N ≈ 0,138**
# (Amit, Gutfreund e Sompolinsky, 1985). Acima disso, a interferência entre memórias (*crosstalk*) domina.
# Em redes pequenas a transição é suave; aumente N para vê-la ficar mais abrupta.

# %%
N = 100                                        # número de neurônios (tente 200 ou 400: fica mais lento)
valores_p = range(2, 36, 2)                    # quantidades de padrões testadas
repeticoes = 10                                # redes sorteadas para cada p (média reduz a variação)
alfas, m_medio, perfeitos = [], [], []

for p in valores_p:
    ms, oks = [], []
    for r in range(repeticoes):
        Pc = rng.choice([-1, 1], size=(p, N))  # p padrões aleatórios
        Wc = treinar_hebb(Pc)
        for mu in range(min(p, 5)):            # testa até 5 padrões por rede (economiza tempo)
            s, _, _ = recuperar_assincrono(Wc, Pc[mu], rng=rng, guardar_energia=False)  # começa EXATAMENTE no padrão
            ms.append(sobreposicao(s, Pc[mu])) # quanto do padrão sobreviveu à dinâmica
            oks.append(np.array_equal(s, Pc[mu]))  # ficou 100% igual?
    alfas.append(p / N); m_medio.append(np.mean(ms)); perfeitos.append(np.mean(oks))

plt.figure(figsize=(7, 3.2))
plt.plot(alfas, m_medio, "o-", color="#0F62FE", label="overlap médio m")
plt.plot(alfas, perfeitos, "s-", color="#FF8321", label="fração recuperada sem erro")
plt.axvline(0.138, ls="--", color="#DA1E28", label="α ≈ 0,138")
plt.xlabel("carga α = p / N"); plt.ylim(-0.05, 1.05); plt.grid(alpha=0.3); plt.legend()
plt.title(f"Capacidade da rede de Hopfield (N = {N})"); plt.show()

# %% [markdown]
# ## 7. Padrões correlacionados se confundem
#
# A análise de capacidade supõe padrões **aleatórios e independentes**. Letras parecidas, como C e O,
# compartilham muitos pixels e interferem fortemente entre si.

# %%
print("overlap(C, O) =", sobreposicao(V["C"], V["O"]))   # próximo de 1 → padrões muito parecidos
print("overlap(A, T) =", sobreposicao(V["A"], V["T"]))   # próximo de 0 → padrões pouco parecidos

def taxa_recuperacao(memorias, ruido=0.20, tentativas=100):
    # Para cada letra armazenada, conta para onde a rede vai a partir de uma versão ruidosa.
    Pm = np.array([V[k] for k in memorias])              # matriz de padrões
    Wm = treinar_hebb(Pm)                                # treina a rede com essas letras
    for k in memorias:
        destino = {"a própria letra": 0, "outra letra": 0, "estado espúrio": 0}
        for t in range(tentativas):
            s0 = adicionar_ruido(V[k], ruido, rng)       # letra com ruído
            s, _, _ = recuperar_assincrono(Wm, s0, rng=rng, guardar_energia=False)
            if np.array_equal(s, V[k]):
                destino["a própria letra"] += 1          # recuperação correta
            elif any(np.array_equal(s, V[o]) for o in memorias):
                destino["outra letra"] += 1              # confundiu com outra memória
            else:
                destino["estado espúrio"] += 1           # parou em um mínimo não armazenado
        print(f"  '{k}': {destino}")

print("\nMemórias pouco correlacionadas (A, T, X, L):")
taxa_recuperacao(["A", "T", "X", "L"])
print("\nMemórias com C e O (muito correlacionadas):")
taxa_recuperacao(["A", "T", "C", "O"])

# %% [markdown]
# ## 8. Exercícios para entregar
#
# 1. **Ruído.** Para cada letra (A, T, X, L), varie o ruído de 10% a 50% (passo de 10%), com 20 repetições por nível.
#    Faça um gráfico da taxa de recuperação perfeita × ruído e explique por que a curva cai perto de 50%.
# 2. **Conta à mão.** Refaça no caderno o exemplo da Seção 2 partindo de ξ² com o neurônio 5 corrompido.
#    Confira cada hᵢ com a função `recuperar_passo_a_passo`.
# 3. **Energia.** Mostre algebricamente que, com W simétrica e wᵢᵢ = 0, a mudança de um único neurônio produz
#    ΔE = −Δsₖ·hₖ ≤ 0. Depois, remova a linha `np.fill_diagonal(W, 0)` e verifique no código se a garantia continua valendo.
# 4. **Capacidade.** Rode a Seção 6 com N = 50, 100 e 200. Compare as curvas e discuta o papel de N na nitidez da transição.
# 5. **Correlação.** Explique, usando o termo de *crosstalk* do campo local, por que C e O se confundem.
