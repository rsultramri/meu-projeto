"""
===============================================================================
  ADALINE E A REGRA DELTA  -  script de apoio a aula
  Redes Neurais Artificiais  |  Prof. Robson de Souza Ultramari
===============================================================================

  Script Python unico e comentado, para acompanhar os slides da aula.
  Basta ter numpy e matplotlib instalados (ja vem nos dois no Colab), entendeu?

  COMO USAR
  ---------
  * Terminal / VS Code:  python adaline_regra_delta.py
    O script roda de cima para baixo e abre 6 janelas de grafico, uma de
    cada vez. Feche cada janela para o script seguir.

  * Google Colab, opcao A (rapido):
        from google.colab import files; files.upload()   # envie este .py
        !python adaline_regra_delta.py

  * Google Colab, opcao B (recomendado para dar aula):
    cole cada SECAO numerada abaixo em uma celula separada e execute na
    ordem, de cima para baixo. Os graficos aparecem inline e a turma
    acompanha um bloco de cada vez.

  MAPA DAS SECOES x SLIDES
  ---------------------------------------------------------------------------
    1  O neuronio linear: onde o erro e medido ................. slides 4-5
    2  A regra delta deduzida (e conferida no codigo) .......... slides 7-10
    3  A classe Adaline ........................................ slides 27-28
    4  Epoca a epoca, na mao ................................... slides 20-21
    5  Treinamento completo da porta AND ....................... slide  22
    6  Fronteira de decisao e curva do EQM ..................... slides 23, 30
    7  A superficie de erro e uma parabola ..................... slides 13-14
    8  O efeito da taxa de aprendizado ......................... slide  15
    9  Adaline x Perceptron lado a lado ........................ slides 11, 16
   10  Dados ruidosos: onde o perceptron nunca pararia ......... slide  31
   11  O limite do Adaline: o XOR .............................. slides 34-35
   12  Bonus: o filtro adaptativo LMS .......................... slide  33
   13  Exercicios .............................................. slide  35

  ATENCAO: as secoes compartilham variaveis (X, d, a classe Adaline, o
  modelo treinado...). Execute-as SEMPRE na ordem; pular a secao 3, por
  exemplo, quebra todas as seguintes.
===============================================================================
"""



# ===========================================================================
#  SECAO 0 - PREPARACAO DO AMBIENTE
#  (bibliotecas e configuracao)
# ===========================================================================
#  numpy  -> vetores e o produto escalar w.x
#  pyplot -> os graficos: fronteira, curva do EQM, superficie de erro
# ===========================================================================


# numpy  -> vetores e operacoes matriciais (o produto escalar w.x)
# pyplot -> graficos (fronteira, curva do EQM, superficie de erro)
import numpy as np
import matplotlib.pyplot as plt

# Fixamos a semente do gerador aleatorio para que todos os alunos
# obtenham EXATAMENTE os mesmos numeros nos exemplos com ruido.
np.random.seed(42)

# Deixa os prints de arrays mais legiveis (3 casas decimais, sem notacao cientifica)
np.set_printoptions(precision=3, suppress=True)

print("Ambiente pronto. NumPy", np.__version__)


# ===========================================================================
#  SECAO 1 - O NEURONIO LINEAR: ONDE O ERRO E MEDIDO
#  (slides 4 e 5)
# ===========================================================================
#  Entradas, pesos, vies e soma ponderada sao IDENTICOS aos do perceptron.
#  A diferenca inteira esta no ponto em que o erro e calculado:
#
#    perceptron:  e = d - y_chapeu,  com y_chapeu = sinal(u)  -> erro discreto
#    adaline   :  e = d - u,         ANTES da ativacao        -> erro continuo
#
#  O erro continuo diz nao so QUE o neuronio errou, mas POR QUANTO - e e
#  justamente isso que permite deriva-lo.
# ===========================================================================


# Porta AND com saidas BIPOLARES: d pertence a {-1, +1}  (slide 18)
# Bipolar em vez de 0/1 porque o alvo simetrico centraliza a solucao.
X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])
d = np.array([-1.0, -1.0, -1.0, +1.0])

# Pesos quaisquer, so para ilustrar o calculo (ainda NAO estamos treinando)
w = np.array([0.4, 0.3])
b = -0.2

print("amostra    d     u = w.x + b   sinal(u)   e_perceptron   e_adaline")
print("-" * 68)
for xi, di in zip(X, d):
    u = np.dot(w, xi) + b        # soma ponderada: a saida LINEAR do neuronio
    y_chapeu = 1 if u >= 0 else -1   # ativacao degrau: so serve para CLASSIFICAR
    e_perc = di - y_chapeu       # erro DEPOIS da ativacao  -> discreto
    e_adal = di - u              # erro ANTES  da ativacao  -> continuo
    print(f"{xi.astype(int)}   {di:+.0f}     {u:+7.3f}      {y_chapeu:+d}"
          f"          {e_perc:+.0f}          {e_adal:+7.3f}")

print("\nRepare: o erro do perceptron so assume -2, 0 ou +2 (aqui, com d bipolar);")
print("o erro do Adaline mede a DISTANCIA ate o alvo. E essa distancia e derivavel.")


# ===========================================================================
#  SECAO 2 - A REGRA DELTA DEDUZIDA - E CONFERIDA NO CODIGO
#  (slides 7 a 10)
# ===========================================================================
#  Custo de uma amostra:   E = (1/2) * e^2,   e = d - u,   u = soma(w_i x_i) + b
#
#  Pela regra da cadeia (tres elos):
#
#      dE/dw_i = (dE/de) * (de/du) * (du/dw_i) = (e) * (-1) * (x_i) = -e * x_i
#
#  Substituindo no passo do gradiente descendente  w <- w - eta * dE/dw,
#  os dois sinais negativos se cancelam e nasce a REGRA DELTA:
#
#      w_i <- w_i + eta * e * x_i          b <- b + eta * e
#
#  O codigo abaixo CONFERE a deducao numericamente: calcula a derivada pela
#  definicao (diferencas finitas) e compara com a formula -e*x. Se a conta do
#  quadro estiver certa, os dois valores tem de bater.
# ===========================================================================


def custo(w, b, x, d_alvo):
    '''Custo E = 1/2 * e^2 de UMA amostra, para os pesos (w, b) dados.'''
    u = np.dot(w, x) + b
    e = d_alvo - u
    return 0.5 * e ** 2

# Ponto de trabalho: pesos atuais e uma amostra qualquer do conjunto
w = np.array([0.4, 0.3])
b = -0.2
x = X[3]          # a amostra (1, 1)
d_alvo = d[3]     # cujo alvo e +1

# --- (a) derivada NUMERICA: E(w + h) - E(w - h) sobre 2h -------------------
h = 1e-6
grad_num = np.zeros(2)
for i in range(2):
    w_mais, w_menos = w.copy(), w.copy()
    w_mais[i] += h
    w_menos[i] -= h
    grad_num[i] = (custo(w_mais, b, x, d_alvo) - custo(w_menos, b, x, d_alvo)) / (2 * h)

# --- (b) derivada ANALITICA: a formula deduzida no slide 9 ----------------
u = np.dot(w, x) + b
e = d_alvo - u
grad_analitico = -e * x        # dE/dw_i = -e * x_i

print("gradiente numerico  (diferencas finitas):", grad_num)
print("gradiente analitico (-e * x)            :", grad_analitico)
print("diferenca                               :", np.abs(grad_num - grad_analitico))
print("\nOs valores coincidem: a deducao da regra da cadeia esta correta.")

# --- (c) e o passo do gradiente e a propria regra delta -------------------
eta = 0.1
w_novo_gradiente = w - eta * grad_analitico   # w <- w - eta * dE/dw
w_novo_regradelta = w + eta * e * x           # w <- w + eta * e * x
print("\npasso pelo gradiente :", w_novo_gradiente)
print("passo pela regra delta:", w_novo_regradelta, " <- sao a MESMA coisa")


# ===========================================================================
#  SECAO 3 - A CLASSE ADALINE
#  (slides 27 e 28)
# ===========================================================================
#  Tres pontos que merecem atencao na hora de explicar para a turma:
#
#    1) net_input devolve u  -> e o que o TREINAMENTO usa;
#    2) prever aplica o sinal -> e o que a CLASSIFICACAO usa, nunca o treino;
#    3) o criterio de parada nao e mais "epoca sem erro" (que deixou de
#       existir), e sim a ESTABILIZACAO do EQM: |EQM_atual - EQM_anterior| <= eps.
# ===========================================================================


class Adaline:

    def __init__(self, n_entradas, eta=0.1, epocas=200, eps=1e-6):
        # w: um peso por entrada. Comecar em zero e legitimo aqui, porque a
        #    superficie de erro do Adaline tem um unico minimo (slide 13).
        self.w = np.zeros(n_entradas)
        self.b = 0.0            # vies (bias)
        self.eta = eta          # taxa de aprendizado: o tamanho do passo morro abaixo
        self.epocas = epocas    # teto de epocas, para o laco nunca ser infinito
        self.eps = eps          # precisao exigida do EQM para declarar convergencia

    # ---------------------------------------------------------------
    def net_input(self, x):
        '''u = w.x + b  -> a SAIDA LINEAR. E sobre ela que o Adaline treina.'''
        return np.dot(self.w, x) + self.b

    def prever(self, x):
        '''Aplica a funcao sinal a u. Usada SO para classificar, nunca no treino.'''
        return 1 if self.net_input(x) >= 0 else -1

    # ---------------------------------------------------------------
    def eqm(self, X, d):
        '''Erro Quadratico Medio do conjunto inteiro: EQM = (1/p) * soma(e_k^2).'''
        u = np.array([self.net_input(xi) for xi in X])   # saida linear de cada amostra
        e = d - u                                        # vetor de erros continuos
        return np.mean(e ** 2)

    # ---------------------------------------------------------------
    def treinar(self, X, d, verbose=True):
        # Guardamos o EQM ANTES de qualquer ajuste: e o ponto de partida da curva.
        self.historico = [self.eqm(X, d)]
        # Guardamos tambem o caminho percorrido pelos pesos, para desenhar a
        # trajetoria sobre a superficie de erro na secao 7.
        self.trajetoria = [np.concatenate([self.w, [self.b]])]

        for epoca in range(self.epocas):

            # ---- uma EPOCA = uma passada por todas as amostras ----
            for xi, di in zip(X, d):
                u = self.net_input(xi)          # 1) soma ponderada
                e = di - u                      # 2) erro LINEAR (contInuo!)
                self.w = self.w + self.eta * e * xi   # 3) regra delta nos pesos
                self.b = self.b + self.eta * e        # 4) regra delta no vies
                # Observe: NAO ha "if errou". O ajuste acontece em TODA amostra,
                # inclusive naquelas que o sinal ja classifica corretamente.

            # ---- fim da epoca: medimos a qualidade dos pesos atuais ----
            self.historico.append(self.eqm(X, d))
            self.trajetoria.append(np.concatenate([self.w, [self.b]]))

            if verbose and (epoca + 1) % 20 == 0:
                print(f"Epoca {epoca + 1:4d} | EQM: {self.historico[-1]:.6f}")

            # ---- criterio de parada: o EQM praticamente parou de cair ----
            if abs(self.historico[-2] - self.historico[-1]) <= self.eps:
                return epoca + 1        # convergiu

        return self.epocas              # esgotou o limite de epocas

print("Classe Adaline definida.")


# ===========================================================================
#  SECAO 4 - EPOCA A EPOCA, NA MAO
#  (slides 20 e 21)
# ===========================================================================
#  Antes de deixar o laco rodar sozinho, vale reproduzir no codigo a mesma
#  tabela preenchida no quadro. Os numeros abaixo batem EXATAMENTE com os
#  do slide: EQM de 1,14 na primeira epoca e 1,02 na segunda.
# ===========================================================================


w = np.zeros(2)     # pesos zerados
b = 0.0
eta = 0.1           # taxa de aprendizado da aula

for epoca in range(1, 3):
    print(f"\n{'=' * 70}\nEPOCA {epoca}\n{'=' * 70}")
    print(f"{'amostra':<10}{'u':>9}{'e = d - u':>12}   pesos apos o ajuste")
    print("-" * 70)
    print(f"{'inicio':<10}{'':>9}{'':>12}   w = {w}  b = {b:+.3f}")

    erros_da_epoca = []
    for xi, di in zip(X, d):
        u = np.dot(w, xi) + b       # soma ponderada com os pesos ATUAIS
        e = di - u                  # erro continuo
        erros_da_epoca.append(e)

        w = w + eta * e * xi        # regra delta
        b = b + eta * e

        print(f"{str(xi.astype(int)):<10}{u:>9.3f}{e:>12.3f}   w = {w}  b = {b:+.3f}")

    eqm_epoca = np.mean(np.array(erros_da_epoca) ** 2)
    print(f"\nmedia dos e^2 da epoca {epoca}: {eqm_epoca:.3f}")

print("\nDuas leituras para a turma:")
print(" 1) o erro nunca e 'redondo' - ele mede o quanto faltou;")
print(" 2) as tres primeiras amostras JA estao classificadas certo pelo sinal,")
print("    e mesmo assim geram ajuste. O Adaline persegue o menor erro medio,")
print("    nao o primeiro acerto.")


# ===========================================================================
#  SECAO 5 - TREINAMENTO COMPLETO DA PORTA AND
#  (slide 22)
# ===========================================================================
#  Deixando rodar, os pesos caminham para w = [1, 1] e b = -1,5 - exatamente
#  os pesos "magicos" que na aula do perceptron foram entregues prontos.
#  Aqui o algoritmo os ENCONTRA SOZINHO.
#
#  O EQM minimo e 0,25 e nao diminui mais: zerar o erro medio e impossivel
#  (a reta nao passa pelos quatro alvos), mas o SINAL de u acerta as quatro.
# ===========================================================================


ada = Adaline(n_entradas=2, eta=0.1, epocas=500, eps=1e-6)
n_epocas = ada.treinar(X, d, verbose=False)

print(f"Parou na epoca {n_epocas}")
print(f"w = {np.round(ada.w, 3)}   b = {round(ada.b, 3)}")
print(f"EQM final = {ada.historico[-1]:.4f}")

print("\nTeste do modelo treinado (agora sim usamos prever, com o sinal):")
for xi, di in zip(X, d):
    u = ada.net_input(xi)
    print(f"  {xi.astype(int)} -> u = {u:+.3f} | sinal = {ada.prever(xi):+d} | esperado = {di:+.0f}")

# ------------------------------------------------------------------
# CONFERINDO COM A SOLUCAO EXATA
# ------------------------------------------------------------------
# O minimo do erro quadratico de um modelo linear tem solucao fechada
# (minimos quadrados). np.linalg.lstsq calcula esse otimo diretamente.
# E o "fundo da tigela" que a regra delta procura passo a passo.
A = np.hstack([X, np.ones((4, 1))])          # matriz com uma coluna de 1s para o vies
solucao_exata, *_ = np.linalg.lstsq(A, d, rcond=None)
print(f"\nSolucao exata (minimos quadrados): w = {solucao_exata[:2]}  b = {solucao_exata[2]:+.3f}")
print(f"EQM nessa solucao: {np.mean((d - A @ solucao_exata) ** 2):.4f}")
print("\nA regra delta chega perto sem nunca ter resolvido um sistema:")
print("ela apenas desceu a superficie de erro, um passinho por amostra.")


# ===========================================================================
#  SECAO 6 - FRONTEIRA DE DECISAO E CURVA DO EQM
#  (slides 23 e 30)
# ===========================================================================
#  A esquerda, a reta w1*x1 + w2*x2 + b = 0 com as regioes de decisao
#  coloridas. A direita, a assinatura do Adaline: uma queda suave ate um
#  patamar. No perceptron o grafico equivalente e uma escada de numeros
#  inteiros de erros, que chega a zero ou nao chega nunca.
# ===========================================================================


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ---------------- esquerda: a fronteira de decisao ----------------
# Fundo colorido: para cada ponto do plano, de que lado o sinal de u o coloca.
malha_x, malha_y = np.meshgrid(np.linspace(-0.6, 1.6, 300),
                               np.linspace(-0.6, 1.6, 300))
regiao = np.sign(ada.w[0] * malha_x + ada.w[1] * malha_y + ada.b)
ax1.contourf(malha_x, malha_y, regiao, levels=[-2, 0, 2],
             colors=["#dbe4f5", "#f8dede"])

for xi, di in zip(X, d):
    marcador, cor = ("X", "crimson") if di == 1 else ("o", "navy")
    ax1.scatter(xi[0], xi[1], marker=marcador, s=220, c=cor, zorder=3,
                edgecolors="white", linewidths=1.5)

# A reta da fronteira sai de  w1*x1 + w2*x2 + b = 0,  isolando x2:
#     x2 = -(w1/w2) * x1 - b/w2
w1, w2 = ada.w
grade_x1 = np.linspace(-0.6, 1.6, 100)
ax1.plot(grade_x1, -(w1 / w2) * grade_x1 - ada.b / w2, "k--", lw=2,
         label="fronteira aprendida")

ax1.set_title("Fronteira de decisao (porta AND)")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2")
ax1.grid(alpha=0.3); ax1.legend(loc="upper right")
ax1.set_xlim(-0.6, 1.6); ax1.set_ylim(-0.6, 1.6)

# ---------------- direita: a curva do EQM ----------------
ax2.plot(ada.historico, lw=2, color="darkorange")
ax2.axhline(0.25, color="gray", ls=":", lw=2, label="EQM minimo teorico = 0,25")
ax2.set_title("Convergencia do EQM")
ax2.set_xlabel("epoca"); ax2.set_ylabel("EQM")
ax2.grid(alpha=0.3); ax2.legend()

plt.tight_layout(); plt.show()

print("A curva cai rapido no inicio (o gradiente e grande longe do minimo)")
print("e vai achatando ao se aproximar do fundo do vale. Se ela OSCILAR ou")
print("SUBIR, a taxa de aprendizado esta grande demais - e o assunto da secao 8.")


# ===========================================================================
#  SECAO 7 - A SUPERFICIE DE ERRO E UMA PARABOLA
#  (slides 13 e 14)
# ===========================================================================
#  Como u e linear nos pesos e o custo e quadratico, E(w) e um paraboloide:
#  uma tigela perfeita, com UM UNICO MINIMO, e global.
#
#  Para desenhar em 3D precisamos de apenas dois parametros livres. Por isso
#  esta demonstracao FIXA o vies em b = -1,5 e deixa so w1 e w2 aprenderem -
#  assim a superficie e a trajetoria ficam no mesmo espaco.
# ===========================================================================


b_fixo = -1.5   # congelamos o vies para poder visualizar em 3D

def eqm_superficie(w1, w2):
    '''EQM do conjunto AND para um par (w1, w2), com b fixo.'''
    u = X @ np.array([w1, w2]) + b_fixo
    return np.mean((d - u) ** 2)

# --- malha de pesos: cada ponto (w1, w2) e um MODELO possivel ---
eixo = np.linspace(-1.0, 3.0, 120)
W1, W2 = np.meshgrid(eixo, eixo)
Z = np.array([[eqm_superficie(a, c) for a in eixo] for c in eixo])

# --- trajetoria: repetimos o treino ajustando so w1 e w2 ---
w_tr = np.zeros(2)
caminho = [w_tr.copy()]
for epoca in range(60):
    for xi, di in zip(X, d):
        e = di - (np.dot(w_tr, xi) + b_fixo)
        w_tr = w_tr + 0.1 * e * xi      # regra delta, so nos pesos
    caminho.append(w_tr.copy())
caminho = np.array(caminho)

fig = plt.figure(figsize=(13, 5))

# ---------------- esquerda: o paraboloide em 3D ----------------
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.6, linewidth=0)
# O "+ 0.15" so levanta a trajetoria um tico acima da superficie, para nao ficar
# escondida dentro dela.
ax1.plot(caminho[:, 0], caminho[:, 1],
         [eqm_superficie(a, c) + 0.15 for a, c in caminho],
         "r.-", lw=2, ms=4, label="trajetoria dos pesos")
ax1.set_xlabel("w1"); ax1.set_ylabel("w2"); ax1.set_zlabel("EQM")
ax1.set_title("A superficie de erro e uma tigela")
ax1.view_init(elev=28, azim=-60)
ax1.legend()

# ---------------- direita: as curvas de nivel, vistas de cima ----------------
ax2 = fig.add_subplot(1, 2, 2)
niveis = ax2.contour(W1, W2, Z, levels=25, cmap="viridis")
ax2.clabel(niveis, inline=True, fontsize=7, fmt="%.1f")
ax2.plot(caminho[:, 0], caminho[:, 1], "r.-", lw=1.8, ms=5)
ax2.plot(caminho[0, 0], caminho[0, 1], "ks", ms=9, label="inicio (0, 0)")
ax2.plot(caminho[-1, 0], caminho[-1, 1], "r*", ms=18, label="fundo do vale")
ax2.set_xlabel("w1"); ax2.set_ylabel("w2")
ax2.set_title("Curvas de nivel: elipses em torno do minimo")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("Cada ponto do plano (w1, w2) e um modelo possivel; a altura e o quanto ele erra.")
print("O gradiente e perpendicular a curva de nivel e aponta MORRO ACIMA;")
print("a regra delta anda exatamente ao contrario dele.")
print("Como a tigela tem um so fundo, o ponto de partida nao importa - teste trocar")
print("w_tr = np.zeros(2) por np.array([2.5, -0.8]) e rode de novo.")


# ===========================================================================
#  SECAO 8 - O EFEITO DA TAXA DE APRENDIZADO
#  (slide 15)
# ===========================================================================
#  eta pequeno: passos curtos, descida suave, porem lenta.
#  eta grande : o passo atravessa o vale e cai do outro lado - o custo oscila
#               e pode DIVERGIR.
#
#  O eixo vertical esta em escala logaritmica justamente para caber a
#  divergencia no mesmo grafico das curvas bem-comportadas.
# ===========================================================================


plt.figure(figsize=(9, 5))

for eta_teste, cor in zip([0.001, 0.01, 0.1, 0.5, 1.0],
                          ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"]):
    # eps negativo desliga a parada antecipada: queremos as 60 epocas de todas as taxas
    modelo = Adaline(n_entradas=2, eta=eta_teste, epocas=60, eps=-1.0)
    modelo.treinar(X, d, verbose=False)
    plt.plot(modelo.historico, lw=2, color=cor, label=f"eta = {eta_teste}")

plt.axhline(0.25, color="gray", ls=":", label="EQM minimo = 0,25")
plt.yscale("log")                       # escala log: a divergencia cabe no grafico
plt.xlabel("epoca"); plt.ylabel("EQM (escala log)")
plt.title("O mesmo problema, cinco taxas de aprendizado")
plt.legend(); plt.grid(alpha=0.3)
plt.show()

print("Leitura do grafico:")
print("  eta = 0,001  -> em 60 epocas mal saiu do lugar (EQM ~ 0,90). Correto e lento.")
print("  eta = 0,01   -> descida suave, mas ainda a meio caminho (EQM ~ 0,56).")
print("  eta = 0,1    -> o melhor equilibrio para este problema (o valor da aula).")
print("  eta = 0,5    -> ja passa do ponto: estaciona ACIMA do minimo, oscilando.")
print("  eta = 1,0    -> diverge: cada passo atravessa o vale e sobe mais do outro lado.")
print("\nFaixa usual na pratica: 0,001 <= eta <= 0,1.")


# ===========================================================================
#  SECAO 9 - ADALINE X PERCEPTRON LADO A LADO
#  (slides 11 e 16)
# ===========================================================================
#  As duas regras tem a mesma forma; a diferenca inteira esta no erro usado.
#
#                  perceptron            adaline
#      ---------------------------------------------------
#      erro        e = d - y_chapeu      e = d - u
#      valores     {-1, 0, +1}           continuo (R)
#      ajusta      so quando erra        em toda amostra
#      origem      heuristica            gradiente de E
#      converge    se separavel          sempre (eta pequeno)
#
#  O perceptron PARA na primeira reta que separa as classes - qualquer uma
#  serve. O Adaline continua ajustando e escolhe a reta de menor EQM.
# ===========================================================================


class Perceptron:
    '''Mesma estrutura do Adaline; muda o erro e o criterio de parada.'''

    def __init__(self, n_entradas, eta=0.1, epocas=200):
        self.w = np.zeros(n_entradas)
        self.b = 0.0
        self.eta, self.epocas = eta, epocas

    def prever(self, x):
        return 1 if np.dot(self.w, x) + self.b >= 0 else -1

    def treinar(self, X, d):
        self.ajustes = 0                     # quantas vezes mexeu nos pesos
        for epoca in range(self.epocas):
            houve_erro = False
            for xi, di in zip(X, d):
                y_chapeu = self.prever(xi)   # erro DEPOIS da ativacao
                e = di - y_chapeu
                if e != 0:                   # <<< so ajusta QUANDO ERRA
                    self.w = self.w + self.eta * e * xi
                    self.b = self.b + self.eta * e
                    self.ajustes += 1
                    houve_erro = True
            if not houve_erro:               # epoca inteira sem erro -> acabou
                return epoca + 1
        return self.epocas


perc = Perceptron(n_entradas=2, eta=0.1)
n_perc = perc.treinar(X, d)

# Contamos os ajustes do Adaline: 4 por epoca, porque ele ajusta em TODA amostra
ada_cmp = Adaline(n_entradas=2, eta=0.1, epocas=500, eps=1e-6)
n_ada = ada_cmp.treinar(X, d, verbose=False)

print(f"PERCEPTRON  parou na epoca {n_perc:3d} | ajustes feitos: {perc.ajustes:5d}"
      f" | w = {np.round(perc.w, 3)} b = {perc.b:+.3f}")
print(f"ADALINE     parou na epoca {n_ada:3d} | ajustes feitos: {n_ada * 4:5d}"
      f" | w = {np.round(ada_cmp.w, 3)} b = {ada_cmp.b:+.3f}")

# --------- as duas fronteiras no mesmo plano ---------
plt.figure(figsize=(6, 5.5))
for xi, di in zip(X, d):
    marcador, cor = ("X", "crimson") if di == 1 else ("o", "navy")
    plt.scatter(xi[0], xi[1], marker=marcador, s=220, c=cor, zorder=3,
                edgecolors="white", linewidths=1.5)

grade = np.linspace(-0.6, 1.6, 100)
plt.plot(grade, -(perc.w[0] / perc.w[1]) * grade - perc.b / perc.w[1],
         "g-", lw=2, label="perceptron: a primeira reta que separou")
plt.plot(grade, -(ada_cmp.w[0] / ada_cmp.w[1]) * grade - ada_cmp.b / ada_cmp.w[1],
         "k--", lw=2, label="adaline: a reta de menor EQM")
plt.xlim(-0.6, 1.6); plt.ylim(-0.6, 1.6)
plt.title("Duas solucoes validas, uma delas mais centrada")
plt.xlabel("x1"); plt.ylabel("x2")
plt.legend(loc="upper right", fontsize=8); plt.grid(alpha=0.3)
plt.show()

print("\nAs duas retas classificam as 4 amostras corretamente.")
print("A do perceptron passa raspando na amostra que provocou o ultimo ajuste;")
print("a do Adaline se acomoda no meio do caminho - e por isso aguenta melhor o ruido,")
print("que e exatamente o teste da proxima secao.")


# ===========================================================================
#  SECAO 10 - DADOS RUIDOSOS: ONDE O PERCEPTRON NUNCA PARARIA
#  (slide 31)
# ===========================================================================
#  Dois grupos de pontos COM SOBREPOSICAO: nao existe reta que separe tudo.
#  O perceptron ficaria ajustando para sempre (nunca haveria uma epoca sem
#  erro). O Adaline converge para a reta de menor EQM e PARA SOZINHO.
# ===========================================================================


rng = np.random.default_rng(0)                      # semente fixa: todos veem o mesmo
grupo_A = rng.normal([0.0, 0.0], 0.5, (30, 2))      # 30 pontos em torno de (0, 0)
grupo_B = rng.normal([1.5, 1.5], 0.5, (30, 2))      # 30 pontos em torno de (1,5; 1,5)

X_r = np.vstack([grupo_A, grupo_B])                 # 60 amostras, 2 atributos
d_r = np.array([-1.0] * 30 + [+1.0] * 30)           # 30 alvos -1, 30 alvos +1

# eta menor porque agora sao 60 amostras por epoca: muito mais ajustes por passada
ada_r = Adaline(n_entradas=2, eta=0.01, epocas=300, eps=1e-6)
n_r = ada_r.treinar(X_r, d_r, verbose=False)

acertos = sum(ada_r.prever(xi) == di for xi, di in zip(X_r, d_r))
print(f"Parou na epoca {n_r} | acertos: {acertos}/60 | EQM final: {ada_r.historico[-1]:.4f}")
print(f"w = {np.round(ada_r.w, 3)}  b = {ada_r.b:+.3f}")

# --------- e o perceptron, no mesmo conjunto? ---------
perc_r = Perceptron(n_entradas=2, eta=0.01, epocas=300)
n_perc_r = perc_r.treinar(X_r, d_r)
acertos_p = sum(perc_r.prever(xi) == di for xi, di in zip(X_r, d_r))
print(f"\nPerceptron: usou as {n_perc_r} epocas ate o LIMITE (nunca teve epoca sem erro)")
print(f"            e fez {perc_r.ajustes} ajustes | acertos: {acertos_p}/60")

# --------- grafico ---------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.scatter(grupo_A[:, 0], grupo_A[:, 1], marker="o", c="navy", s=45,
            alpha=0.7, label="classe -1")
ax1.scatter(grupo_B[:, 0], grupo_B[:, 1], marker="X", c="crimson", s=45,
            alpha=0.7, label="classe +1")
grade = np.linspace(-1.8, 3.2, 100)
ax1.plot(grade, -(ada_r.w[0] / ada_r.w[1]) * grade - ada_r.b / ada_r.w[1],
         "k--", lw=2, label="fronteira do Adaline")
ax1.set_title("Classes com sobreposicao")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2.plot(ada_r.historico, lw=2, color="darkorange")
ax2.set_title("EQM: cai e estabiliza, mesmo sem separacao perfeita")
ax2.set_xlabel("epoca"); ax2.set_ylabel("EQM")
ax2.grid(alpha=0.3)

plt.tight_layout(); plt.show()

print("\nMoral: com ruido, 'errar zero' deixa de ser um objetivo alcancavel.")
print("O Adaline troca esse objetivo impossivel por outro sempre atingivel -")
print("o menor erro medio - e por isso ele SEMPRE para.")


# ===========================================================================
#  SECAO 11 - O LIMITE DO ADALINE: O XOR
#  (slides 34 e 35)
# ===========================================================================
#  No XOR nao existe reta que separe as classes, e o Adaline nao faz milagre:
#  o melhor ajuste linear possivel e "nao decidir nada". Os pesos vao a zero
#  e o EQM estabiliza em 1,0.
#
#  E esse impasse que a MADALINE contorna combinando dois Adalines - e que o
#  BACKPROPAGATION, na proxima aula, resolve de forma geral.
# ===========================================================================


d_xor = np.array([-1.0, +1.0, +1.0, -1.0])     # 0^0=-1, 0^1=+1, 1^0=+1, 1^1=-1

ada_xor = Adaline(n_entradas=2, eta=0.1, epocas=500, eps=1e-6)
n_xor = ada_xor.treinar(X, d_xor, verbose=False)

print(f"Parou na epoca {n_xor}")
print(f"w = {np.round(ada_xor.w, 3)}   b = {ada_xor.b:+.3f}")
print(f"EQM final = {ada_xor.historico[-1]:.4f}   (o minimo teorico aqui e 1,0)")

acertos_xor = sum(ada_xor.prever(xi) == di for xi, di in zip(X, d_xor))
print(f"\nAcertos do sinal: {acertos_xor}/4")
for xi, di in zip(X, d_xor):
    print(f"  {xi.astype(int)} -> u = {ada_xor.net_input(xi):+.3f} | "
          f"sinal = {ada_xor.prever(xi):+d} | esperado = {di:+.0f}")

print("\nInterpretacao: os pesos tendem a zero e u fica proximo de zero em todas as")
print("amostras. Traduzindo a 'decisao' do modelo: como nenhuma reta ajuda, o melhor")
print("ajuste quadratico e ficar em cima do muro. O EQM em 1,0 e o preco disso.")
print("O algoritmo CONVERGIU (parou) sem RESOLVER - sao coisas diferentes.")
print("\nExperimento: troque eps por 1e-9 e epocas por 5000. Os pesos vao a zero")
print("de verdade e o EQM encosta em 1,0000 - o teto teorico de qualquer reta no XOR.")


# ===========================================================================
#  SECAO 12 - BONUS: O FILTRO ADAPTATIVO LMS
#  (slide 33)
# ===========================================================================
#  A regra delta nasceu na engenharia de sinais com o nome de LMS (Least Mean
#  Squares). Um filtro LMS aprende em tempo real a prever um ruido indesejado
#  e subtrai-lo - e o ERRO E A PROPRIA SAIDA LIMPA.
#
#  E o principio do cancelamento de eco em modems e do cancelamento de ruido
#  em fones. Foi uma das primeiras aplicacoes comerciais de redes neurais,
#  decadas antes do deep learning. E o codigo e literalmente a mesma regra
#  delta desta aula.
# ===========================================================================


rng = np.random.default_rng(1)
N = 800
t = np.arange(N)

sinal_limpo = np.sin(2 * np.pi * t / 40)        # o que queremos ouvir
ruido_ref = rng.normal(0, 1, N)                 # ruido captado pelo microfone externo

# O ruido que chega ao microfone principal e uma VERSAO FILTRADA da referencia
# (ele percorreu outro caminho: paredes, distancia, atraso).
h_verdadeiro = np.array([0.8, -0.5, 0.3])
ruido_contaminante = np.convolve(ruido_ref, h_verdadeiro)[:N]

sinal_sujo = sinal_limpo + ruido_contaminante   # o que o microfone principal capta

# --- O filtro adaptativo: um Adaline com M entradas -------------------
# As entradas nao sao atributos de um objeto, e sim as M ultimas amostras
# do ruido de referencia. O alvo d e o proprio sinal sujo.
M = 3                       # numero de coeficientes (as "entradas" do neuronio)
w_lms = np.zeros(M)
eta_lms = 0.02
saida_limpa = np.zeros(N)

for k in range(M, N):
    x = ruido_ref[k - M + 1:k + 1][::-1]   # as M ultimas amostras da referencia
    u = np.dot(w_lms, x)                  # estimativa do ruido contaminante
    e = sinal_sujo[k] - u                 # <<< o ERRO E A SAIDA LIMPA
    w_lms = w_lms + eta_lms * e * x       # regra delta - identica a da aula
    saida_limpa[k] = e

print("coeficientes aprendidos:", np.round(w_lms, 3))
print("coeficientes verdadeiros:", h_verdadeiro)
print("\nO filtro descobriu sozinho como o ruido chegou ate o microfone.")

potencia_antes = np.mean((sinal_sujo[-200:] - sinal_limpo[-200:]) ** 2)
potencia_depois = np.mean((saida_limpa[-200:] - sinal_limpo[-200:]) ** 2)
print(f"\nruido residual antes:  {potencia_antes:.4f}")
print(f"ruido residual depois: {potencia_depois:.4f}"
      f"   ({potencia_antes / potencia_depois:.0f}x menor)")

fig, eixos = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
eixos[0].plot(t, sinal_limpo, color="green", lw=1.2)
eixos[0].set_title("1. Sinal limpo (o que queremos recuperar)")
eixos[1].plot(t, sinal_sujo, color="crimson", lw=0.8)
eixos[1].set_title("2. Sinal captado pelo microfone (limpo + ruido)")
eixos[2].plot(t, saida_limpa, color="navy", lw=1.2)
eixos[2].plot(t, sinal_limpo, color="green", lw=1.0, ls="--", alpha=0.7)
eixos[2].set_title("3. Saida do filtro LMS (= o erro!) sobreposta ao sinal limpo")
eixos[2].set_xlabel("amostra (tempo)")
for eixo in eixos:
    eixo.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("Repare no inicio do grafico 3: o filtro ainda esta aprendendo.")
print("Depois de algumas centenas de amostras ele ja rastreia o sinal limpo.")
print("Isto e um Adaline em producao - e roda ha decadas dentro de modems e fones.")


# ===========================================================================
#  SECAO 13 - EXERCICIOS
#  (slide 35)
# ===========================================================================
#  Todas as classes ja estao definidas acima - basta trocar os dados e os
#  parametros. Descomente e complete.
#
#   1. Treine a porta OR (saidas -1, +1, +1, +1) e compare os pesos finais
#      com os da porta AND.
#   2. Rode a porta AND com eta = 0,5 e depois eta = 0,001. O que acontece
#      com a curva do EQM em cada caso? (a secao 8 ja mostra o grafico)
#   3. Treine o XOR (saidas -1, +1, +1, -1). O algoritmo para? Em que valor
#      o EQM estabiliza? Interprete os pesos finais. (a secao 11 responde)
#   4. No exercicio 3, use prever() para medir os acertos. Quantas amostras
#      o sinal classifica corretamente?
#   5. Normalize as entradas dos dados ruidosos (media 0, desvio 1) e compare
#      o numero de epocas ate a parada.
#   6. Discussao: a regra delta ajusta pesos mesmo quando a classificacao ja
#      esta certa. Desperdicio ou vantagem? E se a saida linear virar uma
#      sigmoide?
# ===========================================================================


# ------------------------------------------------------------------
# EXERCICIO 1 - porta OR
# ------------------------------------------------------------------
d_or = np.array([-1.0, +1.0, +1.0, +1.0])

# COMPLETE: crie um Adaline, treine com (X, d_or) e imprima w, b e o EQM final.
# Depois compare com a AND: qual dos dois mudou mais, os pesos ou o vies?
#
# ada_or = Adaline(n_entradas=2, eta=0.1, epocas=500, eps=1e-6)
# n_or = ada_or.treinar(X, d_or, verbose=False)
# print(...)


# ------------------------------------------------------------------
# EXERCICIO 5 - normalizando os dados ruidosos
# ------------------------------------------------------------------
# Normalizar = subtrair a media e dividir pelo desvio padrao, coluna a coluna.
X_norm = (X_r - X_r.mean(axis=0)) / X_r.std(axis=0)

# COMPLETE: treine um Adaline com X_norm e d_r usando os MESMOS parametros
# da secao 10 (eta=0.01, epocas=300) e compare o numero de epocas ate parar.
# Dica: com entradas de escalas parecidas, as curvas de nivel viram circulos
# em vez de elipses alongadas - e a descida fica mais direta.


# ==========================================================================
#  O QUE FICA DESTA AULA
# ==========================================================================
#  * O Adaline e um NEURONIO LINEAR: treina sobre u = w.x + b, e o erro
#    e = d - u e continuo.
#  * A regra delta NAO e heuristica: e gradiente descendente sobre o erro
#    quadratico, deduzida pela regra da cadeia - a secao 2 conferiu isso.
#  * A superficie de erro e um paraboloide com um unico minimo global;
#    a taxa de aprendizado controla a descida.
#  * O treinamento para quando o EQM ESTABILIZA - e converge mesmo quando
#    as classes nao sao separaveis.
#  * Regra delta + neuronio DIFERENCIAVEL = BACKPROPAGATION: proxima aula.
# ==========================================================================

