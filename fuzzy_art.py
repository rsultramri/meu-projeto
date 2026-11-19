import numpy as np

class FuzzyART:
    """
    Fuzzy ART
    Entrada: vetores contínuos entre 0 e 1
    Objetivo: agrupamento sem rótulos com complemento de entrada
    """

    def __init__(self, n_features, vigilancia=0.75, beta=1.0, alpha=0.001):
        self.n     = n_features    # tamanho original do vetor
        self.rho   = vigilancia    # parâmetro de vigilância
        self.beta  = beta          # taxa de aprendizado
        self.alpha = alpha         # constante para evitar divisão por zero
        self.W     = []            # pesos (tamanho 2n após complemento)

    # ----------------------------------------------------------
    def _complementar(self, x):
        """Concatena x com (1 - x) → vetor de tamanho 2n"""
        return np.concatenate([x, 1 - x])

    # ----------------------------------------------------------
    def _ativacao(self, x_comp, j):
        """T(j) = |min(x_comp, w(j))| / (α + |w(j)|)"""
        minimo     = np.minimum(x_comp, self.W[j])   # AND fuzzy
        soma_min   = minimo.sum()                     # soma (não conta 1s!)
        soma_peso  = self.W[j].sum()
        return soma_min / (self.alpha + soma_peso)

    # ----------------------------------------------------------
    def _vigilancia_ok(self, x_comp, j):
        """|min(x_comp, w(j))| / |x_comp| >= ρ"""
        minimo   = np.minimum(x_comp, self.W[j])
        soma_min = minimo.sum()
        soma_x   = x_comp.sum()
        if soma_x == 0:
            return False
        return (soma_min / soma_x) >= self.rho

    # ----------------------------------------------------------
    def _atualizar_pesos(self, x_comp, j):
        """w(j) = β * min(x_comp, w(j)) + (1 - β) * w(j)"""
        minimo    = np.minimum(x_comp, self.W[j])
        self.W[j] = self.beta * minimo + (1 - self.beta) * self.W[j]

    # ----------------------------------------------------------
    def treinar(self, x):
        """Processa um vetor x e retorna a categoria atribuída"""
        x      = np.array(x, dtype=float)
        x_comp = self._complementar(x)

        if len(self.W) == 0:
            w_novo = np.ones(2 * self.n)
            self.W.append(w_novo)
            self._atualizar_pesos(x_comp, 0)
            return 0

        ativacoes = [self._ativacao(x_comp, j) for j in range(len(self.W))]
        ordem     = np.argsort(ativacoes)[::-1]

        for j in ordem:
            if self._vigilancia_ok(x_comp, j):
                # RESSONÂNCIA
                self._atualizar_pesos(x_comp, j)
                return int(j)

        # Nenhuma passou — cria nova categoria
        w_novo = np.ones(2 * self.n)
        self.W.append(w_novo)
        j_novo = len(self.W) - 1
        self._atualizar_pesos(x_comp, j_novo)
        return j_novo


# ============================================================
# EXEMPLO DE USO
# ============================================================
if __name__ == "__main__":

    X = np.array([
        [0.9, 0.1, 0.0, 0.1],   # grupo A
        [0.8, 0.2, 0.1, 0.0],   # grupo A
        [0.0, 0.9, 0.8, 0.9],   # grupo B
        [0.1, 0.8, 0.9, 0.8],   # grupo B
        [0.7, 0.1, 0.0, 0.2],   # grupo A
        [0.0, 0.7, 0.9, 0.7],   # grupo B
    ])

    fart = FuzzyART(n_features=4, vigilancia=0.6, beta=1.0)

    print("=== Fuzzy ART ===")
    for i, x in enumerate(X):
        cat = fart.treinar(x)
        print(f"  Exemplo {i+1}: {x} → Categoria {cat}")

    print(f"\nTotal de categorias criadas: {len(fart.W)}")