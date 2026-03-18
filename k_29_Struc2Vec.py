# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs
 
# Struc2Vec

# GRASP + kvantna regresija, deterministički, strukturno
# Varijanta: Struc2Vec – strukturna uloga (stepen + okolina), šetnje po ulogama, Word2Vec

"""
Graphs in Space: Graph Embeddings for Machine Learning on Complex Data
Struc2Vec = strukturna sličnost (hijerarhija stepena); ovde uloga = (stepen, zbir stepena suseda), Word2Vec
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import networkx as nx

from itertools import combinations

from gensim.models import Word2Vec

from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Statevector, Pauli

CSV_PATH = "/data/loto7hh_4580_k21.csv"

df = pd.read_csv(CSV_PATH)
print()
print(df)
print()

SEED = 39
np.random.seed(SEED)
algorithm_globals.random_seed = SEED

EMBED_DIM = 3   # 3 dimenzije za embedding
MAX_EPOCHS = 20 # 20 epoha za fitanje
LR = 0.2        # 0.2 learning rate
FD_EPS = 1e-3   # 1e-3 finite difference epsilon

STRUC2VEC_WALK_LENGTH = 10 # 10 steps for walk
STRUC2VEC_NUM_WALKS = 80   # 80 walks for each node
STRUC2VEC_D_BINS = 8       # 8 bins for degree
STRUC2VEC_S_BINS = 8       # 8 bins for sum of neighbor degrees


def load_draws(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding="utf-8")
    expected_cols = [f"Num{i}" for i in range(1, 8)]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Nedostaje kolona {c} u CSV fajlu.")
    draws = []
    for _, row in df.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 8)]
        nums_sorted = sorted(nums)
        draws.append(nums_sorted)
    return draws


def compute_cooccurrence_matrix(draws):
    M = np.zeros((40, 40), dtype=np.int64)
    for draw in draws:
        for i_idx in range(len(draw)):
            for j_idx in range(i_idx + 1, len(draw)):
                a = draw[i_idx]
                b = draw[j_idx]
                M[a, b] += 1
                M[b, a] += 1
    return M


def compute_struc2vec_embeddings(M, k=EMBED_DIM):
    G = nx.Graph()
    for i in range(1, 40):
        G.add_node(i)
    for i in range(1, 40):
        for j in range(i + 1, 40):
            if M[i, j] > 0:
                G.add_edge(i, j, weight=float(M[i, j]))

    degree = dict(G.degree())
    sum_neighbor_deg = {}
    for i in range(1, 40):
        s = sum(degree[j] for j in G.neighbors(i))
        sum_neighbor_deg[i] = s
    d_max = max(degree.values()) or 1
    s_max = max(sum_neighbor_deg.values()) or 1
    role_str = {}
    for i in range(1, 40):
        d_bin = min(degree[i] * STRUC2VEC_D_BINS // (d_max + 1), STRUC2VEC_D_BINS - 1)
        s_bin = min(sum_neighbor_deg[i] * STRUC2VEC_S_BINS // (s_max + 1), STRUC2VEC_S_BINS - 1)
        role_str[i] = f"{d_bin}_{s_bin}"

    walks = []
    nodes = list(range(1, 40))
    for _ in range(STRUC2VEC_NUM_WALKS):
        for start in nodes:
            walk = [start]
            curr = start
            for _ in range(STRUC2VEC_WALK_LENGTH - 1):
                neighbors = list(G.neighbors(curr))
                if not neighbors:
                    break
                wts = [G[curr][nb].get("weight", 1.0) for nb in neighbors]
                wts = np.array(wts, dtype=float)
                wts /= wts.sum()
                idx = np.random.choice(len(neighbors), p=wts)
                curr = neighbors[int(idx)]
                walk.append(curr)
            walks.append([role_str[x] for x in walk])

    model = Word2Vec(
        sentences=walks,
        vector_size=k,
        window=5,
        min_count=0,
        seed=SEED,
        workers=1,
        epochs=10,
    )

    emb = np.zeros((39, k), dtype=float)
    for i in range(1, 40):
        key = role_str[i]
        if key in model.wv:
            emb[i - 1] = model.wv[key]
        else:
            emb[i - 1] = np.zeros(k)

    for d in range(k):
        col = emb[:, d]
        min_v, max_v = col.min(), col.max()
        if max_v - min_v > 0:
            emb[:, d] = (col - min_v) / (max_v - min_v) * np.pi
        else:
            emb[:, d] = 0.0
    return emb


def structural_target_from_graph(M):
    degrees = M.sum(axis=1)
    deg_sub = degrees[1:40].astype(float)
    min_v = deg_sub.min()
    max_v = deg_sub.max()
    if max_v - min_v > 0:
        deg_sub = (deg_sub - min_v) / (max_v - min_v)
    else:
        deg_sub = np.zeros_like(deg_sub)
    return deg_sub


class QuantumRegressor:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        self.ansatz = TwoLocal(
            num_qubits=num_features,
            rotation_blocks="ry",
            entanglement_blocks="cz",
            reps=1,
            insert_barriers=False,
        )
        self.observable = Pauli("Z" * num_features)
        self.num_params = len(self.ansatz.parameters)
        self.theta = np.zeros(self.num_params, dtype=float)
        self.base_circuit = self.feature_map.compose(self.ansatz)

    def _predict_single(self, x_vec, theta_vec):
        param_bind = {}
        for p, val in zip(self.feature_map.parameters, x_vec):
            param_bind[p] = float(val)
        for p, val in zip(self.ansatz.parameters, theta_vec):
            param_bind[p] = float(val)
        bound = self.base_circuit.assign_parameters(param_bind, inplace=False)
        sv = Statevector.from_instruction(bound)
        exp = np.real(sv.expectation_value(self.observable))
        n = self.num_features
        norm_exp = (exp + n) / (2.0 * n)
        return float(norm_exp)

    def predict(self, X):
        preds = [self._predict_single(x, self.theta) for x in X]
        return np.array(preds, dtype=float)

    def _loss(self, theta_vec, X, y):
        preds = [self._predict_single(x, theta_vec) for x in X]
        preds = np.array(preds, dtype=float)
        diff = preds - y
        return float(np.mean(diff * diff))

    def fit(self, X, y, epochs=MAX_EPOCHS, lr=LR, fd_eps=FD_EPS):
        theta = self.theta.copy()
        for _ in range(epochs):
            grad = np.zeros_like(theta)
            for j in range(len(theta)):
                orig = theta[j]
                theta[j] = orig + fd_eps
                loss_plus = self._loss(theta, X, y)
                theta[j] = orig - fd_eps
                loss_minus = self._loss(theta, X, y)
                theta[j] = orig
                grad[j] = (loss_plus - loss_minus) / (2.0 * fd_eps)
            theta = theta - lr * grad
        self.theta = theta


def greedy_best_combo(pred_scores, M):
    order = sorted(range(1, 40), key=lambda i: pred_scores[i], reverse=True)
    chosen = [order[0]]
    while len(chosen) < 7:
        best_candidate = None
        best_value = None
        for cand in order:
            if cand in chosen:
                continue
            value = pred_scores[cand]
            for c in chosen:
                value += M[cand, c]
            if best_value is None or value > best_value:
                best_value = value
                best_candidate = cand
        chosen.append(best_candidate)
    chosen.sort()
    return tuple(chosen)


def main():
    draws = load_draws()
    M = compute_cooccurrence_matrix(draws)
    emb = compute_struc2vec_embeddings(M, k=EMBED_DIM)

    x_train = emb
    y_train = structural_target_from_graph(M)

    qreg = QuantumRegressor(num_features=EMBED_DIM)
    qreg.fit(x_train, y_train)

    y_pred = qreg.predict(x_train)
    pred_scores = {i: float(y_pred[i - 1]) for i in range(1, 40)}
    best_combo = greedy_best_combo(pred_scores, M)

    print()
    print("Predikcija (Struc2Vec + kvantna regresija, deterministički, strukturno):")
    print(best_combo)
    print()
    print("Score:", pred_scores[best_combo[0]])
    print()
    """
    Predikcija (Struc2Vec + kvantna regresija, deterministički, strukturno):
    (1, 2, x, y, z, 26, 34)

    Score: 0.4879825338249706
    """


if __name__ == "__main__":
    main()

"""
Struc2Vec:

Strukturna uloga za čvor i: 
(d_bin, s_bin), gde je d_bin iz stepena čvora, 
s_bin iz zbira stepena suseda 
(oba diskretizovana u 8 binova).

Random šetnje 
(težinske, dužina 10, 80 šetnji po čvoru, seed=39); 
u šetnji se beleže uloge (string "d_bin_s_bin").

Word2Vec nad tim sekvencama (seed=SEED, workers=1).

Embedding čvora i = embedding njegove strukturalne uloge; 
ako uloga nije u rečniku, koristi se nula. 
Zatim normalizacija u [0, π].
"""
