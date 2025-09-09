# arbiter_puf_coco_crack.py
# Requirements: numpy, scikit-learn
# pip install numpy scikit-learn

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Utilities / Feature transform ----------
def challenge_to_phi(challenge):
    c = np.asarray(challenge).astype(int)
    n = c.size
    x = 1 - 2*c  # map 0->+1, 1->-1
    phi = np.empty(n + 1, dtype=int)
    for i in range(n):
        if i == n-1:
            phi[i] = 1
        else:
            phi[i] = np.prod(x[i+1:])
    phi[n] = 1
    return phi.astype(int)

def batch_phi(challenges):
    return np.vstack([challenge_to_phi(c) for c in challenges])

# ---------- Arbiter PUF simulator ----------
class ArbiterPUF:
    def __init__(self, n_bits, sigma=1.0, seed=None):
        self.n = n_bits
        rng = np.random.RandomState(seed)
        self.w = rng.normal(loc=0.0, scale=sigma, size=(n_bits + 1,))

    def response(self, challenge):
        phi = challenge_to_phi(challenge)
        val = np.dot(self.w, phi)
        return 1 if val >= 0 else 0

    def batch_response(self, challenges):
        phis = batch_phi(challenges)
        vals = phis.dot(self.w)
        return (vals >= 0).astype(int)

# ---------- COCO-like PUF composition ----------
class COCOPUF:
    def __init__(self, n_bits, k_apufs=4, mode='xor', sigma=1.0, seed=None):
        self.n = n_bits
        self.k = k_apufs
        self.mode = mode
        rng = np.random.RandomState(seed)
        self.apufs = [ArbiterPUF(n_bits, sigma=sigma, seed=rng.randint(1<<30)) for _ in range(k_apufs)]

    def response(self, challenge):
        bits = np.array([apuf.response(challenge) for apuf in self.apufs], dtype=int)
        if self.mode == 'xor':
            return bits.sum() % 2
        elif self.mode == 'majority':
            return 1 if bits.sum() > (len(bits)/2) else 0
        elif self.mode == 'concat':
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            return val
        else:
            raise ValueError("Unknown mode")

    def batch_response(self, challenges):
        bits_matrix = np.vstack([apuf.batch_response(challenges) for apuf in self.apufs]).T
        if self.mode == 'xor':
            return bits_matrix.sum(axis=1) % 2
        elif self.mode == 'majority':
            return (bits_matrix.sum(axis=1) > (self.k/2)).astype(int)
        elif self.mode == 'concat':
            vals = []
            for row in bits_matrix:
                v = 0
                for b in row:
                    v = (v << 1) | int(b)
                vals.append(v)
            return np.array(vals, dtype=int)

# ---------- Dataset generation ----------
def gen_random_challenges(num, n_bits, seed=None):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(num, n_bits))

def build_dataset(puf, num_crps=20000, n_bits=64, seed=0):
    challenges = gen_random_challenges(num_crps, n_bits, seed=seed)
    responses = puf.batch_response(challenges)
    return challenges, responses

# ---------- ML training / evaluation ----------
def train_and_eval(challenges, responses, test_size=0.2, random_state=0):
    X = batch_phi(challenges)
    y = responses
    X, y = shuffle(X, y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results = {}

    svc = LinearSVC(max_iter=20000)
    svc_grid = {'C': [0.01, 0.1, 1, 10]}
    svc_search = GridSearchCV(svc, svc_grid, cv=3, scoring='accuracy', n_jobs=-1)
    svc_search.fit(X_train, y_train)
    svc_best = svc_search.best_estimator_
    y_pred_svc = svc_best.predict(X_test)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    results['LinearSVC'] = {'best_params': svc_search.best_params_, 'accuracy': acc_svc,
                           'report': classification_report(y_test, y_pred_svc, digits=4),
                           'confusion': confusion_matrix(y_test, y_pred_svc)}

    log = LogisticRegression(max_iter=20000, solver='saga')
    log_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2', 'l1']}
    log_search = GridSearchCV(log, log_grid, cv=3, scoring='accuracy', n_jobs=-1)
    log_search.fit(X_train, y_train)
    log_best = log_search.best_estimator_
    y_pred_log = log_best.predict(X_test)
    acc_log = accuracy_score(y_test, y_pred_log)
    results['LogisticRegression'] = {'best_params': log_search.best_params_, 'accuracy': acc_log,
                                     'report': classification_report(y_test, y_pred_log, digits=4),
                                     'confusion': confusion_matrix(y_test, y_pred_log)}
    return results

# ---------- Example ----------
if __name__ == "__main__":
    n_bits = 64
    k_apufs = 4
    mode = 'xor'
    sigma = 1.0
    seed = 42

    coco = COCOPUF(n_bits=n_bits, k_apufs=k_apufs, mode=mode, sigma=sigma, seed=seed)

    NUM_CRPS = 25000
    print(f"Generating {NUM_CRPS} CRPs, n_bits={n_bits}, COCO mode={mode}, k={k_apufs} ...")
    challenges, responses = build_dataset(coco, num_crps=NUM_CRPS, n_bits=n_bits, seed=seed)

    if mode == 'concat':
        print("Note: 'concat' mode yields multi-bit responses; converting to parity for binary classification.")
        responses = (responses % 2).astype(int)

    print("Training LinearSVC and LogisticRegression...")
    results = train_and_eval(challenges, responses, test_size=0.2, random_state=seed)

    for model_name, info in results.items():
        print("----", model_name, "----")
        print("Best params:", info['best_params'])
        print("Accuracy:", info['accuracy'])
        print("Classification report:\n", info['report'])
        print("Confusion matrix:\n", info['confusion'])
