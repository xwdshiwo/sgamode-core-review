from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SGAMODEConfig
from .gnn import AEMGNN, GuidanceState
from .graph import GraphSpec, build_graph_spec, minmax_norm


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a <= b) and np.any(a < b)


def non_dominated_sort(fitness: np.ndarray) -> list[list[int]]:
    n = len(fitness)
    if n == 0:
        return []
    dom_count = np.zeros(n, dtype=int)
    dom_set = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(fitness[i], fitness[j]):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif dominates(fitness[j], fitness[i]):
                dom_set[j].append(i)
                dom_count[i] += 1
    for i in range(n):
        if dom_count[i] == 0:
            fronts[0].append(i)

    current = 0
    while current < len(fronts) and fronts[current]:
        nxt: list[int] = []
        for i in fronts[current]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        if nxt:
            fronts.append(nxt)
        current += 1
    return fronts


def crowding_distance(front_fitness: np.ndarray) -> np.ndarray:
    n, m = front_fitness.shape
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    d = np.zeros(n, dtype=float)
    for obj in range(m):
        idx = np.argsort(front_fitness[:, obj])
        d[idx[0]] = np.inf
        d[idx[-1]] = np.inf
        fmin = front_fitness[idx[0], obj]
        fmax = front_fitness[idx[-1], obj]
        denom = fmax - fmin
        if denom <= 1e-12:
            continue
        for i in range(1, n - 1):
            left = front_fitness[idx[i - 1], obj]
            right = front_fitness[idx[i + 1], obj]
            d[idx[i]] += (right - left) / denom
    return d


def hypervolume_2d(fitness: np.ndarray, ref: tuple[float, float] = (1.0, 1.0)) -> float:
    if len(fitness) == 0:
        return 0.0
    nd = []
    for i in range(len(fitness)):
        dominated = False
        for j in range(len(fitness)):
            if i != j and dominates(fitness[j], fitness[i]):
                dominated = True
                break
        if not dominated:
            nd.append(fitness[i])
    f = np.array(nd, dtype=float)
    order = np.argsort(f[:, 0])
    f = f[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in f:
        if x >= ref[0]:
            continue
        y_eff = min(y, prev_y)
        if y_eff < ref[1]:
            hv += (ref[0] - x) * (prev_y - y_eff)
            prev_y = y_eff
    return float(hv)


class SubsetEvaluator:
    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: SGAMODEConfig):
        self.X = X
        self.y = y
        self.cfg = cfg
        self.cache: dict[bytes, tuple[float, float, int]] = {}
        self.rng = np.random.default_rng(cfg.random_state)

    def _key(self, mask: np.ndarray) -> bytes:
        return np.packbits(mask.astype(np.uint8)).tobytes()

    def evaluate(self, mask: np.ndarray) -> tuple[float, float, int]:
        key = self._key(mask)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        n_total = mask.shape[0]
        selected = np.flatnonzero(mask)
        ratio = float(len(selected)) / float(max(1, n_total))
        if len(selected) == 0:
            out = (1.0, ratio, min(self.cfg.knn_candidates))
            self.cache[key] = out
            return out

        Xs = self.X[:, selected]
        unique_labels, counts = np.unique(self.y, return_counts=True)
        n_classes = len(unique_labels)
        min_count = int(np.min(counts)) if n_classes > 0 else 0
        n_splits = min(int(self.cfg.inner_cv_folds), min_count)

        best_k = int(self.cfg.knn_candidates[0])
        best_acc = -np.inf
        for k in self.cfg.knn_candidates:
            if n_splits >= 2:
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.cfg.random_state,
                )
                fold_scores: list[float] = []
                for tr, va in splitter.split(Xs, self.y):
                    k_eff = min(int(k), max(1, len(tr) - 1))
                    model = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("knn", KNeighborsClassifier(n_neighbors=k_eff)),
                        ]
                    )
                    model.fit(Xs[tr], self.y[tr])
                    pred = model.predict(Xs[va])
                    fold_scores.append(accuracy_score(self.y[va], pred))
                acc = float(np.mean(fold_scores))
            else:
                k_eff = min(int(k), max(1, len(Xs) - 1))
                model = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("knn", KNeighborsClassifier(n_neighbors=k_eff)),
                    ]
                )
                model.fit(Xs, self.y)
                pred = model.predict(Xs)
                acc = float(accuracy_score(self.y, pred))
            if acc > best_acc:
                best_acc = acc
                best_k = int(k)

        out = (1.0 - best_acc, ratio, best_k)
        self.cache[key] = out
        return out


@dataclass
class SGAMODEResult:
    pareto_masks: np.ndarray
    pareto_fitness: np.ndarray
    best_mask: np.ndarray
    best_fitness: np.ndarray
    hv_history: list[float]
    best_error_history: list[float]
    execution_seconds: float


class SGAMODE:
    def __init__(self, config: SGAMODEConfig | None = None):
        self.config = config or SGAMODEConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        self.gnn = AEMGNN(self.config)
        self.result_: SGAMODEResult | None = None
        self.best_k_: int | None = None

    def decode(self, pop_z: np.ndarray) -> np.ndarray:
        if self.config.decode_with_sigmoid:
            scale = float(max(1e-6, self.config.decode_sigmoid_scale))
            logits = scale * (pop_z - self.config.decode_threshold)
            logits = np.clip(logits, -30.0, 30.0)
            prob = 1.0 / (1.0 + np.exp(-logits))
            masks = self.rng.random(size=prob.shape) < prob
        else:
            masks = pop_z > self.config.decode_threshold
        if masks.ndim == 1:
            if not np.any(masks):
                masks[np.argmax(pop_z)] = True
            return masks
        empty = np.where(~masks.any(axis=1))[0]
        for ridx in empty:
            masks[ridx, int(np.argmax(pop_z[ridx]))] = True
        return masks

    def _subset_weight(self, mask: np.ndarray, pool_importance: np.ndarray) -> float:
        sel = np.flatnonzero(mask)
        if len(sel) == 0:
            return 1e-12
        return float(np.sum(pool_importance[sel]) / (np.sqrt(len(sel)) + 1e-12))

    def _roulette_choice(self, candidates: np.ndarray, weights: np.ndarray) -> int:
        w = weights.astype(float).copy()
        w[w < 0] = 0.0
        s = w.sum()
        if s <= 0:
            return int(self.rng.choice(candidates))
        p = w / s
        return int(self.rng.choice(candidates, p=p))

    def _sample_f(self, mu_f: float) -> float:
        while True:
            f = mu_f + self.config.sigma_f * np.tan(np.pi * (self.rng.random() - 0.5))
            if f > 0:
                return float(min(f, 1.0))

    def _sample_cr(self, mu_cr: float) -> float:
        cr = self.rng.normal(mu_cr, self.config.sigma_cr)
        return float(np.clip(cr, 0.0, 1.0))

    def _semantic_diversity(self, emb: np.ndarray, selected: np.ndarray) -> float:
        if len(selected) <= 1:
            return 0.0
        if len(selected) > self.config.max_semantic_features:
            selected = self.rng.choice(selected, size=self.config.max_semantic_features, replace=False)
        E = emb[selected]
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E = E / norms
        sim = E @ E.T
        iu = np.triu_indices(sim.shape[0], k=1)
        if len(iu[0]) == 0:
            return 0.0
        return float(np.mean(1.0 - sim[iu]))

    def _enhanced_front_scores(
        self,
        front_indices: list[int],
        all_fitness: np.ndarray,
        masks: np.ndarray,
        guidance: GuidanceState,
    ) -> np.ndarray:
        front_fit = all_fitness[front_indices]
        d_obj = crowding_distance(front_fit)
        finite = np.isfinite(d_obj)
        d_obj_norm = np.zeros_like(d_obj)
        if np.any(finite):
            d_obj_norm[finite] = minmax_norm(d_obj[finite])
            d_obj_norm[~finite] = 1.0
        else:
            d_obj_norm[:] = 1.0

        cent_norm = minmax_norm(guidance.centrality)
        d_struct = np.zeros(len(front_indices), dtype=float)
        d_sem = np.zeros(len(front_indices), dtype=float)
        for idx_local, idx_global in enumerate(front_indices):
            sel = np.flatnonzero(masks[idx_global])
            if len(sel) > 0:
                d_struct[idx_local] = float(np.mean(cent_norm[sel]))
            d_sem[idx_local] = self._semantic_diversity(guidance.embeddings, sel)

        d_struct = minmax_norm(d_struct) if len(d_struct) > 1 else d_struct
        d_sem = minmax_norm(d_sem) if len(d_sem) > 1 else d_sem
        eps = 1e-12
        return (d_obj_norm + eps) * (1.0 + self.config.gamma_struct * d_struct) * (
            1.0 + self.config.gamma_sem * d_sem
        )

    def _environmental_selection(
        self,
        pop_z: np.ndarray,
        pop_x: np.ndarray,
        fitness: np.ndarray,
        guidance: GuidanceState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fronts = non_dominated_sort(fitness)
        selected_idx: list[int] = []
        for front in fronts:
            if len(selected_idx) + len(front) <= self.config.pop_size:
                selected_idx.extend(front)
                continue
            remain = self.config.pop_size - len(selected_idx)
            scores = self._enhanced_front_scores(front, fitness, pop_x, guidance)
            order = np.argsort(-scores)
            selected_idx.extend([front[i] for i in order[:remain]])
            break
        idx = np.array(selected_idx, dtype=int)
        return pop_z[idx], pop_x[idx], fitness[idx]

    def _evaluate_population(
        self,
        evaluator: SubsetEvaluator,
        pop_x: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        fitness = np.zeros((len(pop_x), 2), dtype=float)
        best_ks: list[int] = []
        for i, mask in enumerate(pop_x):
            err, ratio, best_k = evaluator.evaluate(mask)
            fitness[i] = [err, ratio]
            best_ks.append(best_k)
        return fitness, best_ks

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        graph_spec: GraphSpec | None = None,
        external_graph_files: list[str] | None = None,
    ) -> "SGAMODE":
        t0 = time.time()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        if graph_spec is None:
            graph_spec = build_graph_spec(
                X=X,
                feature_names=feature_names,
                external_graph_files=external_graph_files,
                corr_threshold=self.config.corr_threshold,
                corr_block_size=self.config.corr_block_size,
            )

        guidance = self.gnn.initialize(X, graph_spec)
        evaluator = SubsetEvaluator(X, y, self.config)

        pool_mask = np.zeros(n_features, dtype=bool)
        pool_mask[guidance.pool_indices] = True
        means = np.where(pool_mask, 0.72, 0.18)
        pop_z = self.rng.normal(loc=means, scale=0.20, size=(self.config.pop_size, n_features))
        pop_z = np.clip(pop_z, 0.0, 1.0)
        pop_x = self.decode(pop_z)
        fitness, pop_best_k = self._evaluate_population(evaluator, pop_x)

        mu_f = self.config.mu_f_init
        mu_cr = self.config.mu_cr_init

        hv_hist: list[float] = []
        best_err_hist: list[float] = []

        for gen in range(1, self.config.max_generations + 1):
            fronts = non_dominated_sort(fitness)
            rank = np.full(len(pop_z), 1_000_000, dtype=int)
            enhanced = np.zeros(len(pop_z), dtype=float)
            for ridx, front in enumerate(fronts):
                rank[np.array(front)] = ridx
                scores = self._enhanced_front_scores(front, fitness, pop_x, guidance)
                enhanced[np.array(front)] = scores

            order = np.lexsort((-enhanced, rank))
            top_count = max(2, int(np.ceil(self.config.p_best_rate * self.config.pop_size)))
            top_idx = order[:top_count]

            offspring_z = np.zeros_like(pop_z)
            offspring_x = np.zeros_like(pop_x)
            offspring_fit = np.zeros_like(fitness)
            successful_f: list[float] = []
            successful_cr: list[float] = []

            subset_weights = np.array([self._subset_weight(m, guidance.pool_importance) for m in pop_x])
            all_idx = np.arange(self.config.pop_size)
            for i in range(self.config.pop_size):
                Fi = self._sample_f(mu_f)
                CRi = self._sample_cr(mu_cr)

                pbest = self._roulette_choice(top_idx, subset_weights[top_idx])
                valid_r1 = all_idx[all_idx != i]
                r1 = self._roulette_choice(valid_r1, subset_weights[valid_r1])
                valid_r2 = all_idx[(all_idx != i) & (all_idx != r1)]
                r2 = int(self.rng.choice(valid_r2))

                mutant = pop_z[i] + Fi * (pop_z[pbest] - pop_z[i]) + Fi * (pop_z[r1] - pop_z[r2])
                mutant = np.clip(mutant, 0.0, 1.0)

                trial = pop_z[i].copy()
                j_rand = int(self.rng.integers(0, n_features))
                cross_mask = self.rng.random(n_features) < CRi
                cross_mask[j_rand] = True
                trial[cross_mask] = mutant[cross_mask]

                if self.config.sparsity_mutation_prob > 0:
                    pre_mask = trial > self.config.decode_threshold
                    sel = np.flatnonzero(pre_mask)
                    if len(sel) > 0:
                        imp = guidance.pool_importance[sel]
                        imp_n = minmax_norm(imp) if len(sel) > 1 else np.ones_like(imp)
                        drop_p = self.config.sparsity_mutation_prob * (1.0 - imp_n)
                        drop = self.rng.random(len(sel)) < drop_p
                        if np.any(drop):
                            to_drop = sel[drop]
                            trial[to_drop] = np.minimum(trial[to_drop], self.config.decode_threshold - 0.05)

                trial_x = self.decode(trial)
                err, ratio, k_best = evaluator.evaluate(trial_x)
                trial_fit = np.array([err, ratio], dtype=float)

                offspring_z[i] = trial
                offspring_x[i] = trial_x
                offspring_fit[i] = trial_fit

                if dominates(trial_fit, fitness[i]) or (
                    not dominates(fitness[i], trial_fit) and self.rng.random() < 0.5
                ):
                    successful_f.append(Fi)
                    successful_cr.append(CRi)
                    pop_best_k[i] = k_best

            if successful_f:
                sf = np.array(successful_f, dtype=float)
                scr = np.array(successful_cr, dtype=float)
                mu_f = (1.0 - self.config.jade_c) * mu_f + self.config.jade_c * (
                    np.sum(sf * sf) / (np.sum(sf) + 1e-12)
                )
                mu_cr = (1.0 - self.config.jade_c) * mu_cr + self.config.jade_c * np.mean(scr)

            comb_z = np.vstack([pop_z, offspring_z])
            comb_x = np.vstack([pop_x, offspring_x])
            comb_f = np.vstack([fitness, offspring_fit])
            pop_z, pop_x, fitness = self._environmental_selection(comb_z, comb_x, comb_f, guidance)

            fronts = non_dominated_sort(fitness)
            f0 = fitness[np.array(fronts[0])]
            hv_hist.append(hypervolume_2d(f0))
            best_err_hist.append(float(np.min(fitness[:, 0])))

            if self.config.feedback_interval > 0 and gen % self.config.feedback_interval == 0:
                nd_mask = pop_x[np.array(fronts[0])]
                freq = nd_mask.mean(axis=0) if len(nd_mask) else np.zeros(n_features, dtype=float)
                self.gnn.feedback_update(freq, guidance.relation_node_masks)
                guidance = self.gnn.refine(X, graph_spec)

            if self.config.verbose and (gen % 10 == 0 or gen == 1 or gen == self.config.max_generations):
                print(
                    f"[Gen {gen:03d}] best_error={best_err_hist[-1]:.4f}, "
                    f"hv={hv_hist[-1]:.4f}, mu_f={mu_f:.3f}, mu_cr={mu_cr:.3f}"
                )

        fronts = non_dominated_sort(fitness)
        front0_idx = np.array(fronts[0], dtype=int)
        pareto_masks = pop_x[front0_idx]
        pareto_fitness = fitness[front0_idx]

        if self.config.selection_strategy == "error_first":
            order = np.lexsort((pareto_fitness[:, 1], pareto_fitness[:, 0]))
            best_idx = int(order[0])
        elif self.config.selection_strategy == "sparse_within_tol":
            best_err = float(np.min(pareto_fitness[:, 0]))
            keep = np.where(pareto_fitness[:, 0] <= best_err + self.config.sparsity_error_tolerance)[0]
            if len(keep) == 0:
                keep = np.arange(len(pareto_fitness))
            rel = np.lexsort((pareto_fitness[keep, 0], pareto_fitness[keep, 1]))
            best_idx = int(keep[rel[0]])
        else:
            err_n = minmax_norm(pareto_fitness[:, 0]) if len(pareto_fitness) > 1 else pareto_fitness[:, 0]
            sz_n = minmax_norm(pareto_fitness[:, 1]) if len(pareto_fitness) > 1 else pareto_fitness[:, 1]
            dist = np.sqrt(err_n * err_n + sz_n * sz_n)
            best_idx = int(np.argmin(dist))
        best_mask = pareto_masks[best_idx].copy()
        best_fit = pareto_fitness[best_idx].copy()
        _, _, best_k = evaluator.evaluate(best_mask)
        self.best_k_ = best_k

        self.result_ = SGAMODEResult(
            pareto_masks=pareto_masks,
            pareto_fitness=pareto_fitness,
            best_mask=best_mask,
            best_fitness=best_fit,
            hv_history=hv_hist,
            best_error_history=best_err_hist,
            execution_seconds=time.time() - t0,
        )
        return self

    def get_selected_feature_indices(self) -> np.ndarray:
        if self.result_ is None:
            raise ValueError("Model has not been fitted.")
        return np.flatnonzero(self.result_.best_mask)

    def evaluate_test_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        if self.result_ is None:
            raise ValueError("Model has not been fitted.")
        selected = self.get_selected_feature_indices()
        if len(selected) == 0:
            return {
                "accuracy": 0.0,
                "f1_weighted": 0.0,
                "n_features": 0,
                "feature_ratio": 0.0,
                "best_k": self.best_k_,
            }
        k = int(self.best_k_ or 5)
        k = min(k, max(1, len(X_train) - 1))
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        model.fit(X_train[:, selected], y_train)
        pred = model.predict(X_test[:, selected])
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
            "n_features": int(len(selected)),
            "feature_ratio": float(len(selected) / X_train.shape[1]),
            "best_k": k,
        }

    @staticmethod
    def nested_cv_evaluate(
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        config: SGAMODEConfig,
        outer_folds: int = 5,
        outer_repeats: int = 10,
        seed: int = 42,
        external_graph_files: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        splitter = RepeatedStratifiedKFold(
            n_splits=outer_folds,
            n_repeats=outer_repeats,
            random_state=seed,
        )
        for fold_idx, (tr, te) in enumerate(splitter.split(X, y), start=1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]

            model = SGAMODE(config=config)
            graph_spec = build_graph_spec(
                X=Xtr,
                feature_names=feature_names,
                external_graph_files=external_graph_files,
                corr_threshold=config.corr_threshold,
                corr_block_size=config.corr_block_size,
            )
            model.fit(Xtr, ytr, feature_names=feature_names, graph_spec=graph_spec)
            test_res = model.evaluate_test_split(Xtr, ytr, Xte, yte)
            test_res["fold"] = fold_idx
            test_res["hv"] = float(model.result_.hv_history[-1] if model.result_ and model.result_.hv_history else 0.0)
            test_res["best_error_traincv"] = float(model.result_.best_fitness[0]) if model.result_ else 1.0
            results.append(test_res)
        return results
