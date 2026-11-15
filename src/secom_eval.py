"""SECOM 공통 평가 도구 (secom_model.ipynb 셀 3에서 분리)

사용 예:
    from src.secom_eval import SecomEval, impute_steps, timed
    ev = SecomEval(X2, yb, y["timestamp"], dev=True)
    evaluate = ev.evaluate
    evaluate(pipe, "모델 이름", tune_threshold=True)

원칙: 데이터를 보고 fit하는 전처리는 전부 모델(Pipeline) 안에 둔다 → fold/학습 구간에서만 fit.
"""
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TunedThresholdClassifierCV

METRICS = ["BER", "recall", "PR_AUC", "F1"]


# ---------------------------------------------------------------- 전처리
class MissingFlagger(BaseEstimator, TransformerMixin):
    """학습 데이터에서 결측 비율 > threshold인 컬럼에만 결측 플래그(원래컬럼_na, 0/1)를 붙인다.
    플래그끼리 완전히 같은 열은 (학습 데이터 기준) 첫 번째 컬럼 것만 남긴다. 원래 컬럼은 그대로 통과 (NaN 유지)."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns.astype(str), dtype=object)
        self.n_features_in_ = X.shape[1]
        rate = X.isna().mean()
        flags = X.loc[:, rate > self.threshold].isna()
        self.flag_cols_ = list(flags.T.drop_duplicates().index)   # 같은 패턴은 첫 번째만
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = self.feature_names_in_
        flags = X[self.flag_cols_].isna().astype(float)
        flags.columns = [f"{c}_na" for c in self.flag_cols_]
        return pd.concat([X, flags], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + [f"{c}_na" for c in self.flag_cols_], dtype=object)


def impute_steps():
    """공통 전처리: 결측 플래그(50% 초과 컬럼) → 중앙값 대체"""
    return [("flag", MissingFlagger(threshold=0.5)),
            ("impute", SimpleImputer(strategy="median"))]


# ---------------------------------------------------------------- 분할
def make_cv(dev=True):
    """outer CV: DEV면 10-fold × 1, 아니면 10-fold × 5"""
    return RepeatedStratifiedKFold(n_splits=10, n_repeats=1 if dev else 5, random_state=42)


def make_inner_cv():
    """하이퍼파라미터 튜닝용 (outer 학습 fold 내부)"""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def time_split(timestamps, frac=0.8):
    """timestamp 기준 정렬 후 앞 frac / 뒤 (1-frac) 위치 인덱스"""
    order = np.argsort(np.asarray(timestamps), kind="stable")
    n_tr = int(len(order) * frac)
    return order[:n_tr], order[n_tr:]


# ---------------------------------------------------------------- 지표
def score(model, Xt, yt):
    """BER, 불량 recall, PR-AUC, F1 (불량 = 1 기준)"""
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    return {
        "BER": 1 - balanced_accuracy_score(yt, pred),
        "recall": recall_score(yt, pred, zero_division=0),
        "PR_AUC": average_precision_score(yt, prob),
        "F1": f1_score(yt, pred, zero_division=0),
    }


def unwrap(m):
    """TunedThresholdClassifierCV → estimator_, GridSearchCV → best_estimator_ 순으로 벗겨서 실제 모델 반환"""
    if hasattr(m, "best_threshold_"):
        m = m.estimator_
    if hasattr(m, "best_estimator_"):
        m = m.best_estimator_
    return m


def count_used(m):
    """모델이 실제로 쓰는 변수 수"""
    m = unwrap(m)
    if hasattr(m, "named_steps") and "select" in m.named_steps:
        return int(m.named_steps["select"].get_support().sum())
    est = m[-1] if hasattr(m, "steps") else m
    if hasattr(est, "coef_"):                      # 로지스틱: 0이 아닌 계수
        return int(np.count_nonzero(est.coef_))
    if hasattr(est, "booster_"):                   # LightGBM: split에 한 번이라도 쓰인 변수
        return int((est.booster_.feature_importance("split") > 0).sum())
    return int(est.n_features_in_)                 # 그 외: 입력 차원 (결측 플래그 포함)


@contextmanager
def timed(label=""):
    """셀/실험 시작·끝 시간 출력"""
    t0 = time.time()
    print(f"[시작] {label} {time.strftime('%H:%M:%S')}")
    try:
        yield
    finally:
        print(f"[끝] {label} {time.strftime('%H:%M:%S')} (소요 {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------- 평가
class SecomEval:
    """같은 outer CV + 같은 시간순 분할로 모델을 평가하고 results에 누적"""

    def __init__(self, X, y, timestamps, dev=True, frac=0.8):
        self.X, self.y = X, y
        self.dev = dev
        self.cv = make_cv(dev)
        self.inner_cv = make_inner_cv()
        self.idx_tr, self.idx_te = time_split(timestamps, frac)
        self.X_tr_t, self.X_te_t = X.iloc[self.idx_tr], X.iloc[self.idx_te]
        self.y_tr_t, self.y_te_t = y.iloc[self.idx_tr], y.iloc[self.idx_te]
        self.results = []

    def summary(self):
        print(f"DEV={self.dev}, outer folds = {self.cv.get_n_splits()}")
        print(f"시간순: 학습 {len(self.y_tr_t)} (불량 {self.y_tr_t.sum()}) / "
              f"테스트 {len(self.y_te_t)} (불량 {self.y_te_t.sum()})")

    def evaluate(self, model, name, tune_threshold=False):
        """(a) outer CV  (b) 시간순 80/20  (c) results에 한 행 추가. 반환: (fold별 학습 모델 리스트, 시간순 학습 모델)
        tune_threshold=True: TunedThresholdClassifierCV로 감싸 학습 데이터 내부 5-fold에서 BER 최소 임계값 선택"""
        X, y = self.X, self.y
        if tune_threshold:
            model = TunedThresholdClassifierCV(
                model, scoring="balanced_accuracy",
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0))

        fold_scores, fold_models = [], []
        for tr, te in self.cv.split(X, y):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            s = score(m, X.iloc[te], y.iloc[te])
            s["n_feat"] = count_used(m)
            fold_scores.append(s)
            fold_models.append(m)
        fs = pd.DataFrame(fold_scores)

        m_time = clone(model).fit(self.X_tr_t, self.y_tr_t)   # 전처리 포함 학습 구간에서만 fit
        st = score(m_time, self.X_te_t, self.y_te_t)

        row = {"model": name}
        for k in METRICS:
            row[f"{k}_mean"] = fs[k].mean()
            row[f"{k}_std"] = fs[k].std()
        for k in METRICS:
            row[f"{k}_time"] = st[k]
        row["n_feat"] = fs["n_feat"].mean()
        row["n_folds"] = len(fs)
        if tune_threshold:
            thr = [m.best_threshold_ for m in fold_models]
            row["thr_mean"], row["thr_std"], row["thr_time"] = np.mean(thr), np.std(thr), m_time.best_threshold_

        self.results[:] = [r for r in self.results if r["model"] != name] + [row]   # 재실행 시 같은 이름 행 교체

        print(f"[{name}]  ({len(fs)} folds)")
        print("  CV    : " + " | ".join(
            f"{k} {row[k + '_mean'] * 100:.1f}±{row[k + '_std'] * 100:.1f}" if k != "PR_AUC"
            else f"{k} {row[k + '_mean']:.3f}±{row[k + '_std']:.3f}" for k in METRICS))
        print("  시간순: " + " | ".join(
            f"{k} {st[k] * 100:.1f}" if k != "PR_AUC" else f"{k} {st[k]:.3f}" for k in METRICS))
        print(f"  사용 변수 수(fold 평균): {row['n_feat']:.1f}")
        if tune_threshold:
            print(f"  임계값 fold별: {np.round(thr, 3).tolist()}")
            print(f"  임계값 평균 {row['thr_mean']:.3f}±{row['thr_std']:.3f} | 시간순 {row['thr_time']:.3f}")
        return fold_models, m_time
