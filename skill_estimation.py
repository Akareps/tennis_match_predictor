"""
Estimate player serve & return skills using a regression that controls for
opponent strength, surface, and recency.

For each "player-match-row" with serve % spw against a given opponent on a
given surface, model:
    logit(spw) = serve_skill[player, surface]
               - return_skill[opponent, surface]
               + intercept[surface]

Equivalent additive form (Barnett-Clarke style on logit scale).

We fit this via weighted ridge regression (closed-form), with weights that
exponentially decay by match age. Surface is handled by a separate model per
surface (or a shared model with surface-specific player effects — we go
with separate models per surface for simplicity in v1).

Output: a SkillTable that gives, for any (player, surface, as_of_date), an
estimated serve_skill and return_skill on logit scale, plus the implied
spw vs an average opponent.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, eye as speye
from scipy.sparse.linalg import lsmr
from dataclasses import dataclass
from typing import Optional


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass
class SkillTable:
    """Holds estimated serve/return skills per player, on a given surface."""
    surface: str
    as_of: pd.Timestamp
    serve_skill: dict     # player_id -> logit-scale serve skill
    return_skill: dict    # player_id -> logit-scale return skill
    intercept: float      # logit(avg spw) on this surface
    avg_serve: float      # mean serve skill (for shrinkage / unknown players)
    avg_return: float     # mean return skill
    n_obs: dict           # player_id -> number of observations used (for shrinkage)

    def get_serve(self, player_id, min_obs: int = 5) -> float:
        """Return serve skill; falls back to mean for new/sparse players."""
        n = self.n_obs.get(player_id, 0)
        if n < min_obs:
            return self.avg_serve
        return self.serve_skill.get(player_id, self.avg_serve)

    def get_return(self, player_id, min_obs: int = 5) -> float:
        n = self.n_obs.get(player_id, 0)
        if n < min_obs:
            return self.avg_return
        return self.return_skill.get(player_id, self.avg_return)

    def predict_spw(self, server_id, returner_id) -> float:
        """P(server wins point) given server vs returner on this surface."""
        s = self.get_serve(server_id)
        r = self.get_return(returner_id)
        return _sigmoid(self.intercept + s - r)


def fit_skills(
    long_df: pd.DataFrame,
    surface: str,
    as_of: pd.Timestamp,
    half_life_days: float = 365.0,
    ridge: float = 1.0,
    max_lookback_days: int = 730,
) -> SkillTable:
    """
    Fit serve & return skills for all players on a given surface, using
    matches strictly before `as_of`.

    Args:
        long_df: long-format match data (one row per player-match)
        surface: which surface to fit (e.g., "Clay")
        as_of: estimation date — only uses matches before this date
        half_life_days: exponential decay half-life for match age
        ridge: ridge penalty (shrinks toward 0 = average player)
        max_lookback_days: hard cutoff on how far back to look
    """
    # Filter to surface and time window
    df = long_df[
        (long_df["surface"] == surface)
        & (long_df["date"] < as_of)
        & (long_df["date"] >= as_of - pd.Timedelta(days=max_lookback_days))
    ].copy()
    if len(df) < 100:
        raise ValueError(f"Too few matches ({len(df)}) for surface={surface}")

    # Recency weights: exp decay
    age_days = (as_of - df["date"]).dt.days.values
    weights = np.exp(-np.log(2) * age_days / half_life_days)

    # We weight each row by (svpt * recency) so matches with more points
    # contribute more.
    weights = weights * df["svpt"].values

    # Build design matrix.
    # Each row corresponds to one player serving in one match.
    # Target: logit(spw)
    # Features: dummy for player (serve effect) + dummy for opponent (return effect)
    players = sorted(set(df["player_id"]).union(set(df["opp_id"])))
    pid_to_idx = {p: i for i, p in enumerate(players)}
    n_p = len(players)

    rows = []
    cols = []
    vals = []
    y = []
    w = []

    for i, (_, r) in enumerate(df.iterrows()):
        # Serve skill of player_id
        rows.append(i); cols.append(pid_to_idx[r["player_id"]]); vals.append(1.0)
        # Return skill of opponent (negative coefficient)
        rows.append(i); cols.append(n_p + pid_to_idx[r["opp_id"]]); vals.append(-1.0)
        y.append(_logit(r["spw"]))
        w.append(weights[i])

    n_obs = len(y)
    n_features = 2 * n_p
    X = csr_matrix((vals, (rows, cols)), shape=(n_obs, n_features))
    y = np.array(y)
    w = np.array(w)

    # Apply weights: scale rows of X and y by sqrt(w)
    sqrt_w = np.sqrt(w)
    X_w = X.multiply(sqrt_w[:, None]).tocsr()
    y_w = y * sqrt_w

    # Add ridge penalty rows: sqrt(ridge) * I, target 0
    ridge_mat = np.sqrt(ridge) * speye(n_features, format="csr")
    X_aug = csr_matrix(np.vstack([X_w.toarray(), ridge_mat.toarray()]))
    y_aug = np.concatenate([y_w, np.zeros(n_features)])

    # Solve least squares
    result = lsmr(X_aug, y_aug, atol=1e-8, btol=1e-8, maxiter=2000)
    beta = result[0]

    serve_skill = {p: float(beta[pid_to_idx[p]]) for p in players}
    return_skill = {p: float(beta[n_p + pid_to_idx[p]]) for p in players}

    # Count observations per player (across both serving AND being served-to)
    n_obs_per = (
        df["player_id"].value_counts().to_dict()
    )
    # add return observations
    opp_counts = df["opp_id"].value_counts().to_dict()
    for pid, c in opp_counts.items():
        n_obs_per[pid] = n_obs_per.get(pid, 0) + c

    # Intercept: weighted mean of logit(spw) — captures surface-level effect
    intercept = float(np.average(y, weights=w))

    # Compute averages for fallback on unknown players
    avg_serve = float(np.mean(list(serve_skill.values())))
    avg_return = float(np.mean(list(return_skill.values())))

    return SkillTable(
        surface=surface,
        as_of=as_of,
        serve_skill=serve_skill,
        return_skill=return_skill,
        intercept=intercept,
        avg_serve=avg_serve,
        avg_return=avg_return,
        n_obs=n_obs_per,
    )


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_matches

    print("Generating synthetic data...")
    long, truth = generate_synthetic_matches(n_players=100, n_matches=5000)

    print("Fitting skill table for Hard, as of 2024-06-01...")
    st = fit_skills(long, surface="Hard", as_of=pd.Timestamp("2024-06-01"))
    print(f"  n players in skill table: {len(st.serve_skill)}")
    print(f"  intercept: {st.intercept:.3f}  (logit-scale, sigmoid={_sigmoid(st.intercept):.3f})")
    print(f"  avg_serve: {st.avg_serve:.3f}, avg_return: {st.avg_return:.3f}")

    # Compare estimated vs ground-truth skills (correlation)
    truth_hard = truth[truth["surface"] == "Hard"].set_index("player_id")
    est = pd.DataFrame({
        "player_id": list(st.serve_skill.keys()),
        "est_serve": list(st.serve_skill.values()),
        "est_return": list(st.return_skill.values()),
    }).set_index("player_id")
    merged = truth_hard.join(est, how="inner")
    # Need at least min_obs to be non-fallback
    merged["n_obs"] = merged.index.map(lambda p: st.n_obs.get(p, 0))
    merged = merged[merged["n_obs"] >= 10]
    print(f"\nComparing estimates vs truth (n={len(merged)}, players with >= 10 obs):")
    # Note: estimates are on logit scale, truth is on prob scale.
    # Convert truth to logit for comparison.
    merged["truth_serve_logit"] = _logit(merged["true_serve_skill"]) - st.intercept
    merged["truth_return_logit"] = -(_logit(1 - merged["true_return_skill"]) - st.intercept)
    # The relationship between true latent skill and est skill is more complex
    # under our generating model (not exactly logit-additive), but rank
    # correlation should be strong.
    serve_rank_corr = merged["est_serve"].corr(merged["true_serve_skill"], method="spearman")
    return_rank_corr = merged["est_return"].corr(merged["true_return_skill"], method="spearman")
    print(f"  Spearman corr(est_serve, true_serve_skill) = {serve_rank_corr:.3f}")
    print(f"  Spearman corr(est_return, true_return_skill) = {return_rank_corr:.3f}")
    print("  (high correlation = our estimator recovers true ranking well)")

    # Show top 5 servers and returners by estimate
    est_sorted = est.sort_values("est_serve", ascending=False).head(5)
    print(f"\nTop 5 servers by estimated skill:")
    print(est_sorted)
