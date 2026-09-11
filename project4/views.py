import os
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render, redirect
from django.http import FileResponse, Http404, JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

# ── Data loading ─────────────────────────────────────────────────────────────

_CACHE = {}

def load_movies():
    if 'movies' in _CACHE:
        return _CACHE['movies']

    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'movie_metadata.csv')
    df = pd.read_csv(csv_path, encoding='latin-1')

    # ── Task 1: Feature extraction ───────────────────────────────────────────
    # We use the following features, justified as follows:
    # - imdb_score:        direct signal of overall quality/reception
    # - duration:          proxy for viewing commitment required
    # - gross (log):       popularity/commercial appeal
    # - title_year:        recency preference
    # - genres (binary):   captures genre taste — the most direct driver of
    #                      movie preference for most users
    # We drop rows missing any of these fields since we cannot impute reliably.

    df = df.dropna(subset=['movie_title', 'imdb_score', 'duration',
                            'gross', 'title_year', 'genres'])
    df = df.drop_duplicates(subset=['movie_title'])
    df = df.reset_index(drop=True)

    # Clean title
    df['movie_title'] = df['movie_title'].str.strip()

    # Log-scale gross (very skewed distribution)
    df['log_gross'] = np.log1p(df['gross'])

    # Binary genre features
    all_genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
                  'History', 'Horror', 'Music', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']
    for g in all_genres:
        df[f'genre_{g}'] = df['genres'].str.contains(g, case=False, na=False).astype(float)

    # Normalise numeric features to [0,1] so they are on the same scale
    for col in ['imdb_score', 'duration', 'log_gross', 'title_year']:
        mn, mx = df[col].min(), df[col].max()
        df[f'norm_{col}'] = (df[col] - mn) / (mx - mn + 1e-9)

    feature_cols = (
        ['norm_imdb_score', 'norm_duration', 'norm_log_gross', 'norm_title_year'] +
        [f'genre_{g}' for g in all_genres]
    )

    _CACHE['movies'] = (df, feature_cols, all_genres)
    return df, feature_cols, all_genres


# ── Bradley-Terry / Plackett-Luce utilities ───────────────────────────────────
#
# Task 2: Extension of Bradley-Terry to full rankings.
#
# Standard Bradley-Terry (pairwise):
#   P(i beats j) = exp(w^T x_i) / (exp(w^T x_i) + exp(w^T x_j))
#
# Plackett-Luce extension (ranking of n items i1 > i2 > ... > in):
#   P(i1 > i2 > ... > in) = prod_{k=1}^{n} exp(w^T x_{ik}) / sum_{j=k}^{n} exp(w^T x_{ij})
#
# This is the unique consistent extension of Bradley-Terry to full rankings:
# at each position k, the probability of item ik being chosen from the
# remaining items follows the same ratio rule as Bradley-Terry pairwise,
# applied to the remaining set. This is also known as the "Luce choice axiom."

def utility(w, X):
    """Linear utility: w^T x for each row of X."""
    return X @ w


def bt_pairwise_prob(w, xi, xj):
    """P(i preferred to j) under Bradley-Terry."""
    ui, uj = w @ xi, w @ xj
    return np.exp(ui) / (np.exp(ui) + np.exp(uj))


def plackett_luce_log_prob(w, X_ranked):
    """
    Log-probability of a full ranking under Plackett-Luce.
    X_ranked: feature matrix with rows in ranked order (best first).
    """
    utils = X_ranked @ w
    log_prob = 0.0
    for k in range(len(utils)):
        remaining = utils[k:]
        log_prob += utils[k] - np.log(np.sum(np.exp(remaining - remaining.max())) + 1e-9) - remaining.max()
    return log_prob


def fit_w_from_comparisons(comparisons, feature_cols, df, n_features, lr=0.05, steps=200):
    """
    MLE for w given a list of (winner_idx, loser_idx) pairwise comparisons
    using gradient ascent on the Bradley-Terry log-likelihood.
    """
    w = np.zeros(n_features)
    X = df[feature_cols].values

    for _ in range(steps):
        grad = np.zeros(n_features)
        for (wi, li) in comparisons:
            xi, xj = X[wi], X[li]
            p = bt_pairwise_prob(w, xi, xj)
            grad += (1 - p) * xi - p * xj
        w += lr * grad
        # L2 regularisation to prevent divergence
        w -= 0.01 * w
    return w


def fit_w_from_rankings(rankings, feature_cols, df, n_features, lr=0.05, steps=200):
    """
    MLE for w given a list of ranked index lists under Plackett-Luce,
    using gradient ascent on the log-likelihood.
    """
    w = np.zeros(n_features)
    X = df[feature_cols].values

    for _ in range(steps):
        grad = np.zeros(n_features)
        for ranked_indices in rankings:
            X_r = X[ranked_indices]
            utils = X_r @ w
            for k in range(len(ranked_indices)):
                remaining_utils = utils[k:]
                exp_u = np.exp(remaining_utils - remaining_utils.max())
                denom = exp_u.sum() + 1e-9
                # gradient contribution
                for j in range(k, len(ranked_indices)):
                    coeff = -exp_u[j - k] / denom
                    if j == k:
                        coeff += 1.0
                    grad += coeff * X_r[j]
        w += lr * grad
        w -= 0.01 * w
    return w


def recommend_top_k(w, df, feature_cols, k=5, exclude_indices=None):
    """Return top-k movie indices by predicted utility."""
    X = df[feature_cols].values
    utils = X @ w
    if exclude_indices:
        utils[exclude_indices] = -np.inf
    return np.argsort(utils)[::-1][:k].tolist()


# ── Session helpers ───────────────────────────────────────────────────────────

def get_session_data(request):
    if 'p4_data' not in request.session:
        request.session['p4_data'] = {
            'design': None,
            'comparisons': [],   # list of [winner_idx, loser_idx]
            'rankings': [],      # list of ranked index lists
            'seen_indices': [],
            'round': 0,
        }
    return request.session['p4_data']


def save_session_data(request, data):
    request.session['p4_data'] = data
    request.session.modified = True


# ── Views ─────────────────────────────────────────────────────────────────────

def index(request):
    return render(request, 'project4/index.html')


def study_view(request):
    """Landing page for the study — participant chooses Design 1 or Design 2."""
    # Reset session for a fresh run
    request.session['p4_data'] = {
        'design': None,
        'comparisons': [],
        'rankings': [],
        'seen_indices': [],
        'round': 0,
    }
    return render(request, 'project4/study.html')


def pairwise_view(request):
    """Design 1: show 2 movies, user picks preferred one."""
    df, feature_cols, all_genres = load_movies()
    data = get_session_data(request)
    data['design'] = 'pairwise'

    MAX_ROUNDS = 10

    if data['round'] >= MAX_ROUNDS:
        return redirect('project4:results')

    # Sample 2 unseen movies
    seen = set(data['seen_indices'])
    available = [i for i in range(len(df)) if i not in seen]
    if len(available) < 2:
        return redirect('project4:results')

    chosen = np.random.choice(available, size=2, replace=False).tolist()
    data['current_pair'] = chosen
    save_session_data(request, data)

    m1 = df.iloc[chosen[0]]
    m2 = df.iloc[chosen[1]]

    context = {
        'round': data['round'] + 1,
        'max_rounds': MAX_ROUNDS,
        'movie1': {
            'idx': chosen[0],
            'title': m1['movie_title'],
            'year': int(m1['title_year']),
            'genres': m1['genres'].split('|')[0] if '|' in str(m1['genres']) else m1['genres'],
            'score': m1['imdb_score'],
            'director': m1.get('director_name', 'Unknown'),
        },
        'movie2': {
            'idx': chosen[1],
            'title': m2['movie_title'],
            'year': int(m2['title_year']),
            'genres': m2['genres'].split('|')[0] if '|' in str(m2['genres']) else m2['genres'],
            'score': m2['imdb_score'],
            'director': m2.get('director_name', 'Unknown'),
        },
    }
    return render(request, 'project4/pairwise.html', context)


def submit_pairwise(request):
    """Record pairwise choice and go to next round."""
    if request.method == 'POST':
        df, feature_cols, all_genres = load_movies()
        data = get_session_data(request)

        winner_idx = int(request.POST.get('winner'))
        pair = data.get('current_pair', [])
        loser_idx = pair[0] if pair[1] == winner_idx else pair[1]

        data['comparisons'].append([winner_idx, loser_idx])
        data['seen_indices'].extend(pair)
        data['round'] += 1
        save_session_data(request, data)

    return redirect('project4:pairwise')


def ranking_view(request):
    """Design 2: show 10 movies, user ranks them."""
    df, feature_cols, all_genres = load_movies()
    data = get_session_data(request)
    data['design'] = 'ranking'

    MAX_ROUNDS = 3

    if data['round'] >= MAX_ROUNDS:
        return redirect('project4:results')

    seen = set(data['seen_indices'])
    available = [i for i in range(len(df)) if i not in seen]
    if len(available) < 10:
        return redirect('project4:results')

    chosen = np.random.choice(available, size=10, replace=False).tolist()
    data['current_ranking_pool'] = chosen
    save_session_data(request, data)

    movies = []
    for idx in chosen:
        m = df.iloc[idx]
        movies.append({
            'idx': idx,
            'title': m['movie_title'],
            'year': int(m['title_year']),
            'genres': m['genres'].split('|')[0] if '|' in str(m['genres']) else m['genres'],
            'score': m['imdb_score'],
            'director': m.get('director_name', 'Unknown'),
        })

    context = {
        'round': data['round'] + 1,
        'max_rounds': MAX_ROUNDS,
        'movies': movies,
        'movies_json': json.dumps(movies),
    }
    return render(request, 'project4/ranking.html', context)


def submit_ranking(request):
    """Record ranking and go to next round."""
    if request.method == 'POST':
        df, feature_cols, all_genres = load_movies()
        data = get_session_data(request)

        # ranked_order is a comma-separated list of movie indices
        ranked_str = request.POST.get('ranked_order', '')
        if ranked_str:
            ranked_indices = [int(x) for x in ranked_str.split(',')]
            data['rankings'].append(ranked_indices)
            data['seen_indices'].extend(data.get('current_ranking_pool', []))
            data['round'] += 1
            save_session_data(request, data)

    return redirect('project4:ranking')


def results_view(request):
    """Show estimated preferences and top recommendations."""
    df, feature_cols, all_genres = load_movies()
    data = get_session_data(request)
    n_features = len(feature_cols)

    design = data.get('design', 'pairwise')
    comparisons = data.get('comparisons', [])
    rankings = data.get('rankings', [])

    if design == 'pairwise' and len(comparisons) > 0:
        w = fit_w_from_comparisons(comparisons, feature_cols, df, n_features)
    elif design == 'ranking' and len(rankings) > 0:
        w = fit_w_from_rankings(rankings, feature_cols, df, n_features)
    else:
        w = np.zeros(n_features)

    # Top recommendations
    seen = data.get('seen_indices', [])
    top_indices = recommend_top_k(w, df, feature_cols, k=5, exclude_indices=seen)

    recommendations = []
    for idx in top_indices:
        m = df.iloc[idx]
        recommendations.append({
            'title': m['movie_title'],
            'year': int(m['title_year']),
            'genres': m['genres'],
            'score': m['imdb_score'],
            'director': m.get('director_name', 'Unknown'),
        })

    # Plot top feature weights
    numeric_labels = ['IMDB Score', 'Duration', 'Gross', 'Year']
    numeric_weights = w[:4].tolist()

    genre_weights = list(zip(all_genres, w[4:].tolist()))
    genre_weights.sort(key=lambda x: abs(x[1]), reverse=True)
    top_genres = genre_weights[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors1 = ['#275CB2' if v >= 0 else '#d62728' for v in numeric_weights]
    ax1.barh(numeric_labels, numeric_weights, color=colors1)
    ax1.set_title('Numeric Feature Weights')
    ax1.axvline(0, color='black', linewidth=0.8)

    genre_names = [g[0] for g in top_genres]
    genre_vals = [g[1] for g in top_genres]
    colors2 = ['#275CB2' if v >= 0 else '#d62728' for v in genre_vals]
    ax2.barh(genre_names, genre_vals, color=colors2)
    ax2.set_title('Top Genre Weights')
    ax2.axvline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    weight_plot = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    context = {
        'design': design,
        'n_comparisons': len(comparisons),
        'n_rankings': len(rankings),
        'recommendations': recommendations,
        'weight_plot': weight_plot,
    }
    return render(request, 'project4/results.html', context)


def download_report(request):
    report_path = os.path.join(settings.BASE_DIR, 'project4', 'static', 'project4', 'report.pdf')
    if not os.path.exists(report_path):
        raise Http404("Report not yet generated. Please run generate_report_p4.py first.")
    return FileResponse(open(report_path, 'rb'), as_attachment=True, filename='Project4_Report.pdf')