import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.http import FileResponse, Http404
from django.conf import settings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

# Cache so we don't reload/retrain on every request
_CACHE = {}


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def get_data():
    if 'data' in _CACHE:
        return _CACHE['data']

    import os
    train_path = os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
    test_path  = os.path.join(os.path.dirname(__file__), 'data', 'test.csv')

    train_df = pd.read_csv(train_path).sample(n=8000, random_state=42).reset_index(drop=True)
    test_df  = pd.read_csv(test_path).sample(n=2000, random_state=42).reset_index(drop=True)

    _CACHE['data'] = (train_df, test_df)
    return train_df, test_df


def get_baseline_model():
    """Train (once) and cache the baseline TF-IDF + Logistic Regression classifier."""
    if 'baseline' in _CACHE:
        return _CACHE['baseline']

    train_df, test_df = get_data()

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test  = vectorizer.transform(test_df['text'])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_df['label'])

    test_acc = accuracy_score(test_df['label'], clf.predict(X_test))

    _CACHE['baseline'] = (vectorizer, clf, test_acc, X_train, X_test)
    return _CACHE['baseline']


def index(request):
    return render(request, 'project3/index.html')


def baseline_view(request):
    vectorizer, clf, test_acc, X_train, X_test = get_baseline_model()
    train_df, test_df = get_data()

    # Per-class accuracy for the report
    preds = clf.predict(X_test)
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = test_df['label'] == i
        per_class_acc[name] = round(accuracy_score(test_df['label'][mask], preds[mask]), 4)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(per_class_acc.keys(), per_class_acc.values(), color='steelblue')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Classifier — Per-Class Accuracy')
    ax.set_ylim(0, 1)

    context = {
        'test_acc': round(test_acc, 4),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'per_class_acc': per_class_acc,
        'plot': fig_to_base64(fig),
    }
    return render(request, 'project3/baseline.html', context)


# ── Task 2: Simulated Expert ────────────────────────────────────────────────
#
# Design: the expert is a domain specialist. They are highly accurate on
# "Sports" and "World" news (topics a generalist human follows closely),
# but noisy/unreliable on "Business" and "Sci/Tech" (more specialized,
# jargon-heavy topics). This mimics a real human assessor who has uneven
# expertise across domains rather than being a uniform oracle.

EXPERT_STRONG_CLASSES = [0, 1]   # World, Sports
EXPERT_STRONG_ACC = 0.97
EXPERT_WEAK_ACC = 0.55


def simulate_expert(true_labels, rng):
    """Given true labels, return noisy expert predictions."""
    expert_preds = []
    for label in true_labels:
        if label in EXPERT_STRONG_CLASSES:
            correct_prob = EXPERT_STRONG_ACC
        else:
            correct_prob = EXPERT_WEAK_ACC

        if rng.random() < correct_prob:
            expert_preds.append(label)
        else:
            # Wrong: pick a random other class
            other_classes = [c for c in range(4) if c != label]
            expert_preds.append(rng.choice(other_classes))
    return np.array(expert_preds)


def expert_view(request):
    train_df, test_df = get_data()
    rng = np.random.default_rng(42)

    expert_preds = simulate_expert(test_df['label'].values, rng)
    overall_acc = accuracy_score(test_df['label'], expert_preds)

    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = test_df['label'] == i
        per_class_acc[name] = round(accuracy_score(test_df['label'][mask], expert_preds[mask]), 4)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ca02c' if i in EXPERT_STRONG_CLASSES else '#d62728' for i in range(4)]
    ax.bar(per_class_acc.keys(), per_class_acc.values(), color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Simulated Expert — Per-Class Accuracy')
    ax.set_ylim(0, 1)

    context = {
        'overall_acc': round(overall_acc, 4),
        'per_class_acc': per_class_acc,
        'strong_classes': [CLASS_NAMES[i] for i in EXPERT_STRONG_CLASSES],
        'weak_classes': [CLASS_NAMES[i] for i in range(4) if i not in EXPERT_STRONG_CLASSES],
        'plot': fig_to_base64(fig),
    }
    return render(request, 'project3/expert.html', context)


# ── Task 3: Learning to Defer ───────────────────────────────────────────────
#
# Strategy: confidence-based deferral. The classifier outputs a probability
# for its top prediction. If that confidence is below a threshold, the
# system defers to the expert instead of predicting itself. The threshold
# is tunable — lower threshold = defer less, higher threshold = defer more.
# This is a simple, explainable, and well-justified deferral rule: the
# system is essentially saying "only ask the human when I'm unsure."

def defer_view(request):
    vectorizer, clf, test_acc, X_train, X_test = get_baseline_model()
    train_df, test_df = get_data()
    rng = np.random.default_rng(42)

    threshold = float(request.GET.get('threshold', 0.6))
    context = {'threshold': threshold}

    probs = clf.predict_proba(X_test)
    confidences = probs.max(axis=1)
    model_preds = probs.argmax(axis=1)

    expert_preds = simulate_expert(test_df['label'].values, rng)
    true_labels = test_df['label'].values

    defer_mask = confidences < threshold

    # Final prediction: model's own prediction, or expert's, depending on deferral
    final_preds = np.where(defer_mask, expert_preds, model_preds)
    team_acc = accuracy_score(true_labels, final_preds)

    defer_rate = defer_mask.mean()

    # How "useful" was each deferral? i.e. was the expert actually more
    # likely to be right than the model on the deferred examples?
    if defer_mask.sum() > 0:
        model_acc_on_deferred = accuracy_score(true_labels[defer_mask], model_preds[defer_mask])
        expert_acc_on_deferred = accuracy_score(true_labels[defer_mask], expert_preds[defer_mask])
    else:
        model_acc_on_deferred = None
        expert_acc_on_deferred = None

    # Sweep thresholds to show team accuracy vs deferral rate trade-off
    thresholds = np.linspace(0.25, 0.95, 15)
    sweep_acc = []
    sweep_rate = []
    for t in thresholds:
        dm = confidences < t
        fp = np.where(dm, expert_preds, model_preds)
        sweep_acc.append(accuracy_score(true_labels, fp))
        sweep_rate.append(dm.mean())

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(thresholds, sweep_acc, marker='o', color='steelblue', label='Team Accuracy')
    ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Current threshold ({threshold})')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Team Accuracy', color='steelblue')
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, sweep_rate, marker='s', color='orange', label='Deferral Rate')
    ax2.set_ylabel('Deferral Rate', color='orange')
    ax2.set_ylim(0, 1)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax1.set_title('Team Accuracy & Deferral Rate vs Threshold')

    context.update({
        'model_only_acc': round(test_acc, 4),
        'team_acc': round(team_acc, 4),
        'defer_rate': round(defer_rate, 4),
        'model_acc_on_deferred': round(model_acc_on_deferred, 4) if model_acc_on_deferred is not None else 'N/A',
        'expert_acc_on_deferred': round(expert_acc_on_deferred, 4) if expert_acc_on_deferred is not None else 'N/A',
        'plot': fig_to_base64(fig),
    })
    return render(request, 'project3/defer.html', context)


# ── Task 4: Active Learning for Expert Competence Discovery ────────────────
#
# Strategy: uncertainty sampling. We do NOT have expert labels upfront.
# We iteratively pick the examples where the classifier is LEAST confident
# and "query" the expert for those (in our simulation, this just means we
# reveal the expert's simulated label for that point). This is justified
# because low-confidence points are exactly the candidates for deferral —
# querying them tells us fastest whether the expert is reliable in that
# region of the input space, which is exactly what we need to learn a good
# deferral policy without wasting expert queries on points the classifier
# is already confident about.
#
# We compare against a RANDOM querying baseline to show uncertainty
# sampling is more label-efficient, especially early on (few queries).
#
# Note: with only 14 candidate thresholds and a simple "best on queried
# points so far" rule, the learned threshold can converge after a fairly
# small number of queries for BOTH strategies — so the gap between
# strategies is most visible early in the curve, not necessarily at the
# very end. We therefore report both the early-stage (first batch) and
# final accuracy, and plot the full trajectory.

def run_active_learning(strategy, n_queries_total, query_batch, rng):
    """
    Simulate active learning: iteratively query the expert on selected
    pool points (a held-out slice of the train set with known true labels,
    used to simulate expert responses). Returns, for each batch, the
    cumulative number of queries made so far and the resulting team
    accuracy on the real test set.
    """
    vectorizer, clf, test_acc, X_train, X_test = get_baseline_model()
    train_df, test_df = get_data()

    # Use a slice of the train set as the "queryable pool" (we pretend we
    # don't know the expert's labels for these until queried)
    pool_df = train_df.sample(n=2000, random_state=1).reset_index(drop=True)
    X_pool = vectorizer.transform(pool_df['text'])
    pool_true_labels = pool_df['label'].values
    pool_expert_labels = simulate_expert(pool_true_labels, rng)

    queried_mask = np.zeros(len(pool_df), dtype=bool)

    accs_over_time = []
    n_queried_list = []

    probs_test = clf.predict_proba(X_test)
    conf_test = probs_test.max(axis=1)
    model_preds_test = probs_test.argmax(axis=1)
    expert_preds_test = simulate_expert(test_df['label'].values, rng)
    true_test = test_df['label'].values

    probs_pool = clf.predict_proba(X_pool)
    conf_pool = probs_pool.max(axis=1)
    model_preds_pool = probs_pool.argmax(axis=1)

    # Finer threshold grid so the learned policy keeps improving as more
    # points are queried, instead of saturating after a handful of batches
    threshold_grid = np.linspace(0.25, 0.95, 29)

    n_batches = n_queries_total // query_batch

    for b in range(n_batches):
        unqueried_idx = np.where(~queried_mask)[0]
        if len(unqueried_idx) == 0:
            break

        if strategy == 'uncertainty':
            # Pick lowest-confidence unqueried points
            scores = conf_pool[unqueried_idx]
            chosen = unqueried_idx[np.argsort(scores)[:query_batch]]
        else:  # random
            chosen = rng.choice(unqueried_idx, size=min(query_batch, len(unqueried_idx)), replace=False)

        queried_mask[chosen] = True

        # Learn best threshold using ONLY queried points so far
        q_idx = np.where(queried_mask)[0]
        if len(q_idx) >= 5:
            best_thresh = 0.5
            best_score = -1
            for t in threshold_grid:
                dm = conf_pool[q_idx] < t
                fp = np.where(dm, pool_expert_labels[q_idx], model_preds_pool[q_idx])
                acc = accuracy_score(pool_true_labels[q_idx], fp)
                if acc > best_score:
                    best_score = acc
                    best_thresh = t
        else:
            best_thresh = 0.5

        # Apply learned threshold to the REAL test set
        dm_test = conf_test < best_thresh
        fp_test = np.where(dm_test, expert_preds_test, model_preds_test)
        team_acc = accuracy_score(true_test, fp_test)

        accs_over_time.append(team_acc)
        n_queried_list.append(int(queried_mask.sum()))

    return n_queried_list, accs_over_time


def active_learning_view(request):
    rng_unc = np.random.default_rng(42)
    rng_rand = np.random.default_rng(123)  # different seed so random sampling isn't artificially identical

    n_total = int(request.GET.get('n_queries', 200))
    batch = 10

    n_q_unc, acc_unc = run_active_learning('uncertainty', n_total, batch, rng_unc)
    n_q_rand, acc_rand = run_active_learning('random', n_total, batch, rng_rand)

    y_min = min(min(acc_unc), min(acc_rand)) - 0.01
    y_max = max(max(acc_unc), max(acc_rand)) + 0.01

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_q_unc, acc_unc, marker='o', label='Uncertainty Sampling', color='steelblue')
    ax.plot(n_q_rand, acc_rand, marker='s', label='Random Sampling', color='gray')
    ax.set_xlabel('Number of Expert Queries')
    ax.set_ylabel('Team Accuracy on Test Set')
    ax.set_title('Active Learning: Team Accuracy vs Expert Queries')
    ax.set_ylim(y_min, y_max)
    ax.legend()

    # Accuracy after just the FIRST batch of queries — this is where
    # uncertainty sampling should show its biggest relative advantage,
    # since it immediately targets the most informative points rather
    # than relying on chance over many queries.
    early_acc_unc = round(acc_unc[0], 4) if acc_unc else None
    early_acc_rand = round(acc_rand[0], 4) if acc_rand else None

    context = {
        'n_total': n_total,
        'early_acc_uncertainty': early_acc_unc,
        'early_acc_random': early_acc_rand,
        'final_acc_uncertainty': round(acc_unc[-1], 4) if acc_unc else None,
        'final_acc_random': round(acc_rand[-1], 4) if acc_rand else None,
        'plot': fig_to_base64(fig),
    }
    return render(request, 'project3/active_learning.html', context)


def download_report(request):
    report_path = os.path.join(settings.BASE_DIR, 'project3', 'static', 'project3', 'report.pdf')
    if not os.path.exists(report_path):
        raise Http404("Report not generated yet.")
    return FileResponse(open(report_path, 'rb'), as_attachment=True, filename='Project3_Report.pdf')