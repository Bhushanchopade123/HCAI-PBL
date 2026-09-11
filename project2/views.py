import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score



# ── helpers ──────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img


def load_data():
    """Load and clean the Palmer Penguins dataset."""
    import os
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'penguins.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna()

    # Encode categorical columns
    le_species  = LabelEncoder()
    le_island   = LabelEncoder()
    le_sex      = LabelEncoder()

    df['species_enc'] = le_species.fit_transform(df['species'])
    df['island_enc']  = le_island.fit_transform(df['island'])
    df['sex_enc']     = le_sex.fit_transform(df['sex'])

    feature_cols = [
        'bill_length_mm', 'bill_depth_mm',
        'flipper_length_mm', 'body_mass_g',
        'island_enc', 'sex_enc'
    ]
    numerical_cols = [
        'bill_length_mm', 'bill_depth_mm',
        'flipper_length_mm', 'body_mass_g'
    ]

    X = df[feature_cols].values
    y = df['species_enc'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (df, X, y, X_train, X_test, y_train, y_test,
            feature_cols, numerical_cols,
            le_species, le_island, le_sex)


# ── views ─────────────────────────────────────────────────────────────────────

def index(request):
    return render(request, 'project2/index.html')


def tree_view(request):
    context = {}
    (df, X, y, X_train, X_test, y_train, y_test,
     feature_cols, numerical_cols,
     le_species, le_island, le_sex) = load_data()

    # --- Task 1: single tree ---
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc  = round(accuracy_score(y_test, clf.predict(X_test)), 4)
    n_leaves = clf.get_n_leaves()

    # Draw tree
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(clf, feature_names=feature_cols,
              class_names=le_species.classes_,
              filled=True, ax=ax, max_depth=3)
    context['tree_plot']  = fig_to_base64(fig)
    context['acc']        = acc
    context['n_leaves']   = n_leaves

    # --- Task 2: lambda slider ---
    lam = float(request.GET.get('lambda', 0.0))
    context['lam'] = lam

    leaf_range = list(range(2, 51))
    best_score = None
    best_model = None
    results    = []

    for max_l in leaf_range:
        m = DecisionTreeClassifier(max_leaf_nodes=max_l, random_state=42)
        m.fit(X_train, y_train)
        a  = accuracy_score(y_test, m.predict(X_test))
        nl = m.get_n_leaves()
        score = a - lam * nl          # maximise accuracy, penalise complexity
        results.append((max_l, a, nl, score))
        if best_score is None or score > best_score:
            best_score = score
            best_model = m
            best_acc   = a
            best_leaves = nl

    context['best_acc']    = round(best_acc, 4)
    context['best_leaves'] = best_leaves

    # Plot accuracy vs leaves
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot([r[0] for r in results], [r[1] for r in results],
             marker='o', label='Test Accuracy')
    ax2.axvline(x=best_leaves, color='red', linestyle='--',
                label=f'Best (λ={lam}): {best_leaves} leaves')
    ax2.set_xlabel('Max Leaf Nodes')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Complexity (Decision Tree)')
    ax2.legend()
    context['acc_plot'] = fig_to_base64(fig2)

    # Draw best tree
    fig3, ax3 = plt.subplots(figsize=(20, 8))
    plot_tree(best_model, feature_names=feature_cols,
              class_names=le_species.classes_,
              filled=True, ax=ax3, max_depth=4)
    context['best_tree_plot'] = fig_to_base64(fig3)

    return render(request, 'project2/tree.html', context)


def logistic_view(request):
    context = {}
    (df, X, y, X_train, X_test, y_train, y_test,
     feature_cols, numerical_cols,
     le_species, le_island, le_sex) = load_data()

    lam = float(request.GET.get('lambda', 0.0))
    context['lam'] = lam

    # C values: higher C = less regularization
    C_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
    results  = []
    best_score = None

    for C in C_values:
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(X_train, y_train)
        a = accuracy_score(y_test, m.predict(X_test))
        # Complexity = mean absolute weight
        omega = np.mean(np.abs(m.coef_))
        score = a - lam * omega
        results.append((C, a, omega, score))
        if best_score is None or score > best_score:
            best_score = score
            best_C     = C
            best_acc   = a
            best_omega = omega

    context['best_acc']   = round(best_acc, 4)
    context['best_C']     = best_C
    context['best_omega'] = round(best_omega, 4)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx([r[0] for r in results], [r[1] for r in results],
                marker='o', label='Test Accuracy')
    ax.axvline(x=best_C, color='red', linestyle='--',
               label=f'Best C={best_C}')
    ax.set_xlabel('C (inverse regularization)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Complexity (Logistic Regression)')
    ax.legend()
    context['acc_plot'] = fig_to_base64(fig)

    return render(request, 'project2/logistic.html', context)


def counterfactual_view(request):
    context = {}
    (df, X, y, X_train, X_test, y_train, y_test,
     feature_cols, numerical_cols,
     le_species, le_island, le_sex) = load_data()

    species_names = list(le_species.classes_)
    context['species_names'] = species_names
    context['n_samples']     = len(df)

    model_type = request.GET.get('model', 'tree')
    lam        = float(request.GET.get('lambda', 0.0))
    sample_idx = int(request.GET.get('sample_idx', 0))
    target_cls = int(request.GET.get('target', 1))

    context['model_type'] = model_type
    context['lam']        = lam
    context['sample_idx'] = sample_idx
    context['target_cls'] = target_cls

    # Train selected model
    if model_type == 'tree':
        leaf_range = range(2, 51)
        best_score = None
        model = None
        for max_l in leaf_range:
            m = DecisionTreeClassifier(max_leaf_nodes=max_l, random_state=42)
            m.fit(X_train, y_train)
            a  = accuracy_score(y_test, m.predict(X_test))
            nl = m.get_n_leaves()
            s  = a - lam * nl
            if best_score is None or s > best_score:
                best_score = s
                model = m
    else:
        C_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        best_score = None
        model = None
        for C in C_values:
            m = LogisticRegression(C=C, max_iter=1000, random_state=42)
            m.fit(X_train, y_train)
            a     = accuracy_score(y_test, m.predict(X_test))
            omega = np.mean(np.abs(m.coef_))
            s     = a - lam * omega
            if best_score is None or s > best_score:
                best_score = s
                model = m

    # Selected point
    x = X[sample_idx]
    context['selected_point'] = dict(zip(feature_cols, x.round(2)))
    context['actual_class']   = species_names[y[sample_idx]]

    # MAD for weighting
    mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
    mad[mad == 0] = 1.0

    # Generate counterfactuals
    N = 10000
    counterfactuals = []
    np.random.seed(42)

    samples = []
    for j in range(X.shape[1]):
        col = X[:, j]
        unique_vals = np.unique(col)
        if len(unique_vals) <= 5:
            # categorical/binary: sample from existing values
            noisy = np.random.choice(unique_vals, size=N)
        else:
            # numerical: sample around x[j]
            std = np.std(col)
            noisy = np.random.normal(loc=x[j], scale=std, size=N)
        samples.append(noisy)

    samples = np.column_stack(samples)
    preds   = model.predict(samples)
    mask    = preds == target_cls
    cf_candidates = samples[mask]

    if len(cf_candidates) > 0:
        # MAD-weighted L1 distance
        dists = np.sum(np.abs(cf_candidates - x) / mad, axis=1)
        top_k = np.argsort(dists)[:5]
        for idx in top_k:
            cf = cf_candidates[idx]
            counterfactuals.append({
                'values': dict(zip(feature_cols, cf.round(2))),
                'distance': round(dists[idx], 3)
            })

    context['counterfactuals'] = counterfactuals
    context['target_name']     = species_names[target_cls]

    return render(request, 'project2/counterfactual.html', context)


def feature_effects_view(request):
    context = {}
    (df, X, y, X_train, X_test, y_train, y_test,
     feature_cols, numerical_cols,
     le_species, le_island, le_sex) = load_data()

    model_type   = request.GET.get('model', 'tree')
    lam          = float(request.GET.get('lambda', 0.0))
    selected_feat = request.GET.get('feature', numerical_cols[0])
    feat_idx     = feature_cols.index(selected_feat)

    context['model_type']      = model_type
    context['lam']             = lam
    context['selected_feat']   = selected_feat
    context['numerical_cols']  = numerical_cols

    # Train selected model
    if model_type == 'tree':
        leaf_range = range(2, 51)
        best_score = None
        model = None
        for max_l in leaf_range:
            m = DecisionTreeClassifier(max_leaf_nodes=max_l, random_state=42)
            m.fit(X_train, y_train)
            a  = accuracy_score(y_test, m.predict(X_test))
            nl = m.get_n_leaves()
            s  = a - lam * nl
            if best_score is None or s > best_score:
                best_score = s
                model = m
    else:
        C_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        best_score = None
        model = None
        for C in C_values:
            m = LogisticRegression(C=C, max_iter=1000, random_state=42)
            m.fit(X_train, y_train)
            a     = accuracy_score(y_test, m.predict(X_test))
            omega = np.mean(np.abs(m.coef_))
            s     = a - lam * omega
            if best_score is None or s > best_score:
                best_score = s
                model = m

    species_names = list(le_species.classes_)
    grid = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)
    n_classes = 3

    # ── PDP ──────────────────────────────────────────────────────────────────
    pdp = np.zeros((n_classes, len(grid)))
    for i, val in enumerate(grid):
        X_mod = X.copy()
        X_mod[:, feat_idx] = val
        probs = model.predict_proba(X_mod)
        pdp[:, i] = probs.mean(axis=0)

    fig_pdp, ax_pdp = plt.subplots(figsize=(8, 4))
    for c in range(n_classes):
        ax_pdp.plot(grid, pdp[c], label=species_names[c])
    ax_pdp.set_xlabel(selected_feat)
    ax_pdp.set_ylabel('Average Predicted Probability')
    ax_pdp.set_title(f'PDP — {selected_feat}')
    ax_pdp.legend()
    context['pdp_plot'] = fig_to_base64(fig_pdp)

    # ── ALE ──────────────────────────────────────────────────────────────────
    n_bins = 20
    quantiles = np.percentile(X[:, feat_idx],
                              np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    ale = np.zeros((n_classes, len(quantiles) - 1))
    ale_centers = []

    for k in range(len(quantiles) - 1):
        mask = ((X[:, feat_idx] >= quantiles[k]) &
                (X[:, feat_idx] <  quantiles[k + 1]))
        if mask.sum() == 0:
            ale_centers.append((quantiles[k] + quantiles[k+1]) / 2)
            continue
        X_low  = X[mask].copy(); X_low[:,  feat_idx] = quantiles[k]
        X_high = X[mask].copy(); X_high[:, feat_idx] = quantiles[k + 1]
        diff = (model.predict_proba(X_high) -
                model.predict_proba(X_low))
        ale[:, k] = diff.mean(axis=0)
        ale_centers.append((quantiles[k] + quantiles[k + 1]) / 2)

    ale = np.cumsum(ale, axis=1)
    ale -= ale.mean(axis=1, keepdims=True)

    fig_ale, ax_ale = plt.subplots(figsize=(8, 4))
    for c in range(n_classes):
        ax_ale.plot(ale_centers, ale[c], label=species_names[c])
    ax_ale.set_xlabel(selected_feat)
    ax_ale.set_ylabel('ALE')
    ax_ale.set_title(f'ALE — {selected_feat}')
    ax_ale.legend()
    context['ale_plot'] = fig_to_base64(fig_ale)

    return render(request, 'project2/feature_effects.html', context)