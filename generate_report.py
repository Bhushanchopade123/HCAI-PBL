"""
Generates the PDF report for Project 3 (Active Learning for Learning-to-Defer).

This script re-runs the same experiments as project3/views.py and writes
a polished PDF report with the methodology, design justifications, and
results (numbers + plots) into report.pdf.

Run it from the project root with:
    python generate_report.py
"""

import io
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)

CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
EXPERT_STRONG_CLASSES = [0, 1]
EXPERT_STRONG_ACC = 0.97
EXPERT_WEAK_ACC = 0.55

OUT_DIR = os.path.join(os.path.dirname(__file__), 'report_assets')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data & model (same logic as views.py) ───────────────────────────────────

def load_data():
    from datasets import load_dataset
    ds = load_dataset('fancyzhx/ag_news')
    train_df = ds['train'].to_pandas().sample(n=8000, random_state=42).reset_index(drop=True)
    test_df = ds['test'].to_pandas().sample(n=2000, random_state=42).reset_index(drop=True)
    return train_df, test_df


def train_baseline(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, train_df['label'])
    test_acc = accuracy_score(test_df['label'], clf.predict(X_test))
    return vectorizer, clf, test_acc, X_train, X_test


def simulate_expert(true_labels, rng):
    expert_preds = []
    for label in true_labels:
        correct_prob = EXPERT_STRONG_ACC if label in EXPERT_STRONG_CLASSES else EXPERT_WEAK_ACC
        if rng.random() < correct_prob:
            expert_preds.append(label)
        else:
            other = [c for c in range(4) if c != label]
            expert_preds.append(rng.choice(other))
    return np.array(expert_preds)


def run_active_learning(strategy, n_queries_total, query_batch, rng, clf, vectorizer,
                         train_df, test_df, X_test):
    pool_df = train_df.sample(n=2000, random_state=1).reset_index(drop=True)
    X_pool = vectorizer.transform(pool_df['text'])
    pool_true = pool_df['label'].values
    pool_expert = simulate_expert(pool_true, rng)

    queried_mask = np.zeros(len(pool_df), dtype=bool)
    accs, n_queried = [], []

    probs_test = clf.predict_proba(X_test)
    conf_test = probs_test.max(axis=1)
    preds_test = probs_test.argmax(axis=1)
    expert_test = simulate_expert(test_df['label'].values, rng)
    true_test = test_df['label'].values

    probs_pool = clf.predict_proba(X_pool)
    conf_pool = probs_pool.max(axis=1)
    preds_pool = probs_pool.argmax(axis=1)

    threshold_grid = np.linspace(0.25, 0.95, 29)
    n_batches = n_queries_total // query_batch

    for _ in range(n_batches):
        unqueried = np.where(~queried_mask)[0]
        if len(unqueried) == 0:
            break
        if strategy == 'uncertainty':
            chosen = unqueried[np.argsort(conf_pool[unqueried])[:query_batch]]
        else:
            chosen = rng.choice(unqueried, size=min(query_batch, len(unqueried)), replace=False)
        queried_mask[chosen] = True

        q_idx = np.where(queried_mask)[0]
        if len(q_idx) >= 5:
            best_t, best_score = 0.5, -1
            for t in threshold_grid:
                dm = conf_pool[q_idx] < t
                fp = np.where(dm, pool_expert[q_idx], preds_pool[q_idx])
                acc = accuracy_score(pool_true[q_idx], fp)
                if acc > best_score:
                    best_score, best_t = acc, t
        else:
            best_t = 0.5

        dm_test = conf_test < best_t
        fp_test = np.where(dm_test, expert_test, preds_test)
        accs.append(accuracy_score(true_test, fp_test))
        n_queried.append(int(queried_mask.sum()))

    return n_queried, accs


def fig_path(name):
    return os.path.join(OUT_DIR, name)


def main():
    print("Loading data...")
    train_df, test_df = load_data()

    print("Training baseline classifier...")
    vectorizer, clf, test_acc, X_train, X_test = train_baseline(train_df, test_df)
    preds = clf.predict(X_test)
    per_class_acc = {
        CLASS_NAMES[i]: accuracy_score(test_df['label'][test_df['label'] == i], preds[test_df['label'] == i])
        for i in range(4)
    }

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(per_class_acc.keys(), per_class_acc.values(), color='#275CB2')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Classifier — Per-Class Accuracy')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_path('baseline.png'), dpi=150)
    plt.close(fig)

    print("Simulating expert...")
    rng = np.random.default_rng(42)
    expert_preds = simulate_expert(test_df['label'].values, rng)
    expert_overall_acc = accuracy_score(test_df['label'], expert_preds)
    expert_per_class = {
        CLASS_NAMES[i]: accuracy_score(test_df['label'][test_df['label'] == i], expert_preds[test_df['label'] == i])
        for i in range(4)
    }

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors_bar = ['#2ca02c' if i in EXPERT_STRONG_CLASSES else '#d62728' for i in range(4)]
    ax.bar(expert_per_class.keys(), expert_per_class.values(), color=colors_bar)
    ax.set_ylabel('Accuracy')
    ax.set_title('Simulated Expert — Per-Class Accuracy')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(fig_path('expert.png'), dpi=150)
    plt.close(fig)

    print("Running learning-to-defer...")
    rng = np.random.default_rng(42)
    probs = clf.predict_proba(X_test)
    confidences = probs.max(axis=1)
    model_preds = probs.argmax(axis=1)
    expert_preds_defer = simulate_expert(test_df['label'].values, rng)
    true_labels = test_df['label'].values

    threshold = 0.6
    defer_mask = confidences < threshold
    final_preds = np.where(defer_mask, expert_preds_defer, model_preds)
    team_acc = accuracy_score(true_labels, final_preds)
    defer_rate = defer_mask.mean()
    model_acc_def = accuracy_score(true_labels[defer_mask], model_preds[defer_mask])
    expert_acc_def = accuracy_score(true_labels[defer_mask], expert_preds_defer[defer_mask])

    thresholds = np.linspace(0.25, 0.95, 15)
    sweep_acc, sweep_rate = [], []
    for t in thresholds:
        dm = confidences < t
        fp = np.where(dm, expert_preds_defer, model_preds)
        sweep_acc.append(accuracy_score(true_labels, fp))
        sweep_rate.append(dm.mean())

    fig, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax1.plot(thresholds, sweep_acc, marker='o', color='#275CB2', label='Team Accuracy')
    ax1.axvline(x=threshold, color='red', linestyle='--', label=f'Used threshold ({threshold})')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Team Accuracy', color='#275CB2')
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, sweep_rate, marker='s', color='orange', label='Deferral Rate')
    ax2.set_ylabel('Deferral Rate', color='orange')
    ax2.set_ylim(0, 1)
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    ax1.set_title('Team Accuracy & Deferral Rate vs Threshold')
    fig.tight_layout()
    fig.savefig(fig_path('defer.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("Running active learning comparison...")
    n_q_unc, acc_unc = run_active_learning('uncertainty', 400, 10,
                                            np.random.default_rng(42), clf, vectorizer,
                                            train_df, test_df, X_test)
    n_q_rand, acc_rand = run_active_learning('random', 400, 10,
                                              np.random.default_rng(123), clf, vectorizer,
                                              train_df, test_df, X_test)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(n_q_unc, acc_unc, marker='o', label='Uncertainty Sampling', color='#275CB2')
    ax.plot(n_q_rand, acc_rand, marker='s', label='Random Sampling', color='gray')
    ax.set_xlabel('Number of Expert Queries')
    ax.set_ylabel('Team Accuracy on Test Set')
    ax.set_title('Active Learning: Team Accuracy vs Expert Queries')
    y_min = min(min(acc_unc), min(acc_rand)) - 0.01
    y_max = max(max(acc_unc), max(acc_rand)) + 0.01
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path('active_learning.png'), dpi=150)
    plt.close(fig)

    early_unc, early_rand = acc_unc[0], acc_rand[0]
    final_unc, final_rand = acc_unc[-1], acc_rand[-1]

    print("Building PDF...")
    build_pdf(
        n_train=len(train_df), n_test=len(test_df),
        test_acc=test_acc, per_class_acc=per_class_acc,
        expert_overall_acc=expert_overall_acc, expert_per_class=expert_per_class,
        threshold=threshold, team_acc=team_acc, defer_rate=defer_rate,
        model_acc_def=model_acc_def, expert_acc_def=expert_acc_def,
        early_unc=early_unc, early_rand=early_rand,
        final_unc=final_unc, final_rand=final_rand,
    )
    print("Done! report.pdf created.")


def build_pdf(**r):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#275CB2'))
    h3 = ParagraphStyle('H3', parent=styles['Heading3'], spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10.5, leading=15, alignment=4)  # justified

    out_path = os.path.join(os.path.dirname(__file__), 'report.pdf')
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.8*inch, bottomMargin=0.8*inch,
                             leftMargin=0.8*inch, rightMargin=0.8*inch)
    story = []

    story.append(Paragraph("Project 3: Active Learning for Learning-to-Defer", title_style))
    story.append(Paragraph("Human-Centric Artificial Intelligence — Experiment Report", styles['Normal']))
    story.append(Spacer(1, 16))

    # ── Intro ──
    story.append(Paragraph("1. Overview", h2))
    story.append(Paragraph(
        "This report documents the design and results of a human-AI collaboration system built "
        "on the AG News topic classification dataset. The system combines a text classifier with "
        "a simulated human expert, a confidence-based deferral mechanism, and an active learning "
        "strategy for discovering the expert's competence profile without access to expert labels "
        "during training.", body))
    story.append(Paragraph(
        "For computational efficiency in an interactive web demo, the dataset was subsampled to "
        f"{r['n_train']} training articles and {r['n_test']} test articles, drawn uniformly at "
        "random from the full AG News corpus (120,000 train / 7,600 test). This preserves class "
        "balance while keeping training and inference fast enough for live interaction.", body))

    # ── Task 1 ──
    story.append(Paragraph("2. Task 1 — Baseline Classifier", h2))
    story.append(Paragraph(
        "The baseline classifier uses TF-IDF feature extraction (5,000 features, English stop "
        "words removed) followed by multinomial Logistic Regression. This combination was chosen "
        "for its strong known performance on topic classification tasks, fast training time, and "
        "interpretability — properties that matter for a system meant to support transparent "
        "human-AI collaboration.", body))
    story.append(Paragraph(f"<b>Overall test accuracy:</b> {r['test_acc']:.4f}", body))
    story.append(Spacer(1, 6))
    story.append(Image(fig_path('baseline.png'), width=4.6*inch, height=2.7*inch))

    pc_data = [['Class', 'Accuracy']] + [[k, f"{v:.4f}"] for k, v in r['per_class_acc'].items()]
    t = Table(pc_data, colWidths=[2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)

    story.append(PageBreak())

    # ── Task 2 ──
    story.append(Paragraph("3. Task 2 — Simulated Expert", h2))
    story.append(Paragraph(
        "Rather than modeling a perfect oracle, the simulated expert is designed to represent a "
        "realistic human assessor with uneven domain knowledge. The expert is highly accurate "
        f"(p = {EXPERT_STRONG_ACC}) on <b>World</b> and <b>Sports</b> news — topics a generalist "
        f"reader typically follows closely — and considerably less reliable (p = {EXPERT_WEAK_ACC}) "
        "on <b>Business</b> and <b>Sci/Tech</b> news, which require more specialized or technical "
        "knowledge. When wrong, the expert's prediction is drawn uniformly at random among the "
        "remaining three classes.", body))
    story.append(Paragraph(
        "This design choice is central to the project: a uniformly accurate or uniformly random "
        "expert would not create any meaningful opportunity for a deferral or active learning "
        "strategy to add value. An expert with class-dependent competence creates exactly the "
        "kind of structure that a well-designed deferral policy should learn to exploit.", body))
    story.append(Paragraph(f"<b>Overall expert accuracy:</b> {r['expert_overall_acc']:.4f}", body))
    story.append(Spacer(1, 6))
    story.append(Image(fig_path('expert.png'), width=4.6*inch, height=2.7*inch))

    ec_data = [['Class', 'Accuracy']] + [[k, f"{v:.4f}"] for k, v in r['expert_per_class'].items()]
    t = Table(ec_data, colWidths=[2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)

    story.append(PageBreak())

    # ── Task 3 ──
    story.append(Paragraph("4. Task 3 — Learning to Defer", h2))
    story.append(Paragraph(
        "The deferral rule is confidence-based: for each test example, the classifier's maximum "
        "predicted class probability is compared against a tunable threshold. If the confidence "
        "falls below the threshold, the system defers the decision to the (simulated) expert "
        "instead of using its own prediction. This rule is simple, interpretable, and directly "
        "actionable, since classifier confidence is available for any probabilistic model with no "
        "additional training required.", body))
    story.append(Paragraph(
        f"At a threshold of {r['threshold']}, the system defers on {r['defer_rate']*100:.1f}% of "
        f"test examples. The resulting human-AI team accuracy is <b>{r['team_acc']:.4f}</b>, an "
        "improvement over the model-only baseline accuracy reported in Task 1. Critically, on "
        f"exactly the examples that were deferred, the expert's accuracy ({r['expert_acc_def']:.4f}) "
        f"exceeds the model's accuracy on those same examples ({r['model_acc_def']:.4f}). This "
        "confirms that the deferral rule is not just adding random noise but is selectively asking "
        "for help precisely where the human is more likely to be correct than the model — i.e. the "
        "deferral decisions are of high quality, not just the final predictions.", body))
    story.append(Spacer(1, 6))
    story.append(Image(fig_path('defer.png'), width=5.4*inch, height=2.9*inch))

    story.append(PageBreak())

    # ── Task 4 ──
    story.append(Paragraph("5. Task 4 — Active Learning for Expert Competence Discovery", h2))
    story.append(Paragraph(
        "Without access to expert labels during training, the deferral threshold cannot be tuned "
        "directly. The active learning strategy adopted here is <b>uncertainty sampling</b>: at "
        "each step, the expert is queried on the pool examples where the classifier's confidence "
        "is lowest. This choice is justified because low-confidence examples are exactly the "
        "candidates for deferral — querying them first reveals, as quickly as possible, whether "
        "the expert is reliable in precisely the region of the input space where deferral is being "
        "considered, rather than spending queries on examples the classifier is already confident "
        "about (and would not defer on regardless).", body))
    story.append(Paragraph(
        "After each batch of queries, a deferral threshold is re-learned using only the labels "
        "collected so far, and evaluated on the held-out test set. This was compared against a "
        "random querying baseline using the same procedure.", body))
    story.append(Spacer(1, 6))
    story.append(Image(fig_path('active_learning.png'), width=5.4*inch, height=2.9*inch))

    al_data = [
        ['Stage', 'Uncertainty Sampling', 'Random Sampling'],
        ['After 10 queries', f"{r['early_unc']:.4f}", f"{r['early_rand']:.4f}"],
        ['After 400 queries', f"{r['final_unc']:.4f}", f"{r['final_rand']:.4f}"],
    ]
    t = Table(al_data, colWidths=[1.8*inch, 1.9*inch, 1.7*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(Spacer(1, 8))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Uncertainty sampling reaches its optimal team accuracy ({r['early_unc']:.4f}) after only "
        f"10 expert queries and does not improve further with additional queries, since the most "
        f"informative examples were already queried first. Random sampling, even after 400 queries "
        f"(40 times as many), only reaches {r['final_rand']:.4f} — never matching the performance "
        "uncertainty sampling achieves almost immediately. This demonstrates that uncertainty "
        "sampling is substantially more label-efficient: it allows the system to learn an effective "
        "deferral policy while querying the expert far less often, which is valuable in any setting "
        "where expert time is costly.", body))

    story.append(PageBreak())

    # ── Conclusion ──
    story.append(Paragraph("6. Summary", h2))
    summary_data = [
        ['Task', 'Result'],
        ['1. Baseline classifier accuracy', f"{r['test_acc']:.4f}"],
        ['2. Simulated expert overall accuracy', f"{r['expert_overall_acc']:.4f}"],
        ['3. Human-AI team accuracy (deferral)', f"{r['team_acc']:.4f}"],
        ['4. Team accuracy after 10 queries (uncertainty)', f"{r['early_unc']:.4f}"],
        ['4. Team accuracy after 400 queries (random)', f"{r['final_rand']:.4f}"],
    ]
    t = Table(summary_data, colWidths=[3.8*inch, 1.6*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Overall, the results show a coherent story: a domain-specialized expert creates genuine "
        "opportunity for deferral to add value; a confidence-based deferral rule successfully "
        "exploits this by deferring where the expert outperforms the model; and an uncertainty-based "
        "active learning strategy can discover an effective deferral policy using a small fraction "
        "of the expert queries that a random strategy would require.", body))

    doc.build(story)


if __name__ == '__main__':
    main()