"""
Generates the PDF report for Project 4 (Preference Elicitation).
Run from the project root with:
    python generate_report_p4.py
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

def build_pdf():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor('#275CB2'))
    h3 = ParagraphStyle('H3', parent=styles['Heading3'], spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle('Body', parent=styles['Normal'], fontSize=10.5, leading=15, alignment=4)

    out_path = os.path.join(os.path.dirname(__file__), 'report_p4.pdf')
    doc = SimpleDocTemplate(out_path, pagesize=letter,
                             topMargin=0.8*inch, bottomMargin=0.8*inch,
                             leftMargin=0.8*inch, rightMargin=0.8*inch)
    story = []

    # ── Title ──
    story.append(Paragraph("Project 4: Preference Elicitation", title_style))
    story.append(Paragraph("Human-Centric Artificial Intelligence — Method & Study Design Report", styles['Normal']))
    story.append(Spacer(1, 16))

    # ── Task 1: Feature Representation ──
    story.append(Paragraph("1. Task 1 — Feature Representation", h2))
    story.append(Paragraph(
        "The utility function U(x) = w^T x requires a fixed-dimensional feature vector x for "
        "each movie. We extract the following features from the IMDB 5000 dataset:", body))

    feat_data = [
        ['Feature', 'Type', 'Justification'],
        ['IMDB Score', 'Numeric (normalised)', 'Direct measure of overall quality and reception'],
        ['Duration', 'Numeric (normalised)', 'Proxy for viewing commitment; users differ in tolerance'],
        ['Gross Revenue (log)', 'Numeric (normalised)', 'Captures commercial popularity and mainstream appeal'],
        ['Release Year', 'Numeric (normalised)', 'Captures recency preference — some users prefer classics'],
        ['Genre (21 binary flags)', 'Binary', 'Most direct driver of movie preference for most users'],
    ]
    t = Table(feat_data, colWidths=[1.5*inch, 1.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Numeric features are normalised to [0,1] using min-max scaling so that they are "
        "on a comparable scale with the binary genre indicators. Log-scaling is applied to "
        "gross revenue before normalisation because the raw revenue distribution is highly "
        "right-skewed (a few blockbusters dominate). Rows with missing values in any of "
        "these fields are dropped, leaving approximately 3,800 usable movies.", body))

    story.append(Paragraph(
        "This feature set was chosen to balance expressiveness with simplicity: genres directly "
        "capture taste, while the numeric features capture secondary preferences such as quality "
        "threshold (IMDB score), time investment (duration), and cultural currency (popularity "
        "and recency). Director and cast features were considered but omitted because they create "
        "a very high-dimensional, sparse representation that is poorly suited to a linear utility "
        "model with few observations.", body))

    story.append(PageBreak())

    # ── Task 2: Bradley-Terry Extension ──
    story.append(Paragraph("2. Task 2 — Extending Bradley-Terry to Rankings", h2))

    story.append(Paragraph("2.1 Standard Bradley-Terry Model (Pairwise)", h3))
    story.append(Paragraph(
        "The Bradley-Terry model defines the probability that item i is preferred over item j as:", body))
    story.append(Paragraph(
        "<b>P(i &gt; j) = exp(U(i)) / (exp(U(i)) + exp(U(j)))</b>", body))
    story.append(Paragraph(
        "where U(i) = w^T x_i is the utility of item i. Given a set of pairwise comparisons, "
        "the weight vector w is estimated by maximum likelihood, i.e. by maximising the sum of "
        "log-probabilities of the observed outcomes. We use gradient ascent with L2 regularisation "
        "to prevent divergence.", body))

    story.append(Paragraph("2.2 Plackett-Luce Extension (Full Rankings)", h3))
    story.append(Paragraph(
        "To model a full ranking i1 > i2 > ... > in of n items, we use the Plackett-Luce model, "
        "which is the unique consistent extension of Bradley-Terry to rankings. It is defined as:", body))
    story.append(Paragraph(
        "<b>P(i1 &gt; i2 &gt; ... &gt; in) = "
        "prod_{k=1}^{n} exp(U(i_k)) / sum_{j=k}^{n} exp(U(i_j))</b>", body))
    story.append(Paragraph(
        "The intuition is straightforward: at each position k in the ranking, the probability "
        "of item i_k being placed next (given that items i_1, ..., i_{k-1} have already been "
        "placed) follows the same ratio rule as Bradley-Terry pairwise comparison, but applied "
        "to the remaining unranked items. This is equivalent to saying the user repeatedly "
        "selects their favourite from the remaining candidates.", body))
    story.append(Paragraph(
        "This extension is justified by the Luce choice axiom (independence of irrelevant "
        "alternatives): the relative probability of preferring i over j does not depend on "
        "which other items are in the choice set. This is a natural and widely-used assumption "
        "in preference modelling, and makes the model tractable — the log-likelihood decomposes "
        "into a sum over positions, enabling efficient gradient-based MLE.", body))
    story.append(Paragraph(
        "Note that a ranking of n items provides C(n,2) = n(n-1)/2 implicit pairwise comparisons, "
        "so Design 2 (ranking 10 items) provides up to 45 comparisons per round, compared to 1 "
        "per round in Design 1. This is a key efficiency advantage of the ranking interface.", body))

    story.append(PageBreak())

    # ── Task 3: User Study Design ──
    story.append(Paragraph("3. Task 3 — User Study Design", h2))

    story.append(Paragraph("3.1 Research Hypothesis", h3))
    story.append(Paragraph(
        "<b>Primary hypothesis:</b> Design 2 (ranking) produces a more accurate estimate of "
        "user preferences (measured by recommendation quality) than Design 1 (pairwise comparison) "
        "given the same total number of movies evaluated by the participant.", body))
    story.append(Paragraph(
        "<b>Secondary hypothesis:</b> Design 1 (pairwise) is perceived as less cognitively "
        "demanding and more enjoyable than Design 2 (ranking), even if it yields less information "
        "per interaction.", body))

    story.append(Paragraph("3.2 Study Design", h3))
    story.append(Paragraph(
        "We use a <b>within-subjects design</b>: each participant completes both interfaces in "
        "counterbalanced order (half start with Design 1, half with Design 2) to control for "
        "learning effects and individual taste differences. The two sessions are separated by "
        "at least 48 hours to minimise carryover effects.", body))

    story.append(Paragraph("3.3 Participants and Recruitment", h3))
    story.append(Paragraph(
        "We target <b>40 participants</b> (20 per counterbalancing condition), recruited via "
        "university mailing lists and online platforms (Prolific). Inclusion criteria: age 18+, "
        "native or fluent English speaker, self-reported interest in movies (watches at least "
        "one movie per month). Participants are compensated at the local minimum wage rate for "
        "their time (estimated 20 minutes total).", body))

    story.append(Paragraph("3.4 Procedure", h3))
    proc_data = [
        ['Step', 'Description'],
        ['1. Consent', 'Participant reads information sheet and provides informed consent'],
        ['2. Demographics', 'Short questionnaire: age, gender, movie-watching frequency'],
        ['3. Session A', 'Complete first interface (Design 1 or 2, counterbalanced)'],
        ['4. Break', 'At least 48 hours between sessions'],
        ['5. Session B', 'Complete second interface (the other design)'],
        ['6. Post-study survey', 'NASA-TLX cognitive load scale + satisfaction rating per design'],
        ['7. Ground truth', 'Participant rates 20 held-out movies on a 1-5 scale (ground truth)'],
    ]
    t = Table(proc_data, colWidths=[1.2*inch, 5.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)

    story.append(Paragraph("3.5 Evaluation Metrics", h3))
    story.append(Paragraph(
        "<b>Recommendation quality (primary):</b> Kendall's tau between the ranking of the 20 "
        "held-out movies predicted by the fitted w and the participant's ground-truth ratings. "
        "Higher tau = better preference recovery.", body))
    story.append(Paragraph(
        "<b>Cognitive load (secondary):</b> NASA-TLX composite score per design. "
        "Lower = less demanding.", body))
    story.append(Paragraph(
        "<b>User satisfaction (secondary):</b> Single 7-point Likert item "
        "'I found this interface enjoyable to use' per design.", body))
    story.append(Paragraph(
        "<b>Time per elicited comparison:</b> Total session time divided by the number of "
        "implicit pairwise comparisons collected. This measures efficiency.", body))

    story.append(Paragraph("3.6 Statistical Analysis", h3))
    story.append(Paragraph(
        "Primary hypothesis tested with a paired t-test on Kendall's tau scores across "
        "participants (one score per participant per design). We use alpha = 0.05 and report "
        "Cohen's d as an effect size measure. A power analysis with d = 0.5 and power = 0.8 "
        "suggests N = 34 participants; our target of 40 provides additional headroom for dropouts. "
        "Secondary hypotheses tested with Wilcoxon signed-rank tests (ordinal data).", body))

    story.append(Paragraph("3.7 Controls and Ethical Considerations", h3))
    story.append(Paragraph(
        "Movie sets shown in each session are disjoint to prevent familiarity effects. "
        "Movie selection is uniform random to avoid confounding by popularity. "
        "Participants may withdraw at any time without penalty. "
        "No personally identifying information is collected beyond demographics. "
        "Data is stored encrypted and deleted after publication.", body))

    story.append(PageBreak())

    # ── Summary ──
    story.append(Paragraph("4. Summary", h2))
    summary_data = [
        ['Component', 'Choice'],
        ['Feature vector', '4 numeric + 21 binary genre flags (25 dimensions total)'],
        ['Pairwise model', 'Bradley-Terry with linear utility, MLE via gradient ascent'],
        ['Ranking model', 'Plackett-Luce (Luce choice axiom extension of Bradley-Terry)'],
        ['Study design', 'Within-subjects, counterbalanced, N=40'],
        ['Primary metric', "Kendall's tau on 20 held-out movies"],
        ['Secondary metrics', 'NASA-TLX cognitive load, satisfaction, time efficiency'],
        ['Statistical test', 'Paired t-test (primary), Wilcoxon signed-rank (secondary)'],
    ]
    t = Table(summary_data, colWidths=[2.2*inch, 4.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#275CB2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(t)

    doc.build(story)
    print(f"Done! Report saved to: {out_path}")


if __name__ == '__main__':
    build_pdf()
