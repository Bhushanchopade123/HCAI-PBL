import os
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score


def index(request):
    return render(request, 'project1/index.html')


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def upload_data(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        filepath = fs.path(filename)
        request.session['csv_path'] = filepath

        df = pd.read_csv(filepath)
        features = list(df.columns[:-1])
        label = df.columns[-1]

        context['features'] = features
        context['label'] = label
        context['n_rows'] = df.shape[0]
        context['n_features'] = df.shape[1] - 1
        context['head'] = df.head(5).to_html(classes='table', index=False)

        n_unique = df[label].nunique()
        problem_type = 'classification' if n_unique <= 20 else 'regression'
        context['problem_type'] = problem_type

        feat_x = request.POST.get('feat_x', features[0])
        feat_y = request.POST.get('feat_y', features[1] if len(features) > 1 else features[0])
        context['feat_x'] = feat_x
        context['feat_y'] = feat_y

        fig, ax = plt.subplots(figsize=(7, 5))
        if problem_type == 'classification':
            for cls in df[label].unique():
                subset = df[df[label] == cls]
                ax.scatter(subset[feat_x], subset[feat_y], label=str(cls), alpha=0.7)
            ax.legend(title=label)
        else:
            sc = ax.scatter(df[feat_x], df[feat_y], c=df[label], cmap='viridis', alpha=0.7)
            plt.colorbar(sc, ax=ax, label=label)

        ax.set_xlabel(feat_x)
        ax.set_ylabel(feat_y)
        ax.set_title(f'{feat_x} vs {feat_y}')
        context['plot'] = fig_to_base64(fig)

    return render(request, 'project1/upload.html', context)


def train_model(request):
    context = {}
    csv_path = request.session.get('csv_path')

    if not csv_path or not os.path.exists(csv_path):
        context['error'] = 'No dataset loaded. Please upload a CSV first.'
        return render(request, 'project1/train.html', context)

    df = pd.read_csv(csv_path)
    features = list(df.columns[:-1])
    label = df.columns[-1]
    n_unique = df[label].nunique()
    problem_type = 'classification' if n_unique <= 20 else 'regression'
    context['problem_type'] = problem_type

    X = df[features].values
    y = df[label].values

    if request.method == 'POST':
        model_name = request.POST.get('model', 'knn')
        test_size = float(request.POST.get('test_size', 0.2))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scores = []
        hyperparams = []

        if problem_type == 'classification':
            if model_name == 'knn':
                for k in range(1, 16):
                    clf = KNeighborsClassifier(n_neighbors=k)
                    clf.fit(X_train, y_train)
                    scores.append(accuracy_score(y_test, clf.predict(X_test)))
                    hyperparams.append(k)
                xlabel = 'k (neighbors)'
            elif model_name == 'svm':
                for C in [0.01, 0.1, 1, 10, 100]:
                    clf = SVC(C=C)
                    clf.fit(X_train, y_train)
                    scores.append(accuracy_score(y_test, clf.predict(X_test)))
                    hyperparams.append(C)
                xlabel = 'C (regularization)'
            elif model_name == 'tree':
                for d in range(1, 16):
                    clf = DecisionTreeClassifier(max_depth=d)
                    clf.fit(X_train, y_train)
                    scores.append(accuracy_score(y_test, clf.predict(X_test)))
                    hyperparams.append(d)
                xlabel = 'Max depth'
            ylabel = 'Accuracy'
        else:
            if model_name == 'knn':
                for k in range(1, 16):
                    reg = KNeighborsRegressor(n_neighbors=k)
                    reg.fit(X_train, y_train)
                    scores.append(r2_score(y_test, reg.predict(X_test)))
                    hyperparams.append(k)
                xlabel = 'k (neighbors)'
            elif model_name == 'svm':
                for C in [0.01, 0.1, 1, 10, 100]:
                    reg = SVR(C=C)
                    reg.fit(X_train, y_train)
                    scores.append(r2_score(y_test, reg.predict(X_test)))
                    hyperparams.append(C)
                xlabel = 'C (regularization)'
            elif model_name == 'tree':
                for d in range(1, 16):
                    reg = DecisionTreeRegressor(max_depth=d)
                    reg.fit(X_train, y_train)
                    scores.append(r2_score(y_test, reg.predict(X_test)))
                    hyperparams.append(d)
                xlabel = 'Max depth'
            ylabel = 'R² Score'

        best_idx = scores.index(max(scores))
        context['best_param'] = hyperparams[best_idx]
        context['best_score'] = round(max(scores), 4)
        context['model_name'] = model_name.upper()
        context['test_size'] = test_size

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hyperparams, scores, marker='o', color='steelblue')
        ax.axvline(x=hyperparams[best_idx], color='red', linestyle='--',
                   label=f'Best: {hyperparams[best_idx]}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{model_name.upper()} performance')
        ax.legend()
        context['score_plot'] = fig_to_base64(fig)

    return render(request, 'project1/train.html', context)