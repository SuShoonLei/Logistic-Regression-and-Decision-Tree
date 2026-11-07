"""Malignant vs benign from mean radius: gradient ascent, sklearn logistic, and a one-split tree."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# Shared helpers

def sigmoid(z):
    """Logistic function."""
    return 1 / (1 + np.exp(-z))


def mean_log_likelihood(x, y, beta_0, beta_1):
    """Mean Bernoulli log-likelihood with a logit link (natural log)."""
    z = beta_0 + beta_1 * x
    return np.mean(y * z - np.logaddexp(0, z))


def gradient_ascent(x, y, learning_rate, tolerance, max_iterations):
    """Batch gradient ascent on the mean log-likelihood. No library optimizer."""
    rng = np.random.default_rng(42)
    beta_0, beta_1 = rng.normal(loc=0.0, scale=0.1, size=2)

    # The initial value is recorded, but it is not an update.
    history = [mean_log_likelihood(x, y, beta_0, beta_1)]
    updates = 0
    reason = "iteration limit"

    for _ in range(max_iterations):
        p = sigmoid(beta_0 + beta_1 * x)
        new_beta_0 = beta_0 + learning_rate * np.mean(y - p)
        new_beta_1 = beta_1 + learning_rate * np.mean((y - p) * x)

        L_old = history[-1]
        beta_0, beta_1 = new_beta_0, new_beta_1
        L_new = mean_log_likelihood(x, y, beta_0, beta_1)
        history.append(L_new)
        updates += 1

        if not np.all(np.isfinite([beta_0, beta_1, L_new])):
            reason = "numerical failure"
            break
        if abs(L_new - L_old) < tolerance:
            reason = "tolerance"
            break

    return beta_0, beta_1, history, updates, reason


def main():
    # Part 1: Prepare the dataset
    print("PART 1: Prepare the dataset")
    cancer = load_breast_cancer()
    radius_column = list(cancer.feature_names).index("mean radius")
    x = cancer.data[:, radius_column]
    y = (cancer.target == 0).astype(int)   # 1 = malignant, 0 = benign

    n_observations = len(y)
    print(f"Number of observations: {n_observations}")
    print(f"Malignant (y=1): {int(np.sum(y == 1))}")
    print(f"Benign (y=0): {int(np.sum(y == 0))}")
    print(f"Mean of x: {np.mean(x):.10f}")
    print("First five x: " + ", ".join(f"{value:.10f}" for value in x[:5]))
    print("First five y: " + ", ".join(str(int(value)) for value in y[:5]))

    # Part 2: Gradient ascent (implemented manually)
    print("\nPART 2: Gradient ascent")
    ga_intercept, ga_coefficient, ga_history, ga_updates, ga_reason = gradient_ascent(
        x, y, learning_rate=0.03, tolerance=0.000001, max_iterations=1000000
    )
    print(f"Final intercept: {ga_intercept:.10f}")
    print(f"Final coefficient: {ga_coefficient:.10f}")
    print(f"Final mean log-likelihood: {ga_history[-1]:.10f}")
    print(f"Completed updates: {ga_updates}")
    print(f"Stopping reason: {ga_reason}")

    z = ga_intercept + ga_coefficient * x
    ga_probabilities = 1 / (1 + np.exp(-z))
    print(
        "First five ga_probabilities: "
        + ", ".join(f"{value:.10f}" for value in ga_probabilities[:5])
    )

    # Part 3: Built-in logistic regression
    print("\nPART 3: Built-in logistic regression")
    try:
        model = LogisticRegression(
            C=np.inf, solver="lbfgs", tol=1e-10, max_iter=10000
        )
        model.fit(x.reshape(-1, 1), y)
    except ValueError:
        model = LogisticRegression(
            penalty=None, solver="lbfgs", tol=1e-10, max_iter=10000
        )
        model.fit(x.reshape(-1, 1), y)

    intercept = model.intercept_[0]
    coefficient = model.coef_[0, 0]
    probabilities = model.predict_proba(x.reshape(-1, 1))[:, 1]
    sklearn_mean_log_likelihood = mean_log_likelihood(x, y, intercept, coefficient)

    print(f"Intercept: {intercept:.10f}")
    print(f"Coefficient: {coefficient:.10f}")
    print(f"Mean log-likelihood: {sklearn_mean_log_likelihood:.10f}")
    print(
        "First five probabilities: "
        + ", ".join(f"{value:.10f}" for value in probabilities[:5])
    )

    # Part 4: Decision tree with one binary split
    print("\nPART 4: Decision tree with one binary split")
    tree = DecisionTreeClassifier(criterion="log_loss", max_depth=1, random_state=42)
    tree.fit(x.reshape(-1, 1), y)
    split = tree.tree_.threshold[0]
    tree_probabilities = tree.predict_proba(x.reshape(-1, 1))[:, 1]
    leaf_ids = tree.apply(x.reshape(-1, 1))

    print(f"Learned threshold: {split:.10f}")
    # Left child is the x <= threshold leaf; right child is x > threshold.
    for leaf_label, leaf_id in (
        ("x <= threshold", tree.tree_.children_left[0]),
        ("x > threshold", tree.tree_.children_right[0]),
    ):
        in_leaf = leaf_ids == leaf_id
        leaf_probability = tree_probabilities[in_leaf][0]
        print(
            f"Leaf ({leaf_label}): observations = {int(np.sum(in_leaf))}, "
            f"predicted probability of malignancy = {leaf_probability:.10f}"
        )
    print(
        "First five tree_probabilities: "
        + ", ".join(f"{value:.10f}" for value in tree_probabilities[:5])
    )

    # Part 5A: Compare the two logistic implementations
    print("\nPART 5A: Compare the two logistic implementations")
    print(
        f"Intercept: gradient ascent = {ga_intercept:.10f}, "
        f"sklearn = {intercept:.10f}, "
        f"absolute difference = {abs(ga_intercept - intercept):.10f}"
    )
    print(
        f"Coefficient: gradient ascent = {ga_coefficient:.10f}, "
        f"sklearn = {coefficient:.10f}, "
        f"absolute difference = {abs(ga_coefficient - coefficient):.10f}"
    )
    print(
        f"Mean log-likelihood: gradient ascent = {ga_history[-1]:.10f}, "
        f"sklearn = {sklearn_mean_log_likelihood:.10f}, "
        f"difference (sklearn - gradient ascent) = "
        f"{sklearn_mean_log_likelihood - ga_history[-1]:.10f}"
    )
    probability_gap = np.abs(ga_probabilities - probabilities)
    print(
        "Predicted probabilities: "
        f"maximum absolute difference = {np.max(probability_gap):.10f}, "
        f"mean absolute difference = {np.mean(probability_gap):.10f}"
    )

    # Part 5B: Training accuracy
    print("\nPART 5B: Training accuracy")
    named_probabilities = (
        ("gradient ascent", ga_probabilities),
        ("sklearn logistic", probabilities),
        ("decision tree", tree_probabilities),
    )
    predicted_labels = {}
    correct_counts = {}
    for name, probs in named_probabilities:
        labels = (probs >= 0.5).astype(int)
        correct = int(np.sum(labels == y))
        predicted_labels[name] = labels
        correct_counts[name] = correct
        print(
            f"{name}: correct = {correct}, "
            f"training accuracy = {correct / n_observations:.10f}"
        )

    best_correct = max(correct_counts.values())
    winners = [name for name, correct in correct_counts.items() if correct == best_correct]
    if len(winners) == 1:
        print(f"Highest accuracy: {winners[0]}")
    else:
        print("Highest accuracy: tie between " + ", ".join(winners))

    n_disagree = int(np.sum(predicted_labels["gradient ascent"] != predicted_labels["sklearn logistic"]))
    print(f"Rows where gradient ascent and sklearn logistic labels disagree: {n_disagree}")


if __name__ == "__main__":
    main()
