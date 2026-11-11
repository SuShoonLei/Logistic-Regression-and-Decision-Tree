# Logistic Regression and Decision Tree

## How to run

```
python3 -m venv venv
source venv/bin/activate
pip install numpy scikit-learn
python logistic_regression.py
```

Part 2 takes a little while to run because gradient ascent does about 19,500 updates.

## Implementation description

The code is in `logistic_regression.py` and is split up by part. Everything is fit and scored on all 569 rows, with no train/test split and no scaling of x or y.

- **Part 1:** Loads the breast cancer dataset, uses raw `mean radius` as `x`, and recodes `y` so that 1 = malignant and 0 = benign (`(cancer.target == 0).astype(int)`).
- **Part 2:** `gradient_ascent` maximizes the mean log-likelihood. Each iteration computes the probabilities `p` once with the sigmoid, then updates both parameters at the same time:
  ```python
  p = sigmoid(beta_0 + beta_1 * x)
  new_beta_0 = beta_0 + learning_rate * np.mean(y - p)
  new_beta_1 = beta_1 + learning_rate * np.mean((y - p) * x)
  ```
  The log-likelihood is calculated with `np.mean(y * z - np.logaddexp(0, z))`, which is a stable way to compute the same thing. After each update it stops if the parameters are not finite (numerical failure), if the change in log-likelihood is less than the tolerance, or if it hits the iteration limit. The probabilities are then calculated by hand with the sigmoid.
- **Part 3:** scikit-learn `LogisticRegression` with `C=np.inf` (no regularization). Its mean log-likelihood is calculated with the same function as Part 2.
- **Part 4:** `DecisionTreeClassifier` with `max_depth=1`, so it makes one split. The threshold is learned by the tree. `tree.apply` is used to find which leaf each row is in.
- **Part 5:** Compares the two logistic fits, then turns each model's probabilities into labels (1 if p >= 0.5) and counts the correct predictions. Nothing is refit.

## Dataset summary

| Item | Value |
|---|---|
| Observations | 569 |
| Malignant (y=1) | 212 |
| Benign (y=0) | 357 |
| Mean of x (mean radius) | 14.1272917399 |
| First five x | 17.99, 20.57, 19.69, 11.42, 20.29 |
| First five y | 1, 1, 1, 1, 1 |

## Outputs

### Part 2: Gradient ascent

| Item | Value |
|---|---|
| Final intercept | -11.2520664419 |
| Final coefficient | 0.7548847702 |
| Final mean log-likelihood | -0.3001729741 |
| Completed updates | 19583 |
| Stopping reason | tolerance |
| First five probabilities | 0.9111947252, 0.9862911946, 0.9737020582, 0.0671520752, 0.9831190681 |

### Part 3: scikit-learn logistic regression

| Item | Value |
|---|---|
| Intercept | -15.2458707775 |
| Coefficient | 1.0335888217 |
| Mean log-likelihood | -0.2899919543 |
| First five probabilities | 0.9660521444, 0.9975642244, 0.9939731944, 0.0309997289, 0.9967493567 |

### Part 4: Decision tree with one split

| Item | Value |
|---|---|
| Learned threshold | 15.0450000763 |
| Leaf x <= threshold | 397 observations, P(malignant) = 0.1284634761 |
| Leaf x > threshold | 172 observations, P(malignant) = 0.9360465116 |
| First five probabilities | 0.9360465116, 0.9360465116, 0.9360465116, 0.1284634761, 0.9360465116 |

**Question: How do the tree and logistic model represent the predictor-probability relationship differently?**

Logistic regression gives a smooth S-curve, so the probability goes up gradually as radius goes up and every row gets its own probability. The tree gives a step function with only two probabilities (0.128 if radius <= 15.045, and 0.936 if above). Everything in the same leaf gets the same probability, and it jumps at the threshold.

### Part 5A: Comparing the logistic implementations

| Quantity | Gradient ascent | sklearn | Difference |
|---|---|---|---|
| Intercept | -11.2520664419 | -15.2458707775 | 3.9938043356 |
| Coefficient | 0.7548847702 | 1.0335888217 | 0.2787040515 |
| Mean log-likelihood | -0.3001729741 | -0.2899919543 | 0.0101810198 |

Predicted probabilities: max absolute difference = 0.0909803728, mean absolute difference = 0.0374521879.

**1. What is similar about the fitted parameters, mean log-likelihoods, and predicted probabilities?**

Both have a negative intercept and a positive coefficient, so the probability of malignancy goes up with radius. The mean log-likelihoods are close (-0.300 vs -0.290), and the probabilities are close too, with an average difference of only 0.037.

**2. What differs? How might the fitting procedures and stopping rules explain those differences?**

The intercept and coefficient are pretty different (about 4.0 and 0.28), and sklearn has a higher log-likelihood. Gradient ascent uses a fixed learning rate and stops as soon as one step changes the log-likelihood by less than 1e-6. Since x is not scaled, the intercept and coefficient are hard to optimize together and gradient ascent moves very slowly near the end. sklearn's lbfgs solver is more advanced and uses a much smaller tolerance (1e-10), so it gets closer to the best answer.

**3. What do the differences tell you about the fit reached by your gradient ascent run? Does meeting its stopping rule mean it has reached the best-fitting parameters?**

My gradient ascent got close to the best fit but did not fully reach it, since its log-likelihood is about 0.01 lower than sklearn's. Meeting the stopping rule does not mean it found the best parameters. It only means the log-likelihood barely changed in one step, which can happen when progress is just very slow.

### Part 5B: Training accuracy

| Approach | Correct | Training accuracy |
|---|---|---|
| Gradient ascent | 501 | 0.8804920914 |
| sklearn logistic | 500 | 0.8787346221 |
| Decision tree | 507 | 0.8910369069 |

Highest accuracy: decision tree. Rows where the two logistic models predict different classes: 9.

**1. Can one model produce a higher log-likelihood but a lower accuracy? Why?**

Yes. sklearn has the higher log-likelihood but one fewer correct prediction than gradient ascent (500 vs 501). Log-likelihood looks at how good the probabilities are, while accuracy only checks which side of 0.5 each prediction is on. A model can have better probabilities overall but still get a few more rows on the wrong side of 0.5.

**2. Does this training comparison establish which approach will predict new samples better? How would you evaluate that?**

No, because all three models were tested on the same data they were trained on. To see how they do on new samples, I would use a train/test split or cross-validation and compare their accuracy on the data they were not trained on.

## AI usage

Chat links: https://claude.ai/share/d781ce3c-7f5e-439c-8c93-f07e899b2f48