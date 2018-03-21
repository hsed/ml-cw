# email classifier

## part a
### problem spam/not spam

- [ ] define dataset
- [x] define features and output
- [ ] show analysis of features e.g. pca variance analysis
- [ ] more importantly use online definitions and mathematical description
- [ ] define error function closely it is ROC AUC maximisation etc.
- [ ] use T P E ,method and.or andrew ng setup mthod and/or lecture slides lecture 2 pg 23, lect3 pg2
- [ ] structural risk min?? maybe later
- [ ] talk about doing max likelihood estimation
- [ ] for our data set y is assumed to be bernoullie (at least for linear model) so we best max liklihood estimator is cross entropy loss P(y|x) is assumed to be bernoulli
- [ ] do not derive formulas but only reference slides or online data which says that this is maximul likehood function etc
- [ ] maybe mention loss fn will depend on the model as well

## part b
- [ ] baseline: perceptron, loss func binary error training iterative maybe pocket algorithm
- [ ] logistic reg (try all these in one go so matrix selection):
    - [ ] normal training with imporved epoch sizes
    - [ ] add l1 norm, try different params
    - [ ] add l2 norm, try different params,
    - [ ] what is best reg for sparse dataset??
    - [ ] try with or without standardisation