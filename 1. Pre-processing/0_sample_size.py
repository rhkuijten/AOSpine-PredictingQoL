import math

outcome_proportion = 0.375
abs_marg_error = 0.05
candidate_predictors = 10
MAPE = 0.05
shrinkage_factor = 0.9
R2_CS = 0.2
max_R2_CS = 0.33

# Calculation of sample size required for precise estimation of the overall outcome probability in the target population
n_1 = ((1.96/abs_marg_error)**2)*outcome_proportion*(1-outcome_proportion)
print("Step 1 (sample size required for precise estimation of overall outcome probability in target population): n = ", n_1)

# Sample size required to help ensure a developed prediction model of a binary outcome will have a small mean
#absolute error in predicted probabilities when applied in other targeted individuals
n_2 = math.exp((-0.508+0.259*math.log(outcome_proportion)+0.504*math.log(candidate_predictors)-math.log(MAPE))/(0.544))
print("Step 2 (sample size required to help ensure small mean absolute error in predicted probabilities): n = ", n_2)

# How to calculate the sample size needed to target a small magnitude of required shrinkage of predictor effects (to
# minimise potential model overfitting) for binary or time-to-event outcomes
n_3 = (candidate_predictors)/((shrinkage_factor-1)*math.log(1-((R2_CS)/(shrinkage_factor))))
print("Step 3 (Sample size needed to target small magnitude of required shrinkage of predictor effects): n = ", n_3)

S = R2_CS/(R2_CS+0.05*max_R2_CS)
n_4 = (candidate_predictors)/((S-1)*math.log(1-((R2_CS)/(S))))
print("Step 4 (Sample size needed to target small optimisim in model fit to minimise potential overfitting): n = ", n_4)
