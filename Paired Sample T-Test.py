## Paired Sample T-test

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, norm

# Create mock data
before = norm.rvs(loc = 500, scale = 100, size = 100, random_state = 42).astype(int)

np.random.seed(42)
after = before + np.random.randint(low = -50, high = 75, size = 100)

plt.hist(before, density = True, alpha = 0.5, label="Before")
plt.hist(after, density = True, alpha = 0.5, label="After")
plt.legend()
plt.show()

before_mean = before.mean()
after_mean = after.mean()
print(before_mean, after_mean)

# set the hypothesis and acceptance criteria

null_hypothesis = "The mean of the before sample is equal to the mean of the after sample"
alternate_hypothesis = "The mean of the before sample is differetn to the mean of after sample"
acceptance_criteria = 0.5

# Execute the hypothesis test

t_statistic, p_value = ttest_rel(before, after)
print(t_statistic, p_value)

# Print the results

if p_value <= acceptance_criteria:
    print(f"AS our p-value of {p_value} is lower than our acceptance criteria of {acceptance_criteria} - we reject the null hypothesis, and conclude that: The mean of the before sample is different to the mean of the after sample.")
else:
    print(f"AS our p-value of {p_value} is higher than our acceptance criteria of {acceptance_criteria} - we retain the null hypothesis, and conclude that: The mean of the before sample is the same to the mean of the after sample.")
    
