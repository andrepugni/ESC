import scipy
import numpy as np
import math

"""
This code is based on Geifman et al. 2017 implementation.
You can find the original code here:https://github.com/geifmany/selective_deep_learning/blob/master/risk_control.py
"""


class risk_control:
    def calculate_bound(self, delta, m, erm):
        # This function is a solver for the inverse of binomial CDF based on binary search.
        precision = 1e-7

        def func(b):
            return (-1 * delta) + scipy.stats.binom.cdf(int(m * erm), m, b)

        a = erm  # start binary search from the empirical risk
        c = 1  # the upper bound is 1
        b = (a + c) / 2  # mid point
        funcval = func(b)
        while abs(funcval) > precision:
            if a == 1.0 and c == 1.0:
                b = 1.0
                break
            elif funcval > 0:
                a = b
            else:
                c = b
            b = (a + c) / 2
            funcval = func(b)
        return b

    def bound(self, rstar, delta, kappa_cal, residuals_cal, kappa_test, residuals_test):
        # A function to calculate the risk bound proposed in the paper, the algorithm is based on algorithm 1 from the paper.
        # Input: rstar - the requested risk bound
        #       delta - the desired delta
        #       kappa - rating function over the points (higher values is more confident prediction)
        #       residuals - a vector of the residuals of the samples 0 is correct prediction and 1 corresponding to an error
        #       split - is a boolean controls whether to split train and test
        # Output - [theta, bound] (also prints latex text for the tables in the paper)

        # when spliting to train and test this represents the fraction of the validation size
        valsize = 0.5

        probs = kappa_cal
        FY = residuals_cal
        FY_val = residuals_test
        probs_val = kappa_test
        m = len(FY)

        probs_idx_sorted = np.argsort(probs)

        a = 0
        b = m - 1
        deltahat = delta / math.ceil(math.log2(m))

        for q in range(math.ceil(math.log2(m)) + 1):
            # the for runs log(m)+1 iterations but actually the bound calculated on only log(m) different candidate thetas
            mid = math.ceil((a + b) / 2)
            mi = len(FY[probs_idx_sorted[mid:]])
            theta = probs[probs_idx_sorted[mid]]
            risk = sum(FY[probs_idx_sorted[mid:]]) / mi
            try:
                testrisk = sum(FY_val[probs_val >= theta]) / len(
                    FY_val[probs_val >= theta]
                )
            except ZeroDivisionError:
                testrisk = 1
            try:
                testcov = len(FY_val[probs_val >= theta]) / len(FY_val)
            except:
                testcov = 0
            bound = self.calculate_bound(deltahat, mi, risk)
            coverage = mi / m
            if bound > rstar:
                a = mid
            else:
                b = mid
        # print(
        #         "%.2f & %.4f & %.4f & %.4f & %.4f & %.4f  \\\\"
        #         % (rstar, risk, coverage, testrisk, testcov, bound)
        #     )
        return theta, bound, testrisk, testcov
