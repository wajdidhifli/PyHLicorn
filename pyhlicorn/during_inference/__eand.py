import numpy as np

def __eand(co,reg_disc_exp):
    if len(co) == 1:
        return reg_disc_exp[co]
    # If there are multiple co-regulators, calculate the combined activation state
    else:
        num = len(co)  # Number of co-regulators
        # Sum the expression values for the specified co-regulators across all columns
        y = reg_disc_exp[co].sum(axis=0)
        # Initialize an array of zeros to store the combined activation state
        x = np.zeros(len(y))
        # Set the activation state to -multip where the sum of expression values is -num (all repressed)
        x[np.where(y == -num)] = -1
        # Set the activation state to multip where the sum of expression values is num (all active)
        x[np.where(y == num)] = 1
        # Append the combined activation state to the list
        return x