import math
import numpy as np
def calculate_q1(d2, S11, S12, dT, S21, S22):
    numerator = -d2 * S11 + S12 - d2 * dT * S21 + dT * S22
    denominator = d2 * dT * S12
    q1_positive = math.sqrt(numerator / denominator)
    q1_negative = -q1_positive
    return q1_positive, q1_negative

def calculate_q2(S12, dT, S22, d2, d3, q1):
    numerator = (S12 + dT * S22)
    denominator = d2 * d3 * (1 + S12 * q1)
    q2 = -numerator / denominator
    return q2

def calculate_q3(q1, q2, d2, S11, S21, dT, d3):
    numerator = q1 + q2 + d2 * S11 * q1 * q2 + S21
    denominator = 1 + ((dT * q1 + d3 * q2) * S11) + (d2 * d3 * q2 * (S21 + q1))
    q3 = -numerator / denominator
    return q3

def calculate_q_values(alpha_x,beta_x,alpha_y,beta_y):
    beta = np.sqrt(beta_x*beta_y)
    alpha = (alpha_x + alpha_y)/2
    print(beta)
    print(alpha)
    # Example usage
    E=45.03 #MeV
    gamma=E/0.511
    d2 = 245e-3 # S1 to S2
    d3 = 382.5e-3 #  S1 to S3
    #d2= .348-0.101
    #d3= .481-0.101
    S11 = alpha  #alpha
    S12 = beta  # beta
    dT = d2+d3 # example value
    S21 = -(1+alpha**2)/beta  # beta
    S22 = -alpha  # example value
    S0=np.zeros((2,2))
    S0[0,0] = S11
    S0[0,1] = S12
    S0[1,0] = S21
    S0[1,1] = S22
    lq = 0.15 #m
    fac= (1/298.)*E
    print ("correlation matrix")
    print (S0)
    print('########Solution 1########\n')
    q1_positive, _ = calculate_q1(d2, S11, S12, dT, S21, S22)
    print("q1:", q1_positive, "Q1:",q1_positive*fac/lq,"current1:",q1_positive*fac/(0.7873*lq))

    q2 = calculate_q2(S12, dT, S22, d2, d3, q1_positive)
    print("q2:", q2,"Q2:",q2*fac/lq,"current2:",q2*fac/(0.7873*lq))

    q3 = calculate_q3(q1_positive, q2, d2, S11, S21, dT, d3)

    print("q3:", q3,"Q3:",q3*fac/(lq), "current3:",q3*fac/(lq*0.7873))

    print('########Solution 2########\n')
    _,q1_negative = calculate_q1(d2, S11, S12, dT, S21, S22)
    print("q1:", q1_negative, "Q1:",q1_negative*fac/lq)

    q22 = calculate_q2(S12, dT, S22, d2, d3, q1_negative)
    print("q2:", q22,"Q2:",q22*fac/lq,"current2:",q22*fac/(0.7873*lq))

    q32 = calculate_q3(q1_negative, q2, d2, S11, S21, dT, d3)
    print("q3:", q32,"Q2:",q32*fac/lq,"current3:",q32*fac/(lq*0.7873))

alpha_x = -0.00852503
beta_x = 3.4272496
alpha_y = -0.0722964657
beta_y= 4.26489807
calculate_q_values(alpha_x,beta_x,alpha_y,beta_y)