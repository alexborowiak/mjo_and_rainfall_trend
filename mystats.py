# All of these functions feed into the bottom to calculate the p-value for the mann-kendall test. Notes
# on this can be found in section 2 of your hand-written notes 





# Someone else code
# https://github.com/mps9506/Mann-Kendall-Trend/blob/master/mk_test.py


# Guide
# https://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

# Paper on the test that was referenced in Andrews text book
# https://pdf-sciencedirectassets-com.ezp.lib.unimelb.edu.au/271842/1-s2.0-S0022169409X00028/1-s2.0-S0022169408005787/main.pdf?X-Amz-Security-Token=AgoJb3JpZ2luX2VjEEEaCXVzLWVhc3QtMSJHMEUCIQC%2BgafAjE%2BA8EnQXkkXF8LhgWx76TCsGWjWRHRbeXLjRQIgcPemLsgQSvPUUC4xfLuliC4DrQMz9MmOKetDGb5xNdQq4wMImv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgwwNTkwMDM1NDY4NjUiDOgtXZ3v3oZsI2BNfCq3A%2F7PIeAEoR8mS7kPthCx41EgUIo4j4021XWkezn6g2E5PvE1Sz3floMw8iWkKhvFJBRVofbbBKUWu%2BUVFPR4o28J7LHcX897I%2FUWKp15jOYOGomooZFcGw%2Fr0XnHRlKfdpd0NEdaIpejd9pTErS8%2FI4H%2Bf6aqIYsQroIq6%2Bf1yapaJzpasC9E6bzAdBsD84kUWGFPSL%2F7FDBn03ZwOIPZXMekc9GVBdj2oYE9PPSQqrYZfPR2hhDB61EacmT7%2B7Eqrl2AKmkAMYeXp7MJeuENwAem5QVHafb6uaTnyH2yBm6D1v50%2FUjft3OQnHU4RpvoyF3HEGL8WjiWXW%2FHPSG1oVOgbo9A0WTSVJivrXk9b9w%2FjcISXxRZl8Ptf9Mq1%2BzFHQO4JUolrkPhiFnmUedhpwWfRz%2BbVW3hJuloOR1kI%2BLiw9w9kthWdH3Chn2CBKIC3P26%2FpNAxbNNAsUreqwr8JkNObN6s0zcDaAGpUrMbCl1HCosVyfEi4f9LmQCRrt09Z7Q95LQrRWkIz%2BZ3V%2B3QzXsxrVJM%2FY9JBVdSErwHl0u3poxo7aasLdyhWdEgzZNnyDEOYsFbEw%2BYea6QU6tAEobaIGSgYpHHhX7hKQ%2FfKb4I2GYkBASK7DQDyzoVQmnAuPXv1T1x%2FSfwDpcbCQky3VgKDWPqKf%2B1sifCSy6G7kU4nhqHCGCmRAkEjrQ0wJPODAyBorXCZtkJtpYzJYymg5bAOpj1G1VtGh9b52UGMmZrM%2F4DPt%2FXiv8aLSpWetDhZRvu7isewzafIxRf1v7MFUWmy2IxcYc%2B09qv4vQ2hz1eJmdDGSK3BJX5Tmwo1P1WsJu4E%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190711T012019Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7BKKOEEP%2F20190711%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d0e4e40feb1ef7c74c1a0a992e0016cdca18e24a6673e3f45f5011d0d36c9e4b&hash=67fd743a3b4a4d7fff889b5960a8e1506cdf7ae726332c3884fc00cb9e489047&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0022169408005787&tid=spdf-2f5713d9-f87d-42a0-88bc-015c4c6897be&sid=74c57e9c7e5215466838746-2595860e9bd7gxrqa&type=clienthttps://pdf-sciencedirectassets-com.ezp.lib.unimelb.edu.au/271842/1-s2.0-S0022169409X00028/1-s2.0-S0022169408005787/main.pdf?X-Amz-Security-Token=AgoJb3JpZ2luX2VjEEEaCXVzLWVhc3QtMSJHMEUCIQC%2BgafAjE%2BA8EnQXkkXF8LhgWx76TCsGWjWRHRbeXLjRQIgcPemLsgQSvPUUC4xfLuliC4DrQMz9MmOKetDGb5xNdQq4wMImv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgwwNTkwMDM1NDY4NjUiDOgtXZ3v3oZsI2BNfCq3A%2F7PIeAEoR8mS7kPthCx41EgUIo4j4021XWkezn6g2E5PvE1Sz3floMw8iWkKhvFJBRVofbbBKUWu%2BUVFPR4o28J7LHcX897I%2FUWKp15jOYOGomooZFcGw%2Fr0XnHRlKfdpd0NEdaIpejd9pTErS8%2FI4H%2Bf6aqIYsQroIq6%2Bf1yapaJzpasC9E6bzAdBsD84kUWGFPSL%2F7FDBn03ZwOIPZXMekc9GVBdj2oYE9PPSQqrYZfPR2hhDB61EacmT7%2B7Eqrl2AKmkAMYeXp7MJeuENwAem5QVHafb6uaTnyH2yBm6D1v50%2FUjft3OQnHU4RpvoyF3HEGL8WjiWXW%2FHPSG1oVOgbo9A0WTSVJivrXk9b9w%2FjcISXxRZl8Ptf9Mq1%2BzFHQO4JUolrkPhiFnmUedhpwWfRz%2BbVW3hJuloOR1kI%2BLiw9w9kthWdH3Chn2CBKIC3P26%2FpNAxbNNAsUreqwr8JkNObN6s0zcDaAGpUrMbCl1HCosVyfEi4f9LmQCRrt09Z7Q95LQrRWkIz%2BZ3V%2B3QzXsxrVJM%2FY9JBVdSErwHl0u3poxo7aasLdyhWdEgzZNnyDEOYsFbEw%2BYea6QU6tAEobaIGSgYpHHhX7hKQ%2FfKb4I2GYkBASK7DQDyzoVQmnAuPXv1T1x%2FSfwDpcbCQky3VgKDWPqKf%2B1sifCSy6G7kU4nhqHCGCmRAkEjrQ0wJPODAyBorXCZtkJtpYzJYymg5bAOpj1G1VtGh9b52UGMmZrM%2F4DPt%2FXiv8aLSpWetDhZRvu7isewzafIxRf1v7MFUWmy2IxcYc%2B09qv4vQ2hz1eJmdDGSK3BJX5Tmwo1P1WsJu4E%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20190711T012019Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7BKKOEEP%2F20190711%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d0e4e40feb1ef7c74c1a0a992e0016cdca18e24a6673e3f45f5011d0d36c9e4b&hash=67fd743a3b4a4d7fff889b5960a8e1506cdf7ae726332c3884fc00cb9e489047&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0022169408005787&tid=spdf-2f5713d9-f87d-42a0-88bc-015c4c6897be&sid=74c57e9c7e5215466838746-2595860e9bd7gxrqa&type=client


import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st

def S_kendall(data):
    S = 0
    n = len(data)
    
    # Outer sum
    for i in np.arange(0, n - 1):
        # Inner sum
        for k in np.arange(i + 1, n):

            S += np.sign(data[k]  - data[i])

    return S


##################

def var_kendall(data):
    unique_vals, count_vals = np.unique(data, return_counts = True)
    
    n = len(data)
    var = n*(n - 1)* (2*n +5)/ 18
    
    ######
    if any(count_vals > 1): # there is a repated value
        
        # These are the number of samples that have been used multiple times
        multi_sampled = count_vals[np.where(count_vals > 1)] 
        
        # This following is doing the sum as seen in the varience equation
        summed = 0
        for i in multi_sampled:
            summand = i * (i - 1) * (2 * i + 5)

            summed += summand
            
        # Divided by 18 and subtract from var
        
        var = var - summed /18

                                                                        
    #####
    else: # Don't need to do anything if the values are not repeated
        pass
        
    
    return var


##################
def Z_kendall(S,var):
    
    if S > 0:
        S = S-1
    else:
        S = S +1
        
    Z = S/np.sqrt(var)
    
    return Z


##################


def mann_kendall(data, return_all = False):
    
    # Calculates the s value
    S = S_kendall(data)
    
    # Calculates the varience, does both repeated and non-repeated values
    var = var_kendall(data)
    
    # The z value
    Z = Z_kendall(S,var)
    
    # The p-value form the normal distribution
    p_val = 2 * (1 - st.norm.cdf(abs(Z)))  # two tail test
    # Not really sure where above comes from, but it is included in the other person function
    # and seems to make more sense

#     p_val = st.norm.cdf(Z)
    
    # In case I want to check what the z-value is
    if return_all:
        return S,var, Z, p_val
    else:
        return p_val