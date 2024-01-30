import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from sys import exit

# 1. A function for the GEV CDF

def gev_cdf(x, mu, sigma, xi) :
 
    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # Using the standardized variable
    s = ( x - mu ) / sigma

    # Depend on the sign of shape parameter, CDF will be :
    if xi == 0. :
        c = np.exp( -1 * np.exp ( -1 * s ) )
    else :
        c = np.where( xi*s > -1, np.exp( -1 * (1 + xi * s)**(-1/xi) ), 0.)
        if xi > 0. :
            c = np.where( xi*s <= -1, 0, c)
        if xi < 0. :
            c = np.where( xi*s <= -1, 1, c)
    return c

# 2. A function for the GEV PDF

def gev_pdf(x, mu, sigma, xi) :
    
    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # Using the standardized variable
    s = ( x - mu ) / sigma

    # This is what I used in Homework 2
    # Depend on the sign of shape parameter, PDF will be :
    #if xi == 0. :
    #    p = np.exp( -1 * s ) * np.exp( -1 * np.exp ( -1 * s ) )
    #else :
    #    a = 1 + xi*s
    #    b = 1 + ( 1/xi)
    #    p = np.where( xi*s > -1, a**(-1*b) * np.exp(-1*(a)**(-1/xi)) , 0.)
        
    # This is the revised version based on Homework keys from Prof. Travis
    # And double-check the Wikipedia page:
    # https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    # The PDF formula provided in summary table of this website is correct
    # But the PDF formula discussed in Specification section is wrong (i.e., lack of * 1/sigma)
    
    if xi == 0:
        t = np.exp(-s)
        p = (t**(xi + 1)) * np.exp(-t) / sigma
    else:
        t = (1 + xi * s)**(-1 / xi)
        # Above line can lead to "RuntimeWarning: invalid value encountered in double_scalars"
        # According to
        # https://www.statology.org/runtimewarning-invalid-value-encountered-in-double_scalars/
        # this error occurs "when you attempt to perform some mathematical operation 
        # that involves extremely small or extremely large numbers 
        # and Python simply outputs a NaN value as the result.
        # To fix it: 
        # Typically the way to fix this type of error is to use a special function
        # from another library in Python that is capable of 
        # handling extremely small or extremely large values in calculations.
        p = np.where( xi*s > -1, (t**(xi + 1)) * np.exp(-t) / sigma , 0.)
            
    return p

# A function for the GEV quantile function
def gev_qf(q, mu, sigma, xi) :
 
    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # As "np.log" calculates the logarithm for all elements of "q",
    # even the ones that aren't selected by the "np.where"
    # And I don't want to use "for" loop, so :
    np.seterr(divide = 'ignore') 
    
    # Depend on the sign of shape parameter, quantile function will be :
    if xi == 0. :
        x = np.where( (q > 0) & (q < 1), mu - sigma * (np.log (-1 * np.log(q))), np.nan)
    else :
        x = np.where( (q > 0) & (q < 1), mu + (sigma/xi)*((-1*np.log(q))**(-1*xi)-1) , np.nan)
        if xi > 0. :
            x = np.where( q == 0, mu - (sigma/xi), x)
        if xi < 0. :
            x = np.where( q == 1, mu - (sigma/xi), x)
    return x

# A function for non-stationary GEV CDF
def gev_ns_cdf(x, t, cmu, mu0, sigma, xi) : 
 
    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # Use a non-stationary model that assume μ(t) = cμ ⋅ t + μ0
    mu = ( cmu * t ) + mu0
   
    # Use gev_cdf()
    c = gev_cdf(x, mu, sigma, xi)
    
    return c

# A function for non-stationary GEV PDF
def gev_ns_pdf(x, t, cmu, mu0, sigma, xi) :
    
    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # Use a non-stationary model that assume μ(t) = cμ ⋅ t + μ0
    mu = ( cmu * t ) + mu0

    # Use gev_pdf()
    p = gev_pdf(x, mu, sigma, xi)
    
    return p

# A function for non-stationary GEV quantile function (QF) 
def gev_ns_qf(q, t, cmu, mu0, sigma, xi) : 

    # Check if scale parameter is positive
    if sigma <= 0. :
       print ("Scale parameter should be positive !")
       exit()

    # Use a non-stationary model that assume μ(t) = cμ ⋅ t + μ0
    mu = ( cmu * t ) + mu0
        
    # Use gev_qf()
    x = gev_qf(q, mu, sigma, xi)
    
    return x

# A plotting function
def compare_cdf_qf(x,cdf,cdflb,x_qf,q,qflb,title) :    
    # Plotting
    plt.plot(x, cdf, c='g', label=cdflb)
    plt.plot(x_qf, q, c='r', label=qflb, linestyle='--')
    # Decoration
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    caption=title
    plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
# Another plotting function
def compare_cdf(x1,cdf_sta,lb1,x2,cdf_non,lb2,title) :    
    # Plotting
    plt.plot(x1, cdf_sta, c='g', label=lb1)
    plt.plot(x2, cdf_non, c='r', label=lb2, linestyle='--')
    # Decoration
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    caption=title
    plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
    
# 3-D plotting of "non-stationary" PDFs
def pdf_3d_plot(x, t, cmu, mu0, sigma, xi, title) :

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
 
    # Start with surface of PDF with all zero values
    pdf=np.zeros((len(t),len(x)))
    
    # Using my non-stationary GEV PDFs
    for i in np.arange(len(t)) :
        pdf[i,:] = gev_ns_pdf(x, t[i], cmu, mu0, sigma, xi) 
    
    # Plotting
    X,T = np.meshgrid(x,t)
    surf = ax.plot_surface(X, T, pdf, cmap=cm.coolwarm)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Decoration
    ax.set_xlabel('X', fontsize=18, rotation = 0)
    ax.set_ylabel('Time', fontsize=18, rotation = 0)
    ax.set_zlabel('Probability', fontsize=18, rotation = 0)
    #
    plt.figtext(0.5, 0.1, title, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# A plotting for "deeply" looking one case of PDF
def compare_pdf(x, t, cmu, mu0, sigma, xi, title) :
    
    # Get the stationary PDF
    pdf_sta = gev_pdf(x, mu0, sigma, xi)
    lb1 = 'Stationary PDF ('+'\u03BE = '+str(xi)+')'

    # Get the non-stationary PDF
    pdf_non = gev_ns_pdf(x, t, cmu, mu0, sigma, xi) 
    lb2 = 'Non-stationary PDF ('+'\u03BE = '+str(xi)+')'
        
    # Use a non-stationary model that assume μ(t) = cμ ⋅ t + μ0
    mu = ( cmu * t ) + mu0

    # Adding two test points
    
    ### 1st point
    ind_test = int(len(t)/3)
    mu_1=mu[ind_test]
    x_1=x[ind_test]
    # Get the stationary PDF at mu_test
    pdf_1 = gev_pdf(x, mu_1, sigma, xi)
    lb3 = 'Stationary PDF ('+'\u03BC = '+str(mu_1)+')'
    p_1 = pdf_1[ind_test]
    
    ### 2nd point
    ind_test = 2*int(len(t)/3)
    mu_2=mu[ind_test]
    x_2=x[ind_test]
    # Get the stationary PDF at mu_test
    pdf_2 = gev_pdf(x, mu_2, sigma, xi)
    lb4 = 'Stationary PDF ('+'\u03BC = '+str(mu_2)+')'
    p_2 = pdf_2[ind_test]
    
    # Plotting
    fig = plt.figure(figsize=(8,8))
    
    plt.plot(x, pdf_sta, c='g', label=lb1)
    plt.plot(x, pdf_non, c='r', label=lb2, linestyle='--')
    
    # Add "tested point" PDFs
    plt.plot(x, pdf_1, c='b', label=lb3, linestyle='--')
    plt.axvline(x=x_1, c='b', alpha=0.3)
    plt.annotate('Test 1', xy =(x_1, p_1), xytext =(x_1-1, p_1), \
                 arrowprops = dict(facecolor ='b',shrink = 0.05),)
    
    plt.plot(x, pdf_2, c='purple', label=lb4, linestyle='--')
    plt.axvline(x=x_2, c='purple', alpha=0.3)
    plt.annotate('Test 2', xy =(x_2, p_2), xytext =(x_2+1, p_2), \
                 arrowprops = dict(facecolor ='purple',shrink = 0.05),)
    
    # Decoration
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.figtext(0.5, 0.0, title, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

# 3-D plotting of "non-stationary" CDFs
def cdf_3d_plot(x, t, cmu, mu0, sigma, xi, title) :

    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
 
   
    # Start with surface of CDF with all zero values
    cdf=np.zeros((len(t),len(x)))
    
    # Using my non-stationary GEV CDF functions
    for i in np.arange(len(t)) :
        cdf[i,:] = gev_ns_cdf(x, t[i], cmu, mu0, sigma, xi) 
    
    # Plotting
    X,T = np.meshgrid(x,t)
    surf = ax.plot_surface(X, T, cdf, cmap=cm.coolwarm)
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Decoration
    ax.set_xlabel('X', fontsize=18, rotation = 0)
    ax.set_ylabel('Time', fontsize=18, rotation = 0)
    ax.set_zlabel('Probability', fontsize=18, rotation = 0)
    #
    plt.figtext(0.5, 0.1, title, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()