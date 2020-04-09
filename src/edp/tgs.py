#####################################################################################################################
##### Methods for transient ground strain
##### Barry Zheng (Slate Geotechnical Consultants)
##### Updated: 2020-04-06
#####################################################################################################################


#####################################################################################################################
### Newmark (1967)
#####################################################################################################################
def newmark_1967(**kwargs):
    
    ## inputs
    Vmax = kwargs.get('Vmax',None) # peak ground velocity
    
    ## case 1: shear wave
    Vs = kwargs.get('Vs',None) # shear wave velocity
    gamma_s = kwargs.get('gamma_s',None) # deg, propagating angle relative to vertical axis
    
    ## case 2: Rayleigh wave
    Cr = kwargs.get('Cr',None) # propagation or phase velocity of Rayleigh wave
    
    ## ground strain parallel to pipe axis
    if Vs is not None:
        # shear wave propagation
        eps_g = Vmax/Vs*np.sin(np.radians(gamma_s))*np.cos(np.radians(gamma_s)) # eq. 10.1
    elif Cr is not None:
        # Rayleigh wave traveling parallel to pipe axis
        eps_g = Vmax/Cr # eq. 10.2
        
    ##
    return eps_g
	

#####################################################################################################################
### Shinozuka and Koike (1979)
#####################################################################################################################
def shinozuka_koike_1979(**kwargs):
    
    ## inputs
    egs_g = kwargs.get('egs_g',None) # peak ground velocity
    D = kwargs.get('D',None) # outer diameter of pipe
    l = kwargs.get('l',None) # wavelength
    t = kwargs.get('t',None) # pipe wall thickness
    E = kwargs.get('E',None) # modulus of elasticity
    G = kwargs.get('G',None) # shear modulus
    
    ## ground to pipe conversion factor
    beta_0 = kwargs.get('beta_0',None)
    if beta_0 is None:
        A = np.pi*(D**2 - (D-2*t)**2) # cross-sectional area of pipe
        Kg = 2*np.pi*G # equivalent spring constant to reflect soil-structural interaction
        beta_0 = 1/(1 + (2*np.pi/l)**2 * A*E/Kg) # eq. 10.5
    
    ## shear strain at soil-pipe interface
    gamma_0 = kwargs.get('gamma_0',None)
    if gamma_0 is None:
        gamma_0 = 2*np.pi/l*E*t/G*eps_g*beta_0 # eq. 10.6
        
    ## critical shear strain, default = 1e-3
    ## if gamma_0 <= gamma_cr, no slippage
    ## if gamma_0 > gamma_cr, slips
    gamma_cr = kwargs.get('gamma_cr',1.0e-3)
    
    ## ground to pipe conversion factor, for large ground movement, i.e., gamma_0 > gamma_cr
    beta_c = gamma_cr/gamma_0*q*beta_0 # eq. 10.8
    
    ## pipe axial strain
    eps_p = beta_c*eps_g # eq. 10.9
        
    ##
    return eps_p
	
	
#####################################################################################################################
### O'Rourke and El Hmadi (1988)
#####################################################################################################################
def orourke_elhmadi_1988(**kwargs):
    
    ## inputs
    egs_g = kwargs.get('egs_g',None) # peak ground velocity
    D = kwargs.get('D',None) # outer diameter of pipe
    l = kwargs.get('l',None) # wavelength
    t = kwargs.get('t',None) # pipe wall thickness
    E = kwargs.get('E',None) # modulus of elasticity
    tu = kwargs.get('tu',None) # maximum frictional resistance at shear interface
    
    # cross-sectional area of pipe
    A = np.pi*(D**2 - (D-2*t)**2)
    
    ## strain due to friction forces acting over 1/4 of wavelength
    ## controls when ground strain becomes large
    eps_f = tu*(l/4)/(A*E) # eq. 10.15
    
    ## pipe axial strain
    eps_p = min(eps_g, eps_f) # eq. 10.16
        
    ##
    return eps_p