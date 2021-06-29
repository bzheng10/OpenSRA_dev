# -----------------------------------------------------------
# Open-Source Seismic Risk Assessment, OpenSRA(TM)
#
# Copyright(c) 2020-2022 The Regents of the University of California and
# Slate Geotechnical Consultants. All Rights Reserved.
#
# Methods for liquefaction triggering: CRR, CSR, FS, and/or pLiq
#
# Created: April 13, 2020
# Updated: July 14, 2020
# --- improved documentation
# @author: Barry Zheng (Slate Geotechnical Consultants)
# -----------------------------------------------------------


# -----------------------------------------------------------
# Python modules
import logging
import numpy as np
from scipy import sparse
from scipy.stats import norm


# -----------------------------------------------------------
def HolzerEtal2011(**kwargs):
    """
    A simplified procedure to quantify probability of liquefaction using **PGA** and **MSF**.

    Parameters
    ----------


    Returns
    -------

    References
    ----------
    .. [1] Holzer, T.L., Noce, T.E., and Bennett, M.J., 2011, Liquefaction Probability Curves for Surficial Geologic Deposits, Environmental and Engineering Geoscience, vol. 17, no. 1, pp. 1-21. doi:10.2113/gseegeosci.17.1.1
    .. [2] Holzer, T.L., Noce, T.E., and Bennett, M.J., 2008, Liquefaction Hazard Maps for Three Earthquake Scenarios for the Communities of San Jose, Campbell, Cupertino, Los Altos, Los Gatos, Milpitas, Mountain View, Palo Alto, Santa Clara, Saratoga, and Sunnyvale, Northern Santa Clara County, California. Open-File Report, 2008, 1270 pp. doi:10.3133/ofr20081270
    .. [3] Youd, T.L., Idriss, I.M., Andrus, R.D., Arango, I., Castro, G., Christian, J.T., Dobry, R., Finn, W.D.L., Harder, L.F., Jr., Hynes, M.E., Ishihara, K., Koester, J.P., Liao, S.S.C., Marcuson, W.F., III, Martin, G.R., Mitchell, J.K., Moriwaki, Y., Power, M.S., Robertson, P.K., Seed, R.B., and Stokoe, K.H., II, 2001, Liquefaction resistance of soils: Summary report from the 1996 NCEER and 1998 NCEER/NSF workshops on evaluation of liquefaction resistance of soils, Journal of Geotechnical and Geoenvironmental Engineering, v. 127, no. 10, pp. 817-833. 

    """

    logging.info("Placeholder - not implemented")
    
    return None


# -----------------------------------------------------------
def YoudEtal2001(**kwargs):
    """
    An SPT-based procedure to quantify liquefaction (**crr** and **fs**) at a given location using the deterministic model by Youd et al. (2001). This function is to be used in conjuction with :func:`edp.corr_spt.youd_etal_2001_corr`.

    Parameters
    ----------
    N1_60_cs : float
        N-value corrected for energy, overburden, and fines content, see :func:`edp.corr_spt.youd_etal_2001_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    Dr : float
        [%] relative density
    tau_stat : float
        [kPa] static shear stress
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa


    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio
    csr : float
        cyclic stress ratio, corrected for magnitude/duration, overburden, and static bias
    fs : float
        factor of safety

    References
    ----------
    .. [1] Youd, T.L., et al., 2001, Liquefaction Resistance of Soils; Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops on Evaluation of Liquefaction Resistance of Soils, Journal of Geotechnical and Geoenvironmental Engineering, vol. 127, no. 10, pp. 817–833.

    """

    # Inputs
    N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    Dr = kwargs.get('Dr',None) # %, relative density
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 101.3 kPa
    tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress

    # magnitude scaling factor
    if M is None:
        msf = 1.0 # eq. 31
    else:
        msf = 10**2.24/M**2.56 # Idriss (1999) factors, recommended by Youd et al. (2001), eq. 24

    # overburden correction factor
    if Dr is None:
        K_sigma = 1.0 # eq. 31
    else:
        f = min(0.8,max(0.6,(0.8-0.6)/(80-40)*(Dr-40)+0.6)) # factor for K_sigma, Figure 15
        K_sigma = (sv0_eff/patm)**(f-1) # eq. 31

    # sloping ground correction factor
    if tau_stat is None:
        K_alpha = 1.0 # eq. 31
    else:
        K_alpha = tau_stat/sv0_eff # eq. 32

    if N1_60_cs is None:
        crr = None
    else:
        # SPT base CRR curve (cyclic resistance ratio)
        crr = 1/(34-min(N1_60_cs,30)) + min(N1_60_cs,30)/135 + 50/(10*min(N1_60_cs,30) + 45)**2 - 1/200 # eq. 4

    # see if enough inputs are given for csr
    if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
        csr = None
        print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
    else:
        # Cyclic stress ratio (demand)
        csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff * rd # Seed and Idriss (1971)
        csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias

    # determine if FS can be calculated
    if crr is None or csr is None:
        fs = None
    else:
        # Factor of safety
        fs = crr/csr

    #
    return crr, csr, fs


# -----------------------------------------------------------
def CetinEtal2004(**kwargs):
    """
    An SPT-Based procedure to quantify liquefaction at a given location using probabilistic model by Cetin et al. (2004). This function produces **p_liq** or equivalent **crr** if **p_liq** is specified. This function is to be used in conjuction with :func:`edp.corr_spt.cetin_etal_2004_corr`.

    Parameters
    ----------
    N1_60 : float
        N-value corrected for energy and overburden, see :func:`edp.corr_spt.cetin_etal_2004_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    fc : float
        [%] relative density
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get **crr**


    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        cyclic stress ratio, corrected for magnitude/duration and overburden
    p_liq : float
        [%] probability of liquefaction

    References
    ----------
    .. [1] Cetin, K.O., Seed, R.B., Der Kiureghian, A., Tokimatsu, K., Harder Jr, L.F., Kayen, R.E., and Moss, R.E., 2004, Standard Penetration Test-Based Probabilistic and Deterministic Assessment of Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 130, no. 12, pp. 1314-1340.

    """

    # Inputs
    N1_60 = kwargs.get('N1_60',None) # corrected SPT blow count
    fc = kwargs.get('fc',0.0) # percent, fines content
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR

    # flag for cases 1 and 2: measurement/estimation errors included/removed, default to remove (Cetin et al., 2004)
    flag_include_error = kwargs.get('flag_include_error',False)

    # model parameters with measurement/estimation errors included/removed (Kramer and Mayfield, 2007)
    if flag_include_error is True:
        t1 = 0.004
        t2 = 13.79
        t3 = 29.06
        t4 = 3.82
        t5 = 0.06
        t6 = 15.25
        sigma = 4.21
    else:
        t1 = 0.004
        t2 = 13.32
        t3 = 29.53
        t4 = 3.70
        t5 = 0.05
        t6 = 16.85
        sigma = 2.70

    # overburden correction factor, K_sigma
    # data from Figure 12, form provided in Idriss and Boulanger (2012, Spectra)
    K_sigma = (sv0_eff/patm)**-0.278 #

    # duration weighting factor (or magnitude scaling factor)
    # data from Figure 10, form provided in Idriss and Boulanger (2012, Spectra)
    dwf = (M/7.5)**-2.217

    # check what inputs are given
    if p_liq is None:

        # cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 10, uncorrected for magnitude and duration
        print('csr_eq = ', str(csr_eq))
        csr = csr_eq/dwf/K_sigma # correction for duration and to 1 atm

        # probability of liquefaction (Cetin et al., 2004)
        p_liq = norm.cdf(-(N1_60*(1+t1*fc) - t2*np.log(csr_eq) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6)/sigma) * 100 # eq. 19

        # cyclic resistance ratio
        crr = np.nan # not used

    else:

        # cyclic stress ratio
        csr = None # not used

        # inverse analysis of CRR given pLiq, (Cetin et al., 2004)
        crr = np.exp((N1_60*(1+t1*fc) - t3*np.log(M) - t4*np.log(sv0_eff/patm) + t5*fc + t6 + sigma*norm.ppf(p_liq/100))/t2) # eq. 20

    #
    return crr, csr, p_liq


# -----------------------------------------------------------
def BoulangerIdriss2014(**kwargs):
    """
    A site-specific procedure to quantify liquefaction at a given location using the deterministic and probabilistic procedures by Idriss and Boulanger (2008) and Boulanger and Idriss (2012; 2014; 2016). This function works with both SPT or CPT inputs, and should be used in conjuction with either :func:`edp.corr_spt.boulanger_idriss_2014_corr_spt` or :func:`edp.corr_cpt.boulanger_idriss_2014_corr_cpt`. A flag is required for deciding whether to follow the deterministic or the probabilistic approach. For the probabilistic approach, this funtion produces **p_liq** or equivalent **crr** if **p_liq** is specified

    Parameters
    ----------
    flag_det_prob : str
        procedure to run: **deterministic** or **probabilistic**
    pen_type : str
        type of penetrometer: specify **spt** or **cpt**
    resistance : float
        [varies] corrected penetration resistance, **N1_60_cs** or **qc_1N_cs**, see :func:`edp.corr_spt.boulanger_idriss_2014_corr_spt` or :func:`edp.corr_cpt.boulanger_idriss_2014_corr_cpt`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    tau_stat : float
        [kPa] static shear stress
    patm : float, optional
        [kPa] atmospheric pressure, default = 101.3 kPa
    Q : float, optional
        fitting parameter for the critical state line used to calculate the relative state parameter; default = 10
    K0 : float, optional
        coefficient of lateral earth pressure; default = 0.5
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get CRR

    Returns
    -------
    crr : float
        [%] equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        [%] cyclic stress ratio, corrected for magnitude/duration and overburden
    fs : float, optional
        factor of safety, depends on **flag_det_prob**
    p_liq : float, optional
        [%] probability of liquefaction, depends on **flag_det_prob**

    References
    ----------
    .. [1] Idriss, I.M., and Boulanger, R.W., 2008, Soil Liquefaction During Earthquakes, Monograph No. 12, Earthquake Engineering Research Institute, Oakland, CA, 261 pp.
    .. [2] Boulanger, R.W., and Idriss, I.M., 2014, CPT and SPT Based Liquefaction Triggering procedures, Report UCD/CGM-14/01, Department of Civil and Environmental Engineering, University of California, Davis, CA, 134 pp.
    .. [3] Boulanger, R.W., and Idriss, I.M., 2012, Probabilistic Standard Penetration Test–Based Liquefaction–Triggering Procedure, Journal of Geotechnical and Geoenvironmental Engineering, vol. 138, no. 10, pp. 1185-1195.
    .. [4] Boulanger, R.W., and Idriss, I.M., 2016, CPT-Based Liquefaction Triggering Procedure, Journal of Geotechnical and Geoenvironmental Engineering, vol. 142, no. 2, 04015065.

    """

    # # flag for deterministic or probabilistic approach (FS vs prob for result)
    flag_det_prob = kwargs.get('flag_det_prob',None)
    if flag_det_prob is None:
        print('Must specify whether to run "deterministic" or "probabilistic" procedure; exiting function')
        return None, None, None

    else:
        # pen_type:
        # - spt (resistance = N1_60_cs)
        # - cpt (resistance = qc_1N_cs)
        pen_type = kwargs.get('pen_type',None)
        resistance = kwargs.get('resistance',None)
        # N1_60_cs = kwargs.get('N1_60_cs',None) # corrected SPT blow count
        # qc_1N_cs = kwargs.get('qc_1N_cs',None) # corrected CPT tip measurement

        # Inputs
        amax = kwargs.get('amax',None) # g, peak surface acceleration
        sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
        sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
        rd = kwargs.get('rd',None) # stress reduction factor with depth
        M = kwargs.get('M',None) # moment magnitude
        patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
        tau_stat = kwargs.get('tau_stat',None) # kPa, static shear stress
        Q = kwargs.get('Q',10) # fitting parameter for critical state line for calculating the relative state parameter
        K0 = kwargs.get('K0',0.5) # coefficient of lateral earth pressure
        p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR=

        # magnitude scaling factor
        if M is None:
            msf = 1.0 # eq. 31
        else:
        #     msf = min(1.8,6.9*np.exp(-M/4)-0.058) # Eq. 2.17, Idriss and Boulanger (2008)
            if 's' in pen_type.lower():
                msf_c1 = 31.5
                msf_c2 = 2

            elif 'c' in pen_type.lower():
                msf_c1 = 180
                msf_c2 = 3

            msf_max = min(2.2,1.09+(resistance/msf_c1)**msf_c2) # eq. 2.20 and 2.21
            msf = 1 + (msf_max-1) * (8.64*np.exp(-M/4)-1.325) # eq. 2.19, Boulanger and Idriss (2014)

        # overburden correction factor
        if 's' in pen_type.lower():
            C_sigma_c1 = 18.9
            C_sigma_c2 = 2.55
            C_sigma_c3 = 0.5
            if resistance > 37:
                print('N1_60_cs > 37, over the limit described in Boulanger and Idriss (2014)')

        elif 'c' in pen_type.lower():
            C_sigma_c1 = 37.3
            C_sigma_c2 = 8.27
            C_sigma_c3 = 0.264
            if resistance > 211:
                print('qc_1N_cs > 211, over the limit described in Boulanger and Idriss (2014)')

        C_sigma = min(0.3,1/(C_sigma_c1-C_sigma_c2*resistance**C_sigma_c3)) # eq. 2.16b and 2.16c
        K_sigma = min(1.1,1 - C_sigma*np.log(sv0_eff/patm)) # eq. 2.16a

        # static shear stress correction factor
        if tau_stat is None:
            K_alpha = 1.0
        else:
            if 's' in pen_type.lower():
                xi_r_c1 = (resistance/46)**0.5 # eq. 64

            elif 'c' in pen_type.lower():
                xi_r_c1 = 0.478*resistance**0.264 - 1.063 # eq. 65
                if resistance < 21:
                    print('qc_1N_cs < 21, below the limit recommended in Idriss and Boulanger (2008)')

            xi_r = 1/(Q-np.log(100*(1+2*K0)/3*sv0_eff/patm)) - xi_r_c1 # eq. 64 and 65
            xi_r = min(0.1,max(-0.6,xi_r)) # eq. 67
            alpha = min(0.35, tau_stat/sv0_eff) # eq. 63, 66
            a = 1267 + 636*alpha**2 - 634*np.exp(alpha) - 632*np.exp(-alpha) # eq. 60
            b = np.exp(-1.11 + 12.3*alpha**2 + 1.31*np.log(alpha + 0.0001)) # eq. 61
            c = 0.138 + 0.126*alpha + 2.52*alpha**3 # eq. 62
            K_alpha = a + b*np.exp(-xi_r/c) # eq. 59

        # see if enough inputs are given for csr
        if amax is None or sv0_tot is None or sv0_eff is None or rd is None:
            csr = None
            print('csr cannot be calculated: missing amax, sv0_tot, sv0_eff, or rd')
        else:
            # cyclic stress ratio (demand)
            csr_m_sigv = 0.65 * amax * sv0_tot/sv0_eff *rd # Seed and Idriss (1971), uncorrected
            csr = csr_m_sigv/msf/K_sigma/K_alpha # CSR for M=7.5 and corrected to 1 atm and static bias

        # see if enough inputs are given for crr or p_liq
        if resistance is None:
            print('N1_60_cs or qc_1N_cs is not provided as an input, exiting procedure')
            return None

        else:
            # overburden correction factor
            if 's' in pen_type.lower():
                crr_c1 = 14.1
                crr_c2 = 126
                crr_c3 = 23.6
                crr_c4 = 25.4

            elif 'c' in pen_type.lower():
                crr_c1 = 113
                crr_c2 = 1000
                crr_c3 = 140
                crr_c4 = 137

            # if deterministic, calculate factor of safety
            if flag_det_prob[0] == 'd':
                Co = 2.8 # value for deterministic analysis, pp. 1197 in BI12

                # SPT base CRR curve (cyclic resistance ratio)
                crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                            (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - Co) # eq. 2.24 and 2.25 in BI14

                if crr is None or csr is None:
                    fs = None
                else:
                    # Factor of safety
                    fs = crr/csr

                #
                return crr, csr, fs

            # if probabilistic, calculate p_liq or equivalent crr
            if flag_det_prob[0] == 'p':

                # check what inputs are given
                if p_liq is None:
                    Co = 2.67 # recommended on pp. 1192 in BI12
                    sigma_R = 0.13 # recommended on pp. 1192 in BI12

                    # Probability of liquefaction
                    p_liq = norm.cdf(-(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                                    (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
                                    Co - np.log(csr))/sigma_R) * 100 # eq. 31 in BI12 and 36 in BI16

                    # Cyclic resistance ratio
                    crr = np.nan # not used

                else:
                    # Inverse analysis of CRR given pLiq, (Cetin et al., 2004)
                    crr = np.exp(resistance/crr_c1 + (resistance/crr_c2)**2 - \
                                (resistance/crr_c3)**3 + (resistance/crr_c4)**4 - \
                                Co + sigma_R*norm.ppf(p_liq/100)) # eq. 30 in BI12 and 34 in BI16

                #
                return crr, csr, p_liq


# -----------------------------------------------------------
def MossEtal2006(**kwargs):
    """
    A CPT-based procedure to quantify liquefaction at a given location using probabilistic model by Moss et al. (2006). This function produces **p_liq** or equivalent **crr** if **p_liq** is specified. This function is to be used in conjuction with :func:`edp.corr_cpt.moss_etal_2006_corr`.

    Parameters
    ----------
    qc_1 : float
        normalized tip resistance corrected for overburden, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    Rf : float
        [%] friction ratio, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    c : float
        overburden normalization exponent, see :func:`edp.corr_cpt.moss_etal_2006_corr`
    amax : float
        [g] peak ground acceleration
    sv0_tot : float
        [g] initial vertical total stress
    sve_tot : float
        [g] initial vertical effective stress
    rd : float
        [g] depth-reduction factor, see :func:`edp.fcn_liq_land.get_rd`
    M : float
        moment magnitude
    p_liq : float, optional
        [%] probability of liquefaction, if looking to get **crr**

    Returns
    -------
    crr : float
        equivalent cyclic resistance ratio if p_liq is specified, otherwise NaN
    csr : float
        cyclic stress ratio, corrected for magnitude/duration
    p_liq : float
        [%] probability of liquefaction

    References
    ----------
    .. [1] Moss, R.E.S., Seed, R.B., Kayen, R.E., Stewart, J.P., Der Kiureghian, A., and Cetin, K.O., 2006a, CPT-Based Probabilistic and Deterministic Assessment of In Situ Seismic Soil Liquefaction Potential, Journal of Geotechnical and Geoenvironmental Engineering, vol. 132, no. 8, pp. 1032-1051.

    """

    # Inputs
    qc_1 = kwargs.get('qc_1',None) # qc corrected for overburden
    Rf = kwargs.get('Rf',None) # %, friction ratio = fs/qc
    c = kwargs.get('Rf',None) # %, overburden normalization exponent
    amax = kwargs.get('amax',None) # g, peak surface acceleration
    sv0_tot = kwargs.get('sv0_tot',None) # kPa, initial vertical total stress
    sv0_eff = kwargs.get('sv0_eff',None) # kPa, initial vertical effective stress
    rd = kwargs.get('rd',None) # stress reduction factor with depth
    M = kwargs.get('M',None) # moment magnitude
    # patm = kwargs.get('patm',101.3) # atmphospheric pressure in units of svo_eff, default to 100 kPa
    p_liq = kwargs.get('p_liq',None) # %, probability of liquefaction, if want to determine equivalent CRR

    # check if enough base inputs are given
    if qc_1 is None or Rf is None or c is None or M is None or sv0_eff is None:

        #
        csr = None
        crr = None
        p_liq = None
        print('Not enough inputs to proceed with procedure')

    #
    else:

        # Duration weighting factor (or magnitude scaling factor)
        dwf = 17.84*M**-1.43
        if M < 5.5 or M > 8.5:
            print('Magnitude is outside the valid range of 5.5-8.5')

        # Cyclic stress ratio
        csr_eq = 0.65*amax*sv0_tot/sv0_eff*rd # eq. 1, uncorrected for magnitude and duration
        csr = csr_eq/dwf # correction for duration, eq. 6

        # Increase in qc_1 and friction ratio due to frictional effects of apparent "fines" and character
        if Rf <= 0.5:
            d_qc = 0
        #
        else:
            x1 = 0.38*(min(Rf,5)) - 0.19
            x2 = 1.46*(min(Rf,5)) - 0.73
            d_qc = x1*np.log(csr) + x2

        # apply correction
        qc_1_mod = qc_1 + d_qc

        # Model parameters
        t1 = 1.045
        t2 = 0.110
        t3 = 0.001
        t4 = 0.850
        t5 = 7.177
        t6 = 0.848
        t7 = 0.002
        t8 = 20.923
        sigma = 1.632

        # check what inputs are given
        if p_liq is None:

            # Probability of liquefaction
            p_liq = norm.cdf(-(qc_1_mod**t1 + qc_1_mod*(t2*Rf) + t3*Rf + \
                            c*(1+t4*Rf) - t5*np.log(csr) - t6*np.log(M) - \
                            t7*np.log(sv0_eff) - t8)/sigma) * 100 # eq. 20

            # Cyclic resistance ratio
            crr = np.nan # not used

        #
        else:

            # Inverse analysis of CRR given pLiq
            crr = np.exp((qc_1_mod**t1 + qc_1_mod*(t2*Rf)  + t3*Rf + \
                        c*(1+t4*Rf) - t6*np.log(M) - t6*np.log(M) - \
                        t7 + sigma*norm.ppf(p_liq/100))/t5) # eq. 21

    #
    return crr, csr, p_liq


# -----------------------------------------------------------
def ZhuEtal2017(**kwargs):
    """
    A map-based procedure to quantify liquefaction at a given location using logistic models by Zhu et al. (2017). Two models are provided:

    1. For distance to coast < cutoff, **p_liq** = f(**pgv**, **vs30**, **precip**, **dc**, **dr**)
    2. For distance to coast >= cutoff, **p_liq** = f(**pgv**, **vs30**, **precip**, **dc**, **dr**, **wtd**)

    Additionally, an ad-hoc magnitude scaling factor is included for PGV correction, see `https://earthquake.usgs.gov/data/ground-failure/background.php <https://earthquake.usgs.gov/data/ground-failure/background.php>`_.

    Parameters
    ----------
    vs30 : float
        [m/s] shear wave velocity
    precip : float
        [mm] mean annual precipitation
    dc : float
        [km] distance to nearest coast
    dr : float
        [km] distance to nearest river
    wtd : float
        [m] global water table depth
    pgv : float, optional
        [cm/s] peak ground velocity, required if seeking liquefaction triggering
    M : float, optional
        moment magnitude, for **pgv** correction; no correction is applied if **M** is not provided
    dc_cutoff : float, optional
        [km] cutoff for **dc** where the procedure switches from model 1 to model 2, default to 20 km
    return_param : str
        choose variables to return; specify any number of the return variables or **all** for all variables; default = 'liq_susc'

    Returns
    -------
    liq_susc : str
        liquefaction susceptibility (none to very high)
    p_liq : float
        [%] probability of liquefaction
    areal_liq : float
        [%] areal liquefaction percent

    References
    ----------
    .. [1] Zhu, J., Baise, L.G., and Thompson, E.M., 2017, An Updated Geospatial Liquefaction Model for Global Application, Bulletin of the Seismological Society of America, vol. 107, no. 3, pp. 1365-1385.

    """

    # two models by Zhu et al. (2017):
    # -- Model 1: better globally
    # -- Model 2: better for noncoastal (coastal cutoff at 20 km)
    
    # get inputs
    vs30 = kwargs.get('Vs30',None) # m/s, shear wave vlocity over first 30 m
    precip = kwargs.get('Precipitation',None) # mm, mean annual precipitation
    dc = kwargs.get('DistanceToCoast',None) # km, distance to nearest coast
    dr = kwargs.get('DistanceToRiver',None) # km, distance to nearest river
    dw = kwargs.get('DistanceToWaterBody',None) # km, distance to nearest water body
    wtd = kwargs.get('WaterTableDepth',None) # m, global water table depth
    M = kwargs.get('M',None) # moment magnitude, for liquefaction triggering
    pgv = kwargs.get('pgv',None) # cm/s, peak ground velocity, for liquefaction triggering
    n_site = kwargs.get('n_site',pgv[0].shape[1]) # number of sites
    n_event = kwargs.get('n_event',pgv[0].shape[0]) # number of events
    dc_cutoff = kwargs.get('ModelTransition',20) # km, transition from model 1 to model 2, default to 20 km
    return_param = kwargs.get('return_param',None) # default to liq susc
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1
    
    # default return_param
    if return_param is None:
        return_param = ['liq_susc']

    # see if basic inputs are provided
    if vs30 is None or precip is None or dc is None:
        print('Insufficient inputs, exiting procedure.')
        return None

    else:
        # set any parameters that are not specified to equal to 1 (instead of 0 due to possibility ln(0)), the coefficients of these parameters are set to 0, and so the parameter should drop out of the equation
        if dr is None:
            dr = np.ones(n_site)
        if dw is None:
            dw = np.ones(n_site)
        if wtd is None:
            wtd = np.ones(n_site)

        # determine model coefficients due to dc_cutoff
        b_0 = np.ones(n_site)*8.801
        b_0[dc<dc_cutoff] = 12.435

        b_ln_pgv = np.ones(n_site)*0.334
        b_ln_pgv[dc<dc_cutoff] = 0.301

        b_ln_vs30 = np.ones(n_site)*-1.918
        b_ln_vs30[dc<dc_cutoff] = -2.615

        b_precip = np.ones(n_site)*5.408e-4
        b_precip[dc<dc_cutoff] = 5.556e-4

        b_sqrt_dc = np.ones(n_site)*0
        b_sqrt_dc[dc<dc_cutoff] = -0.0287

        b_dr = np.ones(n_site)*0
        b_dr[dc<dc_cutoff] = 0.0666

        b_sqrt_dc_dr = np.ones(n_site)*0
        b_sqrt_dc_dr[dc<dc_cutoff] = -0.0369

        b_dw = np.ones(n_site)*-0.2054
        b_dw[dc<dc_cutoff] = 0

        b_wtd = np.ones(n_site)*-0.0333
        b_wtd[dc<dc_cutoff] = 0

        a = np.ones(n_site)*49.15
        a[dc<dc_cutoff] = 42.08

        b = np.ones(n_site)*42.40
        b[dc<dc_cutoff] = 62.59

        c = np.ones(n_site)*9.165
        c[dc<dc_cutoff] = 11.43

        # calculate liquefaction susceptibility
        liq_susc = b_0 + b_ln_vs30*np.log(vs30) + b_precip*precip + \
                    b_sqrt_dc*np.sqrt(dc) + b_dr*dr + b_sqrt_dc_dr*np.sqrt(dc)*dr + \
                    b_dw*dw + b_wtd*wtd
        # print(f"liq_susc = {liq_susc}")

        # set to -999 if on water (distance to coast < 0)
        liq_susc[dc<0] = -999
        # liq_susc = np.asarray([-999 if dc[i] < 0 else liq_susc[i] for i in range(n_site)])

        # convert liq_susc values to categories
        liq_susc_cat = np.empty(n_site,dtype='<U10')
        liq_susc_cat[liq_susc<-38.1] = 'none'
        liq_susc_cat[np.logical_and(liq_susc>=-38.1,liq_susc<-3.20)] = 'very low'
        liq_susc_cat[np.logical_and(liq_susc>=-3.20,liq_susc<-3.15)] = 'low'
        liq_susc_cat[np.logical_and(liq_susc>=-3.15,liq_susc<-1.95)] = 'moderate'
        liq_susc_cat[np.logical_and(liq_susc>=-1.95,liq_susc<-1.15)] = 'high'
        liq_susc_cat[liq_susc>-1.15] = 'very high'

        # see if pgv is given
        if 'p_liq' in return_param or 'areal_liq' in return_param:
            # preset dict
            p_liq = {}
            # areal_liq = {}

            # ad-hoc magnitude scaling factor added by USGS, to be applied to pgv
            # https://earthquake.usgs.gov/data/ground-failure/background.php
            if M is None:
                sf = 1
            else:
                sf = 1/(1+np.exp(-2*(M-6)))
                sf = np.repeat(sf[:,np.newaxis],n_site,axis=1)
                # print(f"SF = {sf}")

            # repeat liq_susc for all ruptures
            liq_susc = np.transpose(np.repeat(liq_susc[:,np.newaxis],n_event,axis=1))
            b_ln_pgv = np.transpose(np.repeat(b_ln_pgv[:,np.newaxis],n_event,axis=1))
            vs30 = np.transpose(np.repeat(vs30[:,np.newaxis],n_event,axis=1))
            dc = np.transpose(np.repeat(dc[:,np.newaxis],n_event,axis=1))

            for k in range(n_sample):
                # get IM of all sites for current sample
                pgv_k = pgv[k].toarray()
                # rows = pgv[k].row
                # cols = pgv[k].col

                # probability of liquefaction
                #p_liq_k = pgv_k.multiply(sf).power(-1).multiply(np.exp(-liq_susc))
                #p_liq_k.data = 1/(p_liq_k.data + 1)
                # p_liq_k = p_liq_k + np.ones(
                #p_liq_k = p_liq_k.power(-1)
                #p_liq_k[np.where(np.logical(_and(pgv_k<3] = 0
                # p_liq_k[:,[vs30 > 620]] = 0

                p_liq_k = np.zeros(pgv_k.shape)
                p_liq_k[pgv_k>3] = 1/(1+np.exp(-(liq_susc[pgv_k>3] + np.multiply(
                                    b_ln_pgv[pgv_k>3],np.log(np.multiply(pgv_k[pgv_k>3],sf[pgv_k>3]))))))

                # print(f"liq_susc = {liq_susc}, b_ln_pgv = {b_ln_pgv}, pgv_k = {pgv_k}, sf = {sf}")

                # X_p_liq_k = liq_susc + np.multiply(b_ln_pgv,np.log(np.multiply(pgv_k,sf)))
                # p_liq_k = 1 / (1 + np.exp(-X_p_liq_k))
                # p_liq_k[pgv_k < 3] = 0
                # p_liq_k[vs30 > 620] = 0
                p_liq_k[vs30 > 620] = 0

                # areal liquefaction percent
                # areal_liq_k = a / (1 + b * np.exp(-c * p_liq_k))**2
                # areal_liq_k = None

                # convert p_liq to percent
                p_liq_k = p_liq_k*100

                # set to -999 if on water (distance to coast < 0)
                p_liq_k[dc < 0] = -999
                # areal_liq_k[dc < 0] = -999

                # append sims
                p_liq.update({k:sparse.coo_matrix(p_liq_k)})
                # areal_liq.update({k:areal_liq_k})
                # p_liq.append(p_liq_k)
                # areal_liq.append(areal_liq_k)

            # set to numpy arrays
            # p_liq = np.asarray(p_liq)
            # areal_liq = np.asarray(areal_liq)

        # probability distribution
        prob_dist_type = 'Uniform'
        factor_aleatory = 3
        factor_epistemic = 4

        # store outputs
        output = {}
        #
        if 'liq_susc' in return_param:
            output.update({'liq_susc': liq_susc_cat})
        if 'p_liq' in return_param:
            output.update({'p_liq': p_liq})            
            output.update({'prob_dist': {'type': prob_dist_type,
                                        'factor_aleatory': factor_aleatory,
                                        'factor_epistemic': factor_epistemic}})
        # if 'areal_liq' in return_param:
            # output.update({'areal_liq': areal_liq})

        #
        return output


# -----------------------------------------------------------
def Hazus2014(**kwargs):
    """
    Compute probability of liquefaction at a given location using a simplified method after Liao et al. (1988).

    Parameters
    ----------
    liq_susc : str
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)
    pga : float
        [g] peak ground acceleration
    M : float
        moment magnitude
    wtd : float, optional
        [m] depth to water table

    Returns
    -------
    p_liq : float
        [%] probability of liquefaction

    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2014, Multi-Hazard Loss Estimation Methodology, Earthquake Model, Hazus MH 2.1 Technical Manual, National Institute of Building Sciences and Federal Emergency Management Agency, Washington, DC, 690 p.
    .. [2] Liao, S.S., Veneziano, D., and Whitman, R.V., 1988, Regression Models for Evaluating Liquefaction Probability, Journal of Geotechnical Engineering, vol. 114, no. 4, pp. 389-411.

    """

    # get inputs
    liq_susc = kwargs.get('liq_susc',None) # liquefaction susceptibility category
    pga = kwargs.get('pga',None) # g, peak ground acceleration
    M = kwargs.get('M',None) # moment magnitude
    wtd = kwargs.get('WaterTableDepth',None) # m, depth to water table
    n_sample = kwargs.get('n_sample',1) # number of samples, default to 1

    # preset list
    p_liq = {}

    # correction factors for magnitude/duration and water table
    k_m = 0.0027 * M**3 - 0.0267 * M**2 - 0.2055 * M + 2.9188 # correction factor for moment magnitudes other than M=7.5, eq. 4-21
    k_w = 0.022 * wtd/2.54 + 0.93 # correction for groudnwater depths other than 5 feet, eq. 4-22

    # loop through all realizations
    for k in range(n_sample):

        # get IM of all sites for current sample
        pga_k = pga[k].toarray()

        # get uncorrected p_liq given pga
        p_liq_pga = np.zeros(pga_k.shape)
        p_liq_pga[liq_susc=='very high'] = np.maximum(np.minimum(9.09*pga_k[liq_susc=='very high']-0.82,1),0)
        p_liq_pga[liq_susc=='high'] = np.maximum(np.minimum(7.67*pga_k[liq_susc=='high']-0.92,1),0)
        p_liq_pga[liq_susc=='moderate'] = np.maximum(np.minimum(6.67*pga_k[liq_susc=='moderate']-1.00,1),0)
        p_liq_pga[liq_susc=='low'] = np.maximum(np.minimum(5.57*pga_k[liq_susc=='low']-1.18,1),0)
        p_liq_pga[liq_susc=='very low'] = np.maximum(np.minimum(4.16*pga_k[liq_susc=='very low']-1.08,1),0)
        p_liq_pga[liq_susc=='none'] = 0

        # get percent of liquefaction given pga
        p_ml = np.zeros(pga_k.shape)
        p_ml[liq_susc=='very high'] = 0.25
        p_ml[liq_susc=='high'] = 0.20
        p_ml[liq_susc=='moderate'] = 0.10
        p_ml[liq_susc=='low'] = 0.05
        p_ml[liq_susc=='very low'] = 0.02
        p_ml[liq_susc=='none'] = 0.00

        # liquefaction likelihood, p_liq
        p_liq_k = p_liq_pga / k_m / k_w * p_ml * 100 # eq. 4-20

        # append sims
        p_liq.update({k:p_liq_k})

    # set to numpy arrays
    # p_liq = np.asarray(p_liq)

    # probability distribution
    prob_dist_type = 'uniform'
    factor_aleatory = 3
    factor_epistemic = 4

    # store outputs
    output = {}
    output.update({'p_liq': p_liq})
    output.update({'prob_dist': {'type': prob_dist_type,
                                'factor_aleatory': factor_aleatory,
                                'factor_epistemic': factor_epistemic}})

    #
    return output