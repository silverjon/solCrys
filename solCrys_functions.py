# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt

def bmpp(phase):
    "Expresses phases as between -pi and pi"
    return (phase + np.pi)%(2*np.pi) - np.pi


def cschc(x):
    "Calculates x/sinh(x) for an array or variable without generating a NAN error for x=0"
    if hasattr(x, '__len__'):
        out = np.empty_like(x)
        ok = np.abs(x) > 1e-6
        out[ok] = x[ok]/np.sinh(x[ok])
        out[~ok] = 1 - x[~ok]**2/6
        return out
    elif np.abs(x) > 1e-6: return x/np.sinh(x)
    else: return 1 - x**2/6


def detect_comb_lines(f_peaks_raw, dB_peaks_raw, min_dB):
    """Fits the pump frequency and comb repetition rate (frequency spacing) to a list of peak
    frequencies and powers in dBm, assuming that the highest peak is the pump and all the 
    comb lines are spaced by multiples of the repetition rate, with most of those multiples 
    being 1. In other words, the peaks should all have frequencies fpump + mode_no*frep for
    integer mode_no, and most values of mode_no between the minimum and maximum should be 
    occupied exactly once. Pre-selects only those peaks with dB values above min_dB. After
    fitting fpump and frep, rejects any peaks that with frequencies more than 0.2*frep away
    from the nearest value of fpump + mode_no*frep, and counts the number rejected."""
    select = dB_peaks_raw > min_dB
    fpeaks = f_peaks_raw[select]
    dBpeaks = dB_peaks_raw[select]
    
    def frep_minfunc(params):
    #Function that is minimised when fpeaks = fpump + mode_no*frep for integer array mode_no
        fpump, frep = params
        return np.sum(1 - np.cos(2*np.pi*(fpeaks - fpump)/frep))
    
    fpump_guess = fpeaks[np.argmax(dBpeaks)] #frequency of the highest peak
    frep_guess = np.abs(np.median(np.diff(fpeaks))) #median spacing
    
    fpump, frep = opt.minimize(frep_minfunc, (fpump_guess, frep_guess)).x
    
    #mode_no[i] is the closest integer to (fpeaks[i] - fpump)/frep
    mode_no = np.rint((fpeaks - fpump)/frep).astype(int)
    #Fractional part of (fpeaks - fpump)/frep expressed as between -0.5 and 0.5:
    frac_mode_no = (fpeaks - fpump)/frep - mode_no
    #Rejects peaks that are more than 0.2 FSR from the nearest comb line according to the fit
    select_peaks = np.abs(frac_mode_no) < 0.2
    no_rejected = np.count_nonzero(~select_peaks)
    
    return fpump, frep, fpeaks[select_peaks], dBpeaks[select_peaks],\
        mode_no[select_peaks], no_rejected


def fit_linear_phase_via_FFT(mode_no, phase, N):
    """Fits parameters D0 and D1 in phase = (D0 + D1*mode_no)(mod 2*pi) using an FFT.
    mode_no should be a 1D array of mostly consecutive integers and phase an array of the
    same length specifiying the phases at those mode numbers. N is the size of the grid
    that is used for the FFT, and should be larger than 2*max(abs(mode_no)). The larger N
    is, the more accurate the result. Also returns the absolute value of the FFT peak."""
    eip_grid = np.zeros(N, dtype=complex)    
    eip_grid[mode_no] = np.exp(1j*phase)
    fft_eip_grid = np.fft.fft(eip_grid)
    n_peak = np.argmax(np.abs(fft_eip_grid))
    D1 = 2*np.pi*n_peak/N
    fft_peak = fft_eip_grid[n_peak]
    D0 = np.angle(fft_peak)
    abs_fft_peak = np.abs(fft_peak)
    return D0, D1, abs_fft_peak


def fit_polynomial_phase(mode_no, phase, p_guess, stdev_cutoff):
    """Fits a polynomial to phase vs. mode number of the form
    phase = (p[0]*mode_no**d + p[1]*mode_no**(d-1) + ... + p[d])(mod 2*pi) where d 
    is the degree of the polynomial (obtained from the length of p_guess). Performs an
    initial fit, then removes outliers with residuals whose absolute values are greater
    than stdev_cutoff times the standard deviation of the residuals, and fits again.
    The fit and initial guess need to both be reasonably good for this to work."""
    def minfunc(polynom, mode_no, phase):
        return np.sum(1 - np.cos(phase - np.polyval(polynom, mode_no)))
    
    pv_guess = np.polyval(p_guess, mode_no)
    phase_minus_pvg = bmpp(phase - pv_guess)
    
    p1_init = opt.minimize(minfunc, np.zeros(len(p_guess)), args=(mode_no, phase_minus_pvg)).x
    residuals = bmpp(phase_minus_pvg - np.polyval(p1_init, mode_no))
    keep_mask = np.abs(residuals) <= stdev_cutoff*np.std(residuals)
    
    p1 = opt.minimize(minfunc, p1_init, args=(mode_no[keep_mask], phase_minus_pvg[keep_mask])).x
    residuals = bmpp(phase_minus_pvg - np.polyval(p1, mode_no))
    return p_guess + p1, residuals, keep_mask


def find_chirality_and_dispersion(phase_fit, phase_meas, mode_no, N):
    """Finds the chirality and dispersion coefficients D0, D1 and D2 that must be applied 
    to phase_fit to make it match up with phase_meas. Specifically,
    phase_meas = (chirality*phase_fit + D0 + D1*n + D2*n**2/2 + residual_offset)(mod 2*pi)
    where n = mode_no. mode_no must be (almost) entirely monotonically increasing by 1. 
    N is the number of points used in fit_linear_phase_via_FFT()."""
    phase_diff = phase_meas - phase_fit
    phase_sum = phase_meas + phase_fit

    s = np.argwhere(np.diff(mode_no) == 1)[:,0]
    dp_diff = phase_diff[s+1] - phase_diff[s]
    dp_sum = phase_sum[s+1] - phase_sum[s]
    
    D1_diff, D2_diff, abs_fft_peak_diff = fit_linear_phase_via_FFT(mode_no[s], dp_diff, N)
    D1_sum, D2_sum, abs_fft_peak_sum = fit_linear_phase_via_FFT(mode_no[s], dp_sum, N)
    
    chirality = 2*(abs_fft_peak_diff > abs_fft_peak_sum) - 1
    phase_offset = phase_diff if chirality == 1 else phase_sum
    D2_guess = D2_diff if chirality == 1 else D2_sum
    phase_offset_D2g_removed = phase_offset - D2_guess*mode_no**2/2
    
    D0_guess, D1_guess, afp = fit_linear_phase_via_FFT(mode_no, phase_offset_D2g_removed, N)
    
    (D2over2, D1, D0), residuals, keep_mask =\
        fit_polynomial_phase(mode_no, phase_offset, (D2_guess/2, D1_guess, D0_guess), 2)
    return chirality, D0, D1, 2*D2over2, residuals, keep_mask


class solCrys:
    """"""
    
    def __init__(self, N=2**10):
        self.N = N
        self.n = np.fft.fftfreq(N, d=1/N)
        
        self.asymm_plot_nmax = 100
        self.epb_rel_cutoff = 7
        self.epb_rel_cutoff_plot_lims = [2,10]
        self.rel_gauss_w_lims = [0.5,1]
        self.gauss_w_steps = 6
        self.rand_coeff = 0.5
        self.bof_threshold = 1


    def set_fit_params(self, asymm_plot_nmax=100, epb_rel_cutoff=3,\
                       epb_rel_cutoff_plot_lims = [2,10], rel_gauss_w_lims = [0.5,1],\
                       gauss_w_steps=6, rand_coeff=0.5, bof_threshold=0.3):
        self.asymm_plot_nmax = asymm_plot_nmax
        self.epb_rel_cutoff = epb_rel_cutoff
        self.epb_rel_cutoff_plot_lims = epb_rel_cutoff_plot_lims
        self.rel_gauss_w_lims = rel_gauss_w_lims
        self.gauss_w_steps = gauss_w_steps
        self.rand_coeff = rand_coeff
        self.bof_threshold = bof_threshold


    def comb_plot_dB(self, ax, Etilde, min_dB, lw, color, label=None):
        dB = 20*np.log10(np.abs(Etilde)+1e-100)
        mclp = np.max(np.where(dB[:self.N//2] > min_dB)) #max comb line to plot
        ax.vlines(np.arange(-mclp,mclp+1), min_dB, np.maximum(np.roll(dB,mclp)[:2*mclp+1], min_dB),\
                  lw=lw, colors=color, label=label)


    def import_spectrum(self, spectrum_filename, plot=True):
        c = 299792458 #m/s
        
        data = np.genfromtxt(spectrum_filename, delimiter=',')
        "First column of data is wavelength in nm, second column is power in dBm"
        self.wl = data[:,0] #in nm
        self.f = c/self.wl/1000 #in THz
        self.dBm = data[:,1]
        
        self.peaks_raw = sig.find_peaks(self.dBm, prominence=3)[0]
        self.f_peaks_raw = self.f[self.peaks_raw]
        self.dBm_peaks_raw = self.dBm[self.peaks_raw]

        max_non_pump_dBm = np.sort(self.dBm_peaks_raw)[-2]

        dBm_cutoff = np.min(self.dBm_peaks_raw)

        while dBm_cutoff < max_non_pump_dBm:
            self.fpump, self.frep, self.f_comblines, self.dBm_comblines, self.mode_no_comblines,\
                no_rejected = detect_comb_lines(self.f_peaks_raw, self.dBm_peaks_raw, dBm_cutoff)
            if (no_rejected == 0) & np.all(np.diff(self.mode_no_comblines) != 0): break
            dBm_cutoff += 1
            
        print("Pump frequency = %g THz, repetition rate = %g GHz" % (self.fpump, self.frep*1000))
            
        if dBm_cutoff > max_non_pump_dBm - 40:
            print("Warning: comb dynamic range below 40 dB, may not fit easily")

        self.wl_comblines = c/self.f_comblines/1000
        
        if plot:
            plt.plot(self.wl, self.dBm, lw=0.5, label="Spectrum")
            plt.plot(self.wl_comblines, self.dBm_comblines, 'x', ms=2, label="Comb lines")
            plt.ylim(np.min(self.dBm_peaks_raw) - 10, None)
            plt.xlabel("Wavelength /nm")
            plt.ylabel("Power /dBm")
            plt.legend(frameon=False)
            plt.show()
        
        self.max_abs_mode_no = np.max(np.abs(self.mode_no_comblines))
        
        #Symmetrising dB about the pump on a grid of length N with the pump at element 0:
        self.dBm_data = np.zeros(self.N)
        self.mW_asymm = np.zeros(self.N)
        self.weights = np.zeros(self.N)
        self.dBm_data[self.mode_no_comblines] = self.dBm_comblines/2
        self.mW_asymm[self.mode_no_comblines] = 10**(self.dBm_comblines/10)
        self.weights[self.mode_no_comblines] = 1/2
        self.dBm_data[-self.mode_no_comblines] += self.dBm_comblines/2
        self.mW_asymm[-self.mode_no_comblines] -= 10**(self.dBm_comblines/10)
        self.weights[-self.mode_no_comblines] += 1/2
        self.dBm_data[self.weights == 1/2] *= 2
        self.mW_asymm[self.weights == 1/2] = 0
        self.amps_data = 10**(self.dBm_data/20)*(self.weights>0)
        
        if plot:
            plt.vlines(self.n[:self.asymm_plot_nmax+1], 0, self.mW_asymm[:self.asymm_plot_nmax+1])
            plt.xlabel("Mode number")
            plt.ylabel("Comb line power asymmetry /mW")
            plt.xlim(0,self.asymm_plot_nmax)
            plt.gca().set_xticks(np.arange(0,self.asymm_plot_nmax+1,10))
            plt.gca().set_xticks(np.arange(0,self.asymm_plot_nmax+1,2), minor=True)
            plt.grid()
            plt.grid(which='minor', lw=0.5)
            plt.show()


    def estimate_params(self):
        "From amplitudes of comb lines (pump = element 0), finds A, A0, beta and ns"
        mom2 = np.sum(self.n**2*self.amps_data**2) # 2nd-order moment about the mean
        mom4 = np.sum(self.n**4*self.amps_data**2) # 4th-order moment about the mean
        beta = np.sqrt(20*np.pi**2/7*mom4/mom2)
        mom0_true = 12*np.pi**2*mom2/beta**2
        mom0_exclPump = np.sum(self.amps_data[1:]**2)
        amp0sq_true = mom0_true - mom0_exclPump
        A0 = self.amps_data[0]/np.sqrt(amp0sq_true)
        ns = amp0sq_true*2*beta/np.pi**2/mom0_true #soliton number
        A = np.sqrt(amp0sq_true)/ns
        return A, A0, beta, ns    


    def estimate_params_better(self, m_cutoff, ax=None):
        """Specify pyplot.axes object ax to plot the fit"""
        autocorrel_data = np.real(np.fft.ifft(self.amps_data**2))[:m_cutoff]
        point_no = np.arange(m_cutoff)
        G_guess = self.beta_fit/self.N
        H_guess = 2*self.A_fit**2*self.beta_fit*self.ns_fit/np.pi**2
        L_guess = self.A_fit**2*(self.A0_fit**2 - 1)*self.ns_fit**2
        def fitfunc(m, G, H, J, K, L): 
            return (H*cschc(G*m) + J*np.cosh(G*m) + K*m*np.sinh(G*m) + L)/self.N
        guess = G_guess, H_guess, 0, 0, L_guess
        fit = opt.curve_fit(fitfunc, point_no, autocorrel_data, guess)[0]
        if ax != None:
            ax.plot(point_no, autocorrel_data/np.max(autocorrel_data), lw=2, label='Data')
            ax.plot(point_no, fitfunc(point_no, *fit)/np.max(autocorrel_data), lw=1, label='Fit')
            ax.legend(frameon=False)
            ax.set_xlabel(r'Point number $m$')
            ax.set_ylabel('Autocorrelation (a.u.)')
        G_fit, H_fit, J_fit, K_fit, L_fit = fit
        beta_fit = self.N*G_fit
        ns_fit = 2*beta_fit/np.pi**2*(self.amps_data[0]**2 - L_fit)/H_fit
        A0_fit = np.sqrt(1 + 2*L_fit*beta_fit/np.pi**2/H_fit/ns_fit)
        A_fit = self.amps_data[0]/A0_fit/ns_fit
        return A_fit, A0_fit, beta_fit, ns_fit


    def plot_epb_ns_fits(self, ax):
        min_m_cutoff = int(np.rint(self.epb_rel_cutoff_plot_lims[0]/self.beta_fit*self.N))
        max_m_cutoff = int(np.rint(self.epb_rel_cutoff_plot_lims[1]/self.beta_fit*self.N))
        
        m_cutoffs = np.arange(min_m_cutoff, max_m_cutoff + 1)
        rel_cutoffs = m_cutoffs*self.beta_fit/self.N
        ns_fits = np.empty(len(m_cutoffs))
        
        for i, m_cutoff in enumerate(m_cutoffs):
            A, A0, beta, ns_fits[i] = self.estimate_params_better(m_cutoff)
        ax.plot(rel_cutoffs, ns_fits)
        set_cutoff = int(np.rint(self.epb_rel_cutoff/self.beta_fit*self.N))
        rel_cutoff_actual = set_cutoff*self.beta_fit/self.N
        ax.axis('tight')
        ax.vlines(rel_cutoff_actual, ax.get_ylim()[0],\
                    ns_fits[m_cutoffs == set_cutoff], ls='dashed', lw=1, color='grey')
        ax.hlines(ns_fits[m_cutoffs == set_cutoff], ax.get_xlim()[0],\
                    rel_cutoff_actual, ls='dashed', lw=1, color='grey')
        ax.set_xlabel(r"Relative cutoff $\mu_\mathrm{c}$")
        ax.set_ylabel(r"Fitted soliton number $n_\mathrm{s}$")


    def gen_Etilde(self, sol_locations, envelope, gen_E=False):
        #sol_locations are real numbers in the range [0,1)
        #envelope is the envelope function Stilde as an array in the frequency domain
        slm = sol_locations*self.N #Soliton locations mapped (onto grid of size N)
        slm_int = slm.astype(int)%self.N
        slm_frac = slm%1
        P = np.zeros(self.N)
        P[slm_int] = 1 - slm_frac
        P[(slm_int + 1)%self.N] += slm_frac
        Ptilde = np.fft.fft(P)
        Etilde = Ptilde*envelope
        if gen_E:
            E = np.fft.ifft(Etilde)
            return E, Etilde
        else: return Etilde


    def fit_pattern(self, ns_init=None, ns_stdev=0.5, plot_ns_fits=False, plot_all_fits=False,\
                    plot_solution=True):
        self.A_fit, self.A0_fit, self.beta_fit, self.ns_fit = self.estimate_params()
        print("First estimate: A = %g, A0 = %g, beta = %g, ns = %g"\
              % (self.A_fit, self.A0_fit, self.beta_fit, self.ns_fit))
        
        if plot_ns_fits:
            self.plot_epb_ns_fits(plt.gca())
            plt.show()
        
        epb_m_cutoff = int(np.rint(self.epb_rel_cutoff/self.beta_fit*self.N))
        if plot_ns_fits: 
            self.A_fit, self.A0_fit, self.beta_fit, self.ns_fit =\
                self.estimate_params_better(epb_m_cutoff, ax=plt.gca())
            plt.show()
        else: self.A_fit, self.A0_fit, self.beta_fit, self.ns_fit =\
                self.estimate_params_better(epb_m_cutoff)
        print("Second estimate: A = %g, A0 = %g, beta = %g, ns = %g"\
              % (self.A_fit, self.A0_fit, self.beta_fit, self.ns_fit))
        
        self.ns_init = self.ns_fit if ns_init == None else ns_init

        self.sech_fit = 1/np.cosh(self.n*np.pi**2/self.beta_fit)
        self.amps_data_norm = self.amps_data/self.A_fit #normalised by A; pump also divided by A0
        self.amps_data_norm[0] /= self.A0_fit
        self.amps_data_norm_no_sech = self.amps_data_norm/self.sech_fit

        cutoff = self.max_abs_mode_no + 1
        weights = self.weights[:cutoff]

        def minfunc(sol_locs, envelope, amps_to_fit):
            #input is locations of all solitons except the initial one, which is fixed at 0.
            Etilde = self.gen_Etilde(np.concatenate(([0], sol_locs)), envelope)[:cutoff]
            amps = amps_to_fit[:cutoff]
            return np.sum(weights*((np.abs(Etilde)**0.5 - amps**0.5))**2)

        init_igw = 1/self.rel_gauss_w_lims[0]/self.ns_fit
        final_igw = 1/self.rel_gauss_w_lims[1]/self.ns_fit
        inv_gauss_ws = np.linspace(np.sqrt(init_igw), np.sqrt(final_igw), self.gauss_w_steps)**2

        no_tries = 0
        done = False 

        print("Fitting soliton pattern:")

        while(done == False):
            self.ns = int(np.rint(np.random.normal(self.ns_init, ns_stdev)))
            print("\rAttempt %i with %i solitons  " % (no_tries+1, self.ns), end="")
            sol_locations_guess = np.linspace(0, 1, self.ns, endpoint=False)
            for igw in inv_gauss_ws:
                sol_locations_guess += self.rand_coeff*igw*(np.random.random(self.ns)-0.5)
                sol_locations_guess = (sol_locations_guess - sol_locations_guess[0])%1

                gauss_env = np.exp(-(igw*self.n)**2) # Gaussian envelope
                amps_gauss = self.amps_data_norm_no_sech*gauss_env
                sol_locations_fit = np.pad(opt.minimize(minfunc, sol_locations_guess[1:],\
                                                        args=(gauss_env, amps_gauss)).x, (1,0))

                #Badness of fit:
                bof = minfunc(sol_locations_fit, self.sech_fit, self.amps_data_norm)

                if plot_all_fits:
                    E_fit, Etilde_fit = self.gen_Etilde(sol_locations_fit, self.sech_fit, True)
                    self.comb_plot_dB(plt, self.amps_data, -50, 2,'C0', label='Measured')
                    self.comb_plot_dB(plt, self.A_fit*Etilde_fit, -50, 1, 'C1', label='Fitted')
                    plt.legend(frameon=False)
                    plt.xlabel("Mode number")
                    plt.ylabel("Power /dBm")
                    plt.title("Badness of fit = %g" % bof)
                    plt.show()

                if(bof < self.bof_threshold):
                    done = True
                    break
                sol_locations_guess = sol_locations_fit
            no_tries += 1

        #One last fit, but this time with the original sech envelope
        self.sol_locations_fit = np.pad(opt.minimize(minfunc, sol_locations_fit[1:],\
                                            args=(self.sech_fit, self.amps_data_norm)).x, (1,0))
        bof = minfunc(self.sol_locations_fit, self.sech_fit, self.amps_data_norm)
        print("\nSolved in %i attempts, badness of fit = %g" % (no_tries, bof))

        self.E_fit, self.Etilde_fit = self.gen_Etilde(self.sol_locations_fit, self.sech_fit, True)
        
        if plot_solution:
            self.comb_plot_dB(plt, self.amps_data, -50, 2,'C0', label='Measured')
            self.comb_plot_dB(plt, self.A_fit*self.Etilde_fit, -50, 1, 'C1', label='Fitted')
            plt.legend(frameon=False)
            plt.xlabel("Mode number")
            plt.ylabel("Power /dBm")
            plt.show()

            E_fit_norm = np.real(self.E_fit)/np.max(np.real(self.E_fit))
            plt.plot(np.arange(self.N)/self.N, E_fit_norm, lw=1)
            plt.xlabel(r"Normalized time $\tau$")
            plt.ylabel("E field amplitude (a.u.)")
            plt.show()
            
            self.sol_locations_fit = np.sort(self.sol_locations_fit%1)
            self.sol_spacings_fit = np.diff(np.concatenate((self.sol_locations_fit, [1])))
            plt.plot(self.sol_spacings_fit, '.')
            plt.xlabel(r"Soliton index $r$")
            plt.ylabel(r"Normalized spacing $\Delta\tau_r$")
            plt.show()
            
     
    def compare_with_measured_phases(self, phases_filename, plot=True):

        phase_data = np.genfromtxt(phases_filename, delimiter=',')
        """First column is frequency of each comb line in THz
           Second column is phase shift applied to that comb line in radians"""
        
        self.f_phasedata = phase_data[:,1]
        self.phase_meas = bmpp(phase_data[:,2]) # puts phase between -pi and pi
        
        self.n_phase = np.rint((self.f_phasedata - self.fpump)/self.frep).astype(int) #mode number
        fitted_Etilde = self.Etilde_fit[self.n_phase]
        self.phase_fit = np.angle(fitted_Etilde)

        self.chirality, self.D0, self.D1, self.D2, self.residual_offset, self.keep_mask =\
            find_chirality_and_dispersion(self.phase_fit, self.phase_meas, self.n_phase, self.N)

        if plot:
            plt.plot(self.n_phase, self.residual_offset, '.', ms=2, label="Outliers")
            plt.plot(self.n_phase[self.keep_mask], self.residual_offset[self.keep_mask], '.', ms=2)
            plt.xlabel("Mode number")
            plt.ylabel("Residual phase offset /rad")
            plt.show()
        
        print("Chirality = %i" % self.chirality)