# -*- coding: utf-8 -*-

import solCrys_functions as solc

spectrum_filename = 'Experimental_Data\Spectrum_5.csv'
phases_filename = 'Experimental_Data\Phase_5.csv'

#Set epb_rel_cutoff < 7 for combs with few, closely spaced solitons
sc = solc.solCrys(epb_rel_cutoff=7) 
sc.import_spectrum(spectrum_filename)
sc.fit_pattern(plot_ns_fits=True)
sc.compare_with_measured_phases(phases_filename)