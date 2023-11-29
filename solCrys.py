# -*- coding: utf-8 -*-

import solCrys_functions as solc

spectrum_filename = 'Experimental_Data\Spectrum_1.csv'
phases_filename = 'Experimental_Data\Phase_1.csv'

sc = solc.solCrys()
sc.import_spectrum(spectrum_filename)
sc.fit_pattern(plot_ns_fits=True)
sc.compare_with_measured_phases(phases_filename)