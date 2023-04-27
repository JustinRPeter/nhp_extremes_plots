1. 001_plot_prcp_climatology.ipynb => Plot monthly climatology for various NRM regions using awrams package.
2. 002_plot_clim_extremes_flood.ipynb => Calculation and plot change in flood scenario for various NRM regions using awrams package.
3. 002_plot_clim_extremes_flood.py => .py version of "002_plot_clim_extremes_flood.ipynb".
4. 003a_clim_extremes_flood_grid_based.ipynb => Grid based flood scenario analysis for one ensemble only (used break in loop); it contains function for EV1, GEV (using corrected L-moments);
												first doing spatial mean over a NRM region for historical (AWAP) and future period seperately then deduction.
5. 003b_clim_extremes_flood_grid_based.py => .py version of "003a_clim_extremes_flood_grid_based.ipynb"; it is run with shell script.
6. 003c_plot_clim_extremes_flood.ipynb => Boxplot for grid based flood scenario; this is the default boxplot - lower: 25th and Upper: 75th percentile.
7. 003d_clim_extremes_flood_purely_grid_based.ipynb => Grid based flood scenario analysis for one ensemble only (used break in loop); it contains function for EV1, GEV (using corrected L-moments);
													first calculate difference between historical (GCM) and future period then spatial mean over a NRM region. This is preferred approach.
8. 003e_clim_extremes_flood_purely_grid_based_test_pr.ipynb => similar to "003d_clim_extremes_flood_purely_grid_based.ipynb" though wrong package for GEV.
9. **003f_plot_clim_extremes_flood_10th_90th_pctl.ipynb => Boxplot for grid based flood scenario; this is modified boxplot - lower: 10th and Upper: 90th percentile.This plot is preferred.**
10. **004a_clim_extremes_flood_grid_based_MN.py =>  .py version of "003d_clim_extremes_flood_purely_grid_based.ipynb".**