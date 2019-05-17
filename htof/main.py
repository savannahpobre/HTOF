#!/usr/bin/env python
"""
Driver script for htof.
The Fitter class is what a user should use to both parse intermediate data and fit data
to the intermediate epochs.
"""

import numpy as np
import math
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy import interpolate

from htof.fit import AstrometricFitter
from htof.parse import HipparcosRereductionData, calculate_covariance_matrices, GaiaData, HipparcosOriginalData, \
                       fractional_year_epoch_to_jd


class Astrometry(object):
    parsers = {'GaiaDR2': GaiaData, 'Hip1': HipparcosOriginalData, 'Hip2': HipparcosRereductionData}

    def __init__(self, data_choice, star_id, intermediate_data_directory, fitter=None, data=None):
        if data is None:
            DataParser = self.parsers[data_choice]
            data = DataParser()
            data.parse(star_id=star_id,
                       intermediate_data_directory=intermediate_data_directory)
            data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=1E5)
        if fitter is None and data is not None:
            fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                       epoch_times=data.julian_day_epoch())
        self.data = data
        self.fitter = fitter

    def fit(self, ra_vs_epoch, dec_vs_epoch, central_epoch_dec=0, central_epoch_ra=0, central_epoch_fmt='MJD'):
        solution_vector = self.fitter.fit_line(ra_vs_epoch=ra_vs_epoch,
                                               dec_vs_epoch=dec_vs_epoch)
        solution_vector = self._shift_to_central_epoch(solution_vector, central_epoch_dec,
                                                       central_epoch_ra, central_epoch_fmt)
        return solution_vector

    @staticmethod
    def _shift_to_central_epoch(solution_vector, central_epoch_dec, central_epoch_ra, central_epoch_fmt):
        ra0, dec0, mu_ra, mu_dec = solution_vector
        if central_epoch_fmt == 'frac_year':
            if central_epoch_dec > 3000 or central_epoch_ra > 3000:
                warnings.warn('central epoch in RA or DEC was chosen to be > 3000. Are you sure this'
                              'is a fractional year date and not an MJD? If MJD, set central_epoch_fmt=MJD.',
                              UserWarning)
            central_epoch_dec = fractional_year_epoch_to_jd(central_epoch_dec, half_day_correction=True)
            central_epoch_ra = fractional_year_epoch_to_jd(central_epoch_ra, half_day_correction=True)
        ra0 += mu_ra * central_epoch_ra
        dec0 += mu_dec * central_epoch_dec
        return np.array([ra0, dec0, mu_ra, mu_dec])

"""
Utility functions which currently just makes plots of interest.
"""


def circle(radius, samples, radians=np.pi/2):
    x = np.arange(0, samples)
    return radius * np.cos(radians / (samples-1) * x),  radius * np.sin(radians / (samples-1) * x)


def plot_fit_to_orbit(ax, data, ra_orbit, dec_orbit, parametric_fit_vector, covariance_matrices, epochs):
    ra0, dec0, mu_ra, mu_dec = parametric_fit_vector
    fit_ra = ra0 + epochs * mu_ra  # x axis
    fit_dec = dec0 + epochs * mu_dec  # y axis
    angles = data.scan_angle.values.flatten()
    for ra, dec, angle, cov_matrix in zip(ra_orbit, dec_orbit, angles, covariance_matrices):
        ax = plot_error_ellipse(ax, mu=(ra, dec), cov_matrix=cov_matrix)
        ax.plot([ra + np.sin(angle)*.2, ra, ra - np.sin(angle)*.2],
                [dec - np.cos(angle)*.2, dec, dec + np.cos(angle)*.2], 'r', lw=2)  # plot scan angle
    ax.arrow(ra0, dec0, mu_ra*500, mu_dec*500, color='k', width=0.5)  # arrow
    ax.plot([ra0, ra0+mu_ra], [dec0, dec0+mu_dec], color='k', label=r'$\mu_{Hip}$', lw=7)  # arrow label
    ax.plot(ra0, dec0, 'gD', markersize=15, label='1991.25')
    ax.plot(ra0, dec0, 'kD', markersize=2)
    return ax


def plot_error_ellipse(ax, mu, cov_matrix, color="b"):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """
    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(cov_matrix)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)
    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)
    return ax


if __name__ == "__main__":
    plot_diagnostic_data = False
    plot_fake_orbit_fit = True
    if plot_diagnostic_data:
        data = HipparcosRereductionData()
        data.parse(intermediate_data_directory='/home/mbrandt21/Downloads/Hip2/IntermediateData/resrec',
                   star_id='27321')
        scan_angles = data.scan_angle.truncate(after=20)
        multiplier = 20
        covariances = calculate_covariance_matrices(scan_angles, cross_scan_along_scan_var_ratio=multiplier)
        f, ax = plt.subplots()
        for i in range(len(scan_angles)):
            center = data.julian_day_epoch()[i]
            ax = plot_error_ellipse(ax, mu=(center, 0), cov_matrix=covariances[i])
            ax.set_xlim((np.min(data.julian_day_epoch()), np.max(data.julian_day_epoch())))
            ax.set_ylim((-multiplier, multiplier))
            angle = scan_angles.values.flatten()[i]
            ax.plot([center, center -np.sin(angle)], [0, np.cos(angle)], 'k')
            ax.set_title('along scan angle {0} degrees east from the northern equatorial pole'.format(angle*180/np.pi))
        plt.axis('equal')
        plt.show()
    if plot_fake_orbit_fit:
        data = HipparcosRereductionData()
        data.parse(intermediate_data_directory='/home/mbrandt21/Downloads/Hip2/IntermediateData/resrec',
                   star_id='003850')
        multiplier = 100
        samples = 20
        fontsize = 22
        #epochs = data.julian_day_epoch()[:samples]
        #
        sample_orbit = np.loadtxt('/home/mbrandt21/Downloads/tim_orbit_fit_code/orbit.dat').T
        sample_orbit[1] *= -1 # fixing reversed axis.
        epochs, ra_vs_epoch, dec_vs_epoch = sample_orbit[0], sample_orbit[1], sample_orbit[2]
        samples = len(dec_vs_epoch)
        #
        data.calculate_inverse_covariance_matrices(cross_scan_along_scan_var_ratio=multiplier)
        fitter = AstrometricFitter(inverse_covariance_matrices=data.inverse_covariance_matrix,
                                   epoch_times=epochs)

        # ra_vs_epoch, dec_vs_epoch = circle(radius=100, samples=samples, radians=np.pi/1.3)
        solution_vector = fitter.fit_line(ra_vs_epoch=ra_vs_epoch,
                                          dec_vs_epoch=dec_vs_epoch)
        # plotting
        cov_matrices = calculate_covariance_matrices(data.scan_angle, cross_scan_along_scan_var_ratio=multiplier)*.02
        f, ax = plt.subplots(figsize=(8, 8))
        ax = plot_fit_to_orbit(ax, data, ra_vs_epoch, dec_vs_epoch, solution_vector, cov_matrices, epochs=epochs)
        ax.set_ylabel(r'Declination (mas)', fontsize=fontsize)
        ax.set_xlabel(r'Right Ascension (mas)', fontsize=fontsize)

        font = {'family': 'serif',
                'size': fontsize}
        plt.rc('font', **font)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.rc('mathtext', fontset="cm")  # fixing latex fonts
        plt.axis('equal')
        plt.legend(loc='best', prop={'size': 14})

        finely_sampled_orbit = np.loadtxt('/home/mbrandt21/Downloads/tim_orbit_fit_code/orbit_full.dat')
        finely_sampled_orbit[:, 1] *= -1 # fixing x axis.

        ax.plot(finely_sampled_orbit[:, 1], finely_sampled_orbit[:, 2], 'r')

        ra = interpolate.interp1d(finely_sampled_orbit[:, 0], finely_sampled_orbit[:, 1])
        dec = interpolate.interp1d(finely_sampled_orbit[:, 0], finely_sampled_orbit[:, 2])
        years = np.arange(1988, 1995)

        for year in years:
            t = (year - 1991.25) * 365.25
            ax.plot(ra(t), dec(t), 'k*', markersize=15)
            ax.plot(ra(t), dec(t), 'y*', markersize=14)
            ax.text(ra(t) + 1, dec(t) - 1, '%d' % (year))

        ax.plot(0, 0, 'D', markersize=15)
        ax.text(0, 0.7, 'Center of Mass', horizontalalignment='center')
        plt.tight_layout()
        plt.show()
