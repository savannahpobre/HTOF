#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import glob
import datetime

import abc

import matplotlib.pyplot as plt


class IntermediateDataParser(object):
    """
    Base class for parsing Hip1 and Hip2 data. self.epoch, self.covariance_matrix and self.scan_angle are saved
    as panda dataframes. use .values (e.g. self.epoch.values) to call the ndarray version.
    """
    def __init__(self, scan_angle=None, covariance_matrix=None, epoch=None, residuals=None):
        self.scan_angle = scan_angle
        self.covariance_matrix = covariance_matrix
        self.epoch = epoch
        self.residuals = residuals

    @staticmethod
    def read_intermediate_data_file(star_hip_id, intermediate_data_directory, skiprows, header):
        filepath = os.path.join(os.path.join(intermediate_data_directory, '**/'), '*' + star_hip_id + '*')
        filepath_list = glob.glob(filepath, recursive=True)
        if len(filepath_list) > 1:
            raise ValueError('More than one input file with hip id {0} found'.format(star_hip_id))
        data = pd.read_csv(filepath_list[0], sep='[\s|]+', skiprows=skiprows, header=header, engine='python')
        return data

    @abc.abstractmethod
    def parse(self, star_id, intermediate_data_parent_directory, **kwargs):
        pass

    @staticmethod
    def _calculate_covariance_matrices(scan_angles):
        pass


class HipparcosOriginalData(IntermediateDataParser):
    def __init__(self, scan_angle=None, covariance_matrix=None, epoch=None, residuals=None):
        super(HipparcosOriginalData, self).__init__(scan_angle=scan_angle,
                                                    covariance_matrix=covariance_matrix,
                                                    epoch=epoch, residuals=residuals)

    def parse(self, star_hip_id, intermediate_data_directory, data_choice='NDAC'):
        """
        :param star_hip_id: a string which is just the number for the HIP ID.
        :param intermediate_data_directory: the path (string) to the place where the intermediate data is stored, e.g.
                Hip2/IntermediateData/resrec
                note you have to specify the file resrec or absrec. We use the residual records, so specify resrec.
        :param data_choice: 'FAST' or 'NDAC'. This slightly affects the scan angles. This mostly affects
        the residuals which are not used.
        """
        if (data_choice is not 'NDAC') and (data_choice is not 'FAST'):
            raise ValueError('data choice has to be either NDAC or FAST')
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory,
                                                skiprows=0, header='infer')
        # select either the data from the NDAC or the FAST consortium.
        data = data[data['IA2'] == data_choice[0]]
        # compute scan angles and observations epochs according to van Leeuwen & Evans 1997, eq. 11 & 12.
        self.scan_angle = np.arctan2(data['IA3'], data['IA4'])  # unit radians
        self.epoch = data['IA6'] / data['IA3'] + 1991.25
        self.residuals = data['IA8']  # unit milli-arcseconds (mas)


class HipparcosRereductionData(IntermediateDataParser):
    def __init__(self, scan_angle=None, covariance_matrix=None, epoch=None, residuals=None):
        super(HipparcosRereductionData, self).__init__(scan_angle=scan_angle,
                                               covariance_matrix=covariance_matrix,
                                               epoch=epoch, residuals=residuals)

    def parse(self, star_hip_id, intermediate_data_directory, **kwargs):
        data = self.read_intermediate_data_file(star_hip_id, intermediate_data_directory, skiprows=1, header=None)
        # compute scan angles and observations epochs from van Leeuwen 2007, table G.8
        # see also Figure 2.1, section 2.5.1, and section 4.1.2
        self.scan_angle = np.arctan2(data[3], data[4])  # data[3] = cos(psi), data[4] = sin(psi)
        self.epoch = data[1] + 1991.25
        self.residuals = data[5]  # unit milli-arcseconds (mas)


class AstrometricFitter(object):
    """
    :param covariance_matrices: ndarray of length epoch times with 2x2 covariance matrices for each epoch
    :param epoch_times: the times for each epoch.
    """
    def __init__(self, covariance_matrices, epoch_times,
                 astrometric_chi_squared_matrices=None, astrometric_solution_vector_components=None):
        self.covariance_matrices = covariance_matrices
        self.epoch_times = epoch_times
        if astrometric_solution_vector_components is None:
            self.astrometric_solution_vector_components = self._init_astrometric_solution_vectors()
        if astrometric_chi_squared_matrices is None:
            self.astrometric_chi_squared_matrices = self._init_astrometric_chi_squared_matrices()

    def fit_line(self, ra_vs_epoch, dec_vs_epoch):
        """
        :param ra_vs_epoch: 1d array of right ascension, ordered the same as the covariance matrices and epochs.
        :param dec_vs_epoch: 1d array of declination, ordered the same as the covariance matrices and epochs.
        :return:
        """
        return np.linalg.solve(self._chi2_matrix(), self._chi2_vector(ra_vs_epoch=ra_vs_epoch,
                                                                      dec_vs_epoch=dec_vs_epoch))

    def _chi2_matrix(self):
        return np.sum(self.astrometric_chi_squared_matrices, axis=0)

    def _chi2_vector(self, ra_vs_epoch, dec_vs_epoch):
        ra_solution_vecs = self.astrometric_solution_vector_components['ra']
        dec_solution_vecs = self.astrometric_solution_vector_components['dec']
        # sum together the individual solution vectors for each epoch
        return np.dot(ra_vs_epoch, ra_solution_vecs) + np.dot(dec_vs_epoch, dec_solution_vecs)

    def _init_astrometric_solution_vectors(self):
        num_epochs = len(self.epoch_times)
        astrometric_solution_vector_components = {'ra': np.zeros((num_epochs, 4)),
                                                  'dec': np.zeros((num_epochs, 4))}
        for epoch in range(num_epochs):
            inverse_det_c = 1 / np.linalg.det(self.covariance_matrices[epoch])
            a, b, c, d = unpack_elements_of_matrix(self.covariance_matrices[epoch])
            epoch_time = self.epoch_times[epoch]
            ra_vec, dec_vec = np.zeros(4).astype(np.float64), np.zeros(4).astype(np.float64)
            ra_vec[0] = -inverse_det_c * (-2 * d * epoch_time)
            ra_vec[1] = -inverse_det_c * ((b + c) * epoch_time)
            ra_vec[2] = -inverse_det_c * (-2 * d)
            ra_vec[3] = -inverse_det_c * (b + c)

            dec_vec[0] = -inverse_det_c * ((b + c) * epoch_time)
            dec_vec[1] = -inverse_det_c * (- 2 * a * epoch_time)
            dec_vec[2] = -inverse_det_c * (b + c)
            dec_vec[3] = -inverse_det_c * (- 2 * a)

            astrometric_solution_vector_components['ra'][epoch] = ra_vec
            astrometric_solution_vector_components['dec'][epoch] = dec_vec
        return astrometric_solution_vector_components

    def _init_astrometric_chi_squared_matrices(self):
        num_epochs = len(self.epoch_times)
        astrometric_chi_squared_matrices = np.zeros((num_epochs, 4, 4))
        for epoch in range(num_epochs):
            inverse_det_c = 1 / np.linalg.det(self.covariance_matrices[epoch])
            a, b, c, d = unpack_elements_of_matrix(self.covariance_matrices[epoch])
            epoch_time = self.epoch_times[epoch]

            A = np.zeros((4, 4))

            A[:, 0] = inverse_det_c * np.array([2 * d * epoch_time,
                                                (-b - c) * epoch_time,
                                                2 * d,
                                                (-b - c)])
            A[:, 1] = inverse_det_c * np.array([(-b - c) * epoch_time,
                                                2 * a * epoch_time,
                                                (-b - c),
                                                2 * a])
            A[:, 2] = inverse_det_c * np.array([2 * d * epoch_time ** 2,
                                                (-b - c) * epoch_time ** 2,
                                                2 * d * epoch_time,
                                                (-b - c) * epoch_time])
            A[:, 3] = inverse_det_c * np.array([(-b - c) * epoch_time ** 2,
                                                2 * a * epoch_time ** 2,
                                                (-b - c) * epoch_time,
                                                2 * a * epoch_time])

            astrometric_chi_squared_matrices[epoch] = A
        return astrometric_chi_squared_matrices


def unpack_elements_of_matrix(matrix):
    return matrix.flatten()


def chi2_vector_components_single_epoch(covariance_matrix, epoch_delta_t, ra, dec):
    """
    :param covariance_matrix:
    :param epoch_delta_t:
    :param ra:
    :param dec:
    :return: the vector c. Where the vector c and matrix A are such that the solution v to A.v = c
    gives v = (mu_alpha, mu_delta, alpha_0, delta_0)
    where alpha_0 and delta_0 are the mean RA and DEC of the star and
    mu_alpha and mu_delta are the best fit proper motions in the RA and DEC
    """
    det_c = np.linalg.det(covariance_matrix)
    a, b, c, d = unpack_elements_of_matrix(covariance_matrix)
    vec = np.zeros(4).astype(np.float64)
    vec[0] = -1/det_c*(-2*d*epoch_delta_t*ra + (b + c)*dec*epoch_delta_t)
    vec[1] = -1/det_c*((b+c)*ra*epoch_delta_t - 2*a*dec * epoch_delta_t)
    vec[2] = -1/det_c*(-2*d*ra+(b+c)*dec)
    vec[3] = -1/det_c*((b+c)*ra-2*a*dec)
    return vec


def chi2_matrix_and_vec_old(covariance_matrix, epoch_delta_t, ra, dec):
    """
    :param covariance_matrix:
    :param epoch_delta_t:
    :param ra:
    :param dec:
    :return: the matrix A. Where the vector c and matrix A are such that the solution v to A.v = c
    gives v = (mu_alpha, mu_delta, alpha_0, delta_0)
    where alpha_0 and delta_0 are the mean RA and DEC of the star and
    mu_alpha and mu_delta are the best fit proper motions in the RA and DEC
    """
    det_c = np.linalg.det(covariance_matrix)
    a, b, c, d = unpack_elements_of_matrix(covariance_matrix)
    vec = np.zeros(4).astype(np.float64)
    vec[0] = -1/det_c*(-2*d*epoch_delta_t*ra + (b + c)*dec*epoch_delta_t)
    vec[1] = -1/det_c*((b+c)*ra*epoch_delta_t - 2*a*dec * epoch_delta_t)
    vec[2] = -1/det_c*(-2*d*ra+(b+c)*dec)
    vec[3] = -1/det_c*((b+c)*ra-2*a*dec)

    A = np.zeros((4, 4))
    A[:, 0] = 1 / det_c * np.array([2*d*epoch_delta_t,
                                (-b-c)*epoch_delta_t,
                                2*d,
                                (-b-c)])
    A[:, 1] = 1 / det_c * np.array([(-b-c)*epoch_delta_t,
                                    2*a*epoch_delta_t,
                                    (-b-c),
                                    2*a])
    A[:, 2] = 1 / det_c * np.array([2*d*epoch_delta_t**2,
                                    (-b-c)*epoch_delta_t**2,
                                    2*d*epoch_delta_t,
                                    (-b-c)*epoch_delta_t])
    A[:, 3] = 1 / det_c * np.array([(-b-c)*epoch_delta_t**2,
                                    2*a*epoch_delta_t**2,
                                    (-b-c)*epoch_delta_t,
                                    2*a*epoch_delta_t])
    return A, vec


def chi2_matrix_many_epochs(covariance_matrices, epoch_delta_ts, ras, decs):
    chi2_matrix = np.zeros((4, 4)).astype(np.float64)
    vec = np.zeros(4).astype(np.float64)
    for covariance_matrix, epoch_delta_t, ra, dec in zip(covariance_matrices, epoch_delta_ts, ras, decs):
        single_epoch_chi2, single_epoch_vec = chi2_matrix_and_vec_old(covariance_matrix, epoch_delta_t, ra, dec)
        chi2_matrix += single_epoch_chi2
        vec += single_epoch_vec
    return chi2_matrix, vec


def line_of_best_fit(astrometric_data):
    """
    :param astrometric_data: dictionary with covariance_matrix, epoch_delta_t, ra, dec keys where
    astrometric_data['epoch_delta_t'][2] = the delta t from the center of the 2nd data acquisition epoch
    """
    chi2_matrix, vec = chi2_matrix_many_epochs(covariance_matrices=astrometric_data['covariance_matrix'],
                                               epoch_delta_ts=astrometric_data['epoch_delta_t'],
                                               ras=astrometric_data['ra'],
                                               decs=astrometric_data['dec'])
    ra_dec_solution_vector = np.linalg.solve(chi2_matrix, vec)
    return ra_dec_solution_vector


"""
integration tests
"""


def generate_parabolic_astrometric_data(correlation_coefficient=0.0, sigma_ra=0.1, sigma_dec=0.1, crescendo=False):
    astrometric_data = {}
    num_measurements = 20
    mu_ra, mu_dec = -1, 2
    acc_ra, acc_dec = -0.1, 0.2
    ra0, dec0 = -30, 40
    epoch_start = 0
    epoch_end = 200
    astrometric_data['epoch_delta_t'] = np.linspace(epoch_start, epoch_end, num=num_measurements)
    astrometric_data['dec'] = dec0 + astrometric_data['epoch_delta_t']*mu_dec + \
                              1 / 2 * acc_dec * astrometric_data['epoch_delta_t'] ** 2
    astrometric_data['ra'] = ra0 + astrometric_data['epoch_delta_t']*mu_ra + \
                             1 / 2 * acc_ra * astrometric_data['epoch_delta_t'] ** 2
    cc = correlation_coefficient
    astrometric_data['covariance_matrix'] = np.zeros((num_measurements, 2, 2))
    astrometric_data['covariance_matrix'][:] = np.array([[sigma_ra**2, sigma_ra*sigma_dec*cc],
                                                       [sigma_ra*sigma_dec*cc, sigma_dec**2]])
    if crescendo:
        astrometric_data['covariance_matrix'][:, 0, 0] *= np.linspace(1/10, 4, num=num_measurements)
        astrometric_data['covariance_matrix'][:, 1, 1] *= np.linspace(4, 1/10, num=num_measurements)
    astrometric_data['solution_vector'] = np.array([ra0, dec0, mu_ra, mu_dec])
    return astrometric_data


def plot_fitting_to_curved_astrometric_data(crescendo=False):
    astrometric_data = generate_parabolic_astrometric_data(correlation_coefficient=0, sigma_ra=5E2, sigma_dec=5E2, crescendo=crescendo)
    # solving
    fitter = AstrometricFitter(covariance_matrices=astrometric_data['covariance_matrix'],
                               epoch_times=astrometric_data['epoch_delta_t'])
    solution_vector = fitter.fit_line(ra_vs_epoch=astrometric_data['ra'],
                                      dec_vs_epoch=astrometric_data['dec'])
    # plotting
    plt.figure()
    plt.errorbar(astrometric_data['epoch_delta_t'], astrometric_data['ra'],
                 xerr=0, yerr=np.sqrt(astrometric_data['covariance_matrix'][:, 0, 0]),
                 fmt='ro', label='RA')
    plt.errorbar(astrometric_data['epoch_delta_t'], astrometric_data['dec'],
                 xerr=0, yerr=np.sqrt(astrometric_data['covariance_matrix'][:, 1, 1]),
                 fmt='bo', label='DEC')
    continuous_t = np.linspace(np.min(astrometric_data['epoch_delta_t']),
                               np.max(astrometric_data['epoch_delta_t']), num=200)
    ra0, dec0, mu_ra, mu_dec = solution_vector
    plt.plot(continuous_t, ra0 + mu_ra * continuous_t, 'r', label='RA fit')
    plt.plot(continuous_t, dec0 + mu_dec * continuous_t, 'b', label='DEC fit')
    plt.xlabel('$\Delta$ epoch')
    plt.ylabel('RA or DEC')
    plt.legend(loc='best')
    plt.title('RA and DEC linear fit using Covariance Matrices')


if __name__ == "__main__":
    plot_fitting_to_curved_astrometric_data(crescendo=False)
    plot_fitting_to_curved_astrometric_data(crescendo=True)
    plt.show()
