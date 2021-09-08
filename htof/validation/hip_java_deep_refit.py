import numpy as np
from argparse import ArgumentParser
import warnings
from htof.parse import partitions, HipparcosRereductionJavaTool


def calculate_sum_chi2_partials(rejects_from_each_orbit, orbit_index, orbit_multiplicity, orbits_to_keep, _orbit_factors,
             residual_factors_transpose_scaled, mask_rejected_resid):
    end_index = orbit_index + orbit_multiplicity - rejects_from_each_orbit
    for s, e in zip(orbit_index, end_index):
        orbits_to_keep[s:e] = True
    # now we want to try a variety of deleting orbits and sliding the other orbits
    # upward to fill the vacancy.
    # this pops the orbits out and shifts all the orbits after:
    orbit_factors = _orbit_factors[orbits_to_keep]
    # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
    # shifted orbits, and calculates the chi2 partials vector per orbit:
    chi2_vector = (residual_factors_transpose_scaled * orbit_factors)
    # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
    sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[mask_rejected_resid], axis=0) ** 2))
    # reset for the next loop:
    orbits_to_keep[:] = False
    return rejects_from_each_orbit, sum_chisquared_partials


def find_epochs_to_reject_java_largest(data, n_additional_reject, orbit_number):
    # this is for any java tool object where n_additional_reject is greater than 3.
    # we assume the scan angles and times of rows in the same orbit are similar, therefore we only have
    # to try all combinations of distributing n_additional_reject rejected epochs among N orbits
    # calculate the chisquared partials
    orbit_prototypes, orbit_index, orbit_multiplicity = np.unique(orbit_number, return_index=True,
                                                                  return_counts=True)
    num_unique_orbits = len(orbit_prototypes)
    sin_scan = np.sin(data.scan_angle.values)
    cos_scan = np.cos(data.scan_angle.values)
    dt = data.epoch - 1991.25
    resid_reject_idx = [len(data) - 1 - i for i in
                        range(int(n_additional_reject))]  # always reject the repeated observations.
    # need to iterate over popping orbit combinations
    orbits_to_keep = np.zeros(len(data), dtype=bool)
    residuals_to_keep = np.ones(len(data), dtype=bool)
    residuals_to_keep[resid_reject_idx] = False

    residual_factors = (data.residuals.values / data.along_scan_errs.values ** 2)[residuals_to_keep]
    mask_rejected_resid = (data.along_scan_errs.values > 0).astype(bool)[residuals_to_keep]
    _orbit_factors = np.array([sin_scan, cos_scan, dt * sin_scan, dt * cos_scan]).T
    candidate_orbit_rejects = []
    candidate_orbit_chisquared_partials = []

    residual_factors_transpose = residual_factors.reshape(-1,1)
    residual_factors_transpose_scaled = residual_factors_transpose * 2

    for rejects_from_each_orbit in partitions(n_additional_reject, num_unique_orbits):
        if np.any(rejects_from_each_orbit > orbit_multiplicity):
            # ignore any trials of rejects that put e.g. 10 rejects into an orbit with only 4 observations.
            continue
        rejects_from_each_orbit, sum_chisquared_partials = calculate_sum_chi2_partials(rejects_from_each_orbit, orbit_index, orbit_multiplicity, orbits_to_keep, _orbit_factors,
             residual_factors_transpose_scaled, mask_rejected_resid)
        if sum_chisquared_partials > 1:
            # if the solution is bad, don't even save it. This will reduce memory usage.
            continue
        candidate_orbit_rejects.append(rejects_from_each_orbit)
        candidate_orbit_chisquared_partials.append(sum_chisquared_partials)
    rejects_from_each_orbit = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    # now transform rejects_from_each_orbit into actual orbit indices that we are going to reject.
    end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
    for s, e in zip(orbit_index, end_index):
        orbits_to_keep[s:e] = True
    orbit_reject_idx = np.where(~orbits_to_keep)[0]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        warnings.warn("Attempted to fix the data corruption, but the chisquared partials are larger than 0.5. "
                      "Treat this source with caution.", UserWarning)

    return {'residual/along_scan_error': list(resid_reject_idx), 'orbit/scan_angle/time': list(orbit_reject_idx)}


if __name__ == "__main__":
    parser = ArgumentParser(description='Script for refitting the java tool IAD. '
                                        'This will output a .txt file for every hip source provided in the inlist'
                                        ' Each .txt will contain the epochs to reject to fix the data corruption')
    parser.add_argument("-dir", "--iad-directory", required=True, default=None,
                        help="full path to the intermediate data directory")
    parser.add_argument("-i", "--inlist", required=False, default=None,
                        help=".txt file with the list of sources you want to refit.")
    parser.add_argument("--debug", action='store_true', default=False, required=False,
                        help='If true, this will run the refit test on only 500 sources. Useful to check for '
                             'filepath problems before running the full test on all ~100000 sources.')

    args = parser.parse_args()
    import time

    hip_ids = np.genfromtxt(args.inlist).flatten().astype(int)
    if args.debug:
        # test a source with 2 corrupted observations and 5.
        hip_ids = [114114, 18517]

    # do the fit.
    for hip_id in hip_ids:
        data = HipparcosRereductionJavaTool()
        # get info on the IAD without doing any rejection:
        header, raw_iad = data.parse(star_id=hip_id, intermediate_data_directory=args.iad_directory,
                                     attempt_adhoc_rejection=False, reject_known=False)
        n_transits, n_expected_transits = header['first']['NRES'], header['second']['NOB']
        n_additional_reject = int(n_transits) - int(n_expected_transits)
        orbit_number = raw_iad[0].values
        correct_id = header['first']['HIP']
        additional_rejected_epochs = find_epochs_to_reject_java_largest(data, n_additional_reject, orbit_number)
        f = open(f"{str(int(correct_id))}.txt", "w")
        f.write(str(additional_rejected_epochs))
        f.close()

