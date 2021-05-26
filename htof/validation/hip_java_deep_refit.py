import numpy as np
from argparse import ArgumentParser
import copy
from scipy.special import comb as nchoosek
from htof.parse import partitions, HipparcosRereductionJavaTool


from multiprocessing import Pool


class Engine(object):
    def __init__(self, orbit_index, orbit_multiplicity, orbits_to_keep, orbit_factors,
                 residual_factors, mask_rejected_resid):
        self.orbit_index = orbit_index
        self.orbit_multiplicity = orbit_multiplicity
        self.orbits_to_keep = copy.deepcopy(orbits_to_keep)
        self.orbit_factors = orbit_factors
        self.residual_factors = residual_factors
        self.mask_rejected_resid = mask_rejected_resid

    def __call__(self, rejects_from_each_orbit):
        end_index = self.orbit_index + self.orbit_multiplicity - rejects_from_each_orbit
        for s, e in zip(self.orbit_index, end_index):
            self.orbits_to_keep[s:e] = True
        # now we want to try a variety of deleting orbits and sliding the other orbits
        # upward to fill the vacancy.
        # this pops the orbits out and shifts all the orbits after:
        orbit_factors = self.orbit_factors[self.orbits_to_keep].T
        # this simultaneously deletes one of the residuals, assigns the remaining residuals to the
        # shifted orbits, and calculates the chi2 partials vector per orbit:
        chi2_vector = (2 * self.residual_factors * orbit_factors).T
        # sum the square of the chi2 partials to decide for whether or not it is a stationary point.
        sum_chisquared_partials = np.sqrt(np.sum(np.sum(chi2_vector[self.mask_rejected_resid], axis=0) ** 2))
        # reset for the next loop:
        self.orbits_to_keep[:] = False
        return [rejects_from_each_orbit, sum_chisquared_partials]


def find_epochs_to_reject_java_large_parallelized(data, n_additional_reject, orbit_number, ncore):
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

    trials = []
    # restrict to trial sets of orbit rejections that are viable.
    for rejects_from_each_orbit in partitions(n_additional_reject, num_unique_orbits):
        if np.all(rejects_from_each_orbit <= orbit_multiplicity):
            trials.append(np.array(rejects_from_each_orbit, dtype=np.uint8))
    trials = np.array(trials, dtype=np.uint8)
    print(len(trials))
    try:
        pool = Pool(ncore)
        engine = Engine(orbit_index, orbit_multiplicity,
                        np.zeros(len(data), dtype=bool), _orbit_factors,
                        residual_factors, mask_rejected_resid)
        out = pool.map(engine, trials)
    finally:  # This makes sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()
    candidate_orbit_rejects = [out[i][0] for i in range(len(out))]
    candidate_orbit_chisquared_partials = [out[i][1] for i in range(len(out))]
    rejects_from_each_orbit = np.array(candidate_orbit_rejects)[np.argmin(candidate_orbit_chisquared_partials)]
    # now transform rejects_from_each_orbit into actual orbit indices that we are going to reject.
    end_index = orbit_index + orbit_multiplicity - np.array(rejects_from_each_orbit)
    for s, e in zip(orbit_index, end_index):
        orbits_to_keep[s:e] = True
    orbit_reject_idx = np.where(~orbits_to_keep)[0]
    if np.min(candidate_orbit_chisquared_partials) > 0.5:
        print("Attempted to fix the write out bug, but the chisquared partials are larger than 0.5. There are "
              "likely more additional rejected epochs than htof can handle.", UserWarning)

    return {'residual/along_scan_error': list(resid_reject_idx), 'orbit/scan_angle/time': list(orbit_reject_idx)}


if __name__ == "__main__":
    parser = ArgumentParser(description='Script for refitting the java tool IAD. '
                                        'This will output a .txt file for every hip source provided in the inlist'
                                        ' Each .txt will contain the epochs to reject to fix the write out bug')
    parser.add_argument("-dir", "--iad-directory", required=True, default=None,
                        help="full path to the intermediate data directory")
    parser.add_argument("-i", "--inlist", required=False, default=None,
                        help=".txt file with the list of sources you want to refit.")
    parser.add_argument("-c", "--cores", required=False, default=1, type=int,
                        help="Number of cores to use. Default is 1.")
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
        n_transits, n_expected_transits = header.iloc[1][4], header.iloc[0][2]
        n_additional_reject = int(n_transits) - int(n_expected_transits)
        orbit_number = raw_iad[0].values
        correct_id = header.iloc[0][0]
        norb = len(np.unique(orbit_number))
        comb = nchoosek(norb + n_additional_reject - 1, norb - 1)
        memory_size = comb * 8
        print(comb)
        #print(memory_size/1e6)
        if comb < 1.2*8145060:  # we just don't try it if there are too many combinations
            additional_rejected_epochs = find_epochs_to_reject_java_large_parallelized(data, n_additional_reject,
                                                                                       orbit_number, args.cores)
            f = open(f"{str(int(correct_id))}.txt", "w")
            f.write(str(additional_rejected_epochs))
            f.close()

