from htof.validation.utils import refit_hip1_object, refit_hip2_object, load_hip2_catalog, refit_hip21_object
from htof.validation.utils import load_hip1_dm_annex, load_hip2_seven_p_annex, load_hip2_nine_p_annex
import os
from astropy.table import Table
import numpy as np
from argparse import ArgumentParser
from glob import glob

from multiprocessing import Pool


class Engine(object):
    @staticmethod
    def format_result(result, hip_id, soltype):
        diffs, errors, chisq, partials = result[:4]
        plx, ra, dec, pm_ra, pm_dec, acc_ra, acc_dec, jerk_ra, jerk_dec = diffs
        return {'hip_id': hip_id, 'diff_ra': ra, 'diff_dec': dec, 'diff_plx': plx, 'diff_pm_ra': pm_ra, 'diff_pm_dec': pm_dec,
                'soltype': soltype, 'diff_acc_ra': acc_ra, 'diff_acc_dec': acc_dec, 'diff_jerk_ra': jerk_ra, 'diff_jerk_dec': jerk_dec,
                'chisquared': chisq, 'dxdparallax': partials[0], 'dxdra0': partials[1], 'dxddec0': partials[2], 'dxdmura': partials[3], 'dxdmudec': partials[4]}


class Hip1Engine(Engine):
    def __init__(self, dirname, use_parallax, hip_dm_g=None):
        self.dirname = dirname
        self.use_parallax = use_parallax
        self.hip_dm_g = hip_dm_g

    def __call__(self, hip_id):
        result = refit_hip1_object(self.dirname, hip_id, self.hip_dm_g, use_parallax=self.use_parallax)
        soltype = result[4]
        return self.format_result(result, hip_id, soltype)

    @staticmethod
    def convert_fname_to_hip_id(fname):
        # ideally this method shouldn't really be in the engine, but whatever for now.
        return os.path.basename(fname).split('.txt')[0]


class Hip2Engine(Engine):
    def __init__(self, dirname, use_parallax, catalog=None, seven_p_annex=None, nine_p_annex=None):
        self.dirname = dirname
        self.catalog = catalog
        self.use_parallax = use_parallax
        self.seven_p_annex = seven_p_annex
        self.nine_p_annex = nine_p_annex

    def __call__(self, hip_id):
        result = refit_hip2_object(self.dirname, hip_id, self.catalog, seven_p_annex=self.seven_p_annex,
                                   nine_p_annex=self.nine_p_annex, use_parallax=self.use_parallax)
        soltype = result[4]
        return self.format_result(result, hip_id, soltype[-1])

    @staticmethod
    def convert_fname_to_hip_id(fname):
        return os.path.basename(fname).split('.d')[0].split('HIP')[1]


class Hip21Engine(Engine):
    def __init__(self, dirname, use_parallax):
        self.dirname = dirname
        self.use_parallax = use_parallax

    def __call__(self, hip_id):
        result = refit_hip21_object(self.dirname, hip_id, use_parallax=self.use_parallax)
        soltype = result[4]
        return self.format_result(result, hip_id, soltype[-1])

    @staticmethod
    def convert_fname_to_hip_id(fname):
        return os.path.basename(fname).split('.d')[0].split('H')[1]


if __name__ == "__main__":
    parser = ArgumentParser(description='Script for refitting the entire hipparcos catalog, 1997 or 2007.'
                                        'This will output a csv type file. Each row gives'
                                        'the difference in the best-fit parameters and the catalog parameters '
                                        'along with other metrics of interest.')
    parser.add_argument("-dir", "--iad-directory", required=True, default=None,
                        help="full path to the intermediate data directory")
    parser.add_argument("-hr", "--hip-reduction", required=True, default=None, type=int,
                        help="integer. 1 for 1997 reduction, 2 for 2007 CD reduction, 21 for 2007 IADT tool.")
    parser.add_argument("-o", "--output-file", required=False, default=None,
                        help="The output filename, with .csv extension. E.g. hip1_refit.csv."
                             "Will default to hip_processid.csv.")
    parser.add_argument("-i", "--inlist", required=False, default=None,
                        help=".txt file with the list of sources you want to refit.")
    parser.add_argument("-c", "--cores", required=False, default=1, type=int,
                        help="Number of cores to use. Default is 1.")
    parser.add_argument("--ignore-parallax", required=False, action='store_true', default=False,
                        help="Whether or not to ignore parallax in the fits. Default is False, i.e. parallax"
                             "will be fit.")
    parser.add_argument("-cpath", "--catalog-path", required=False, default=None,
                        help="path to the Hip re-reduction main catalog, e.g. Main_cat.d. Only required"
                             "if using the 2007 CD data.")
    parser.add_argument("--debug", action='store_true', default=False, required=False,
                        help='If true, this will run the refit test on only 500 sources. Useful to check for '
                             'filepath problems before running the full test on all ~100000 sources.')

    args = parser.parse_args()

    # check arguments and assign values.
    if args.hip_reduction == 2 and args.catalog_path is None:
        raise ValueError('Hip 2 selected but no --catalog-path provided.')

    if args.output_file is None:
        output_file = 'hip' + (str)(args.hip_reduction) + '_refit' + (str)(os.getpid()) + '.csv'
    else:
        output_file = args.output_file
    if args.ignore_parallax:
        print('Warning: ignore_parallax flag is active. parallax will not be fit and so the standard errors will '
              'be slightly different compared to the catalog values for every source')

    # find the intermediate data files
    kwargs = {}
    if args.hip_reduction == 1:
        files = glob(os.path.join(args.iad_directory, '*.txt'))
        engine = Hip1Engine
        kwargs = {'hip_dm_g': load_hip1_dm_annex(os.path.join(args.iad_directory, 'hip_dm_g.dat'))}
    elif args.hip_reduction == 2:
        files = glob(os.path.join(args.iad_directory, '**/H*.d'))
        engine = Hip2Engine
        ninep_path = os.path.join(os.path.dirname(args.catalog_path), 'NineP_Cat.d')
        sevenp_path = os.path.join(os.path.dirname(args.catalog_path), 'SevenP_Cat.d')
        kwargs = {'catalog': load_hip2_catalog(args.catalog_path),
                  'seven_p_annex': load_hip2_seven_p_annex(sevenp_path), 'nine_p_annex': load_hip2_nine_p_annex(ninep_path)}
    else:
        files = glob(os.path.join(args.iad_directory, '**/H*.d'))
        engine = Hip21Engine

    # convert file names to a list of hip_ids
    hip_ids = [engine.convert_fname_to_hip_id(fname) for fname in files]

    if args.inlist is not None:
        # fit a specific set of hip_ids if a list is provided.
        hip_ids = np.genfromtxt(args.inlist).flatten().astype(int)
    if args.debug:
        # fit only a small subset of the desired sources if debugging.
        hip_ids = np.concatenate([hip_ids[:100], [93424, 581, 16468, 52599, 394, 37, 2]]).astype(int)
    hip_ids.sort()
    engine(args.iad_directory, not args.ignore_parallax, **kwargs)(93424)  # debug to make sure things work
    print('will fit {0} total hip {1} objects'.format(len(hip_ids), str(args.hip_reduction)))
    print('will save output table at', output_file)
    # do the fit.
    try:
        pool = Pool(args.cores)
        engine = engine(args.iad_directory, not args.ignore_parallax, **kwargs)
        data_outputs = pool.map(engine, hip_ids)
        out = Table(data_outputs)
        out.sort('hip_id')
        out.write(output_file, overwrite=True)
    finally:  # This makes sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

