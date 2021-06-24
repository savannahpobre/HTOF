import os
import numpy as np
from astropy.io import ascii as ascii_astropy
from astropy.table import Table
from htof.parse import HipparcosRereductionJavaTool

# Directory containing the residual records corresponding to the Java tool data
# Redirect this to the directory containing the IAD of all (6617) sources
test_data_directory = os.path.join(os.getcwd(), 'htof/test/data_for_tests/Hip21')

# Read list of the 6617 discrepant sources discussed in Brandt et al. 2021, Section 4
discrepant = ascii_astropy.read("htof/data/hip21_java_nobs_discrepant.txt", names=["HIP", "diffNobs"])
numDis = len(discrepant[discrepant['diffNobs'] > 0])

# Instantiate a data parser
data = HipparcosRereductionJavaTool()

# For each source, Let's store:
# HIP
# catalog_f2
# htof_f2 without ad-hoc correction
# htof_f2 with ad-hoc correction
# The difference of the htof_f2 with ad-hoc correction and the catalog f2
results = np.zeros((numDis, 5))

hip_ids_to_parse = discrepant[discrepant['diffNobs'] > 0]["HIP"]
# hip_ids_to_parse = ['27321', '37515'] debug
for idx, hip_id in enumerate(hip_ids_to_parse):
    results[idx][0] = hip_id
    # parse data without ad-hoc correction, compute and store f2
    data.parse(star_id=hip_id, intermediate_data_directory=test_data_directory, attempt_adhoc_rejection=False)
    results[idx][1] = data.meta['catalog_f2']
    results[idx][2] = data.meta['calculated_f2']
    # parse data with ad-hoc correction, compute and store f2
    data.parse(star_id=hip_id, intermediate_data_directory=test_data_directory, attempt_adhoc_rejection=True)
    results[idx][3] = data.meta['calculated_f2']
    results[idx][4] = data.meta['calculated_f2'] - data.meta['catalog_f2']

# write out results
output_table = Table(results, names=["HIP", "catalog_f2", "htof_f2_without", "htof_f2", "difference"],
                     dtype=['i8', 'f8', 'f8', 'f8', 'f8'])
output_table.write('f2_comparison_results.csv', overwrite=True)
