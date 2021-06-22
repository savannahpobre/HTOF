import os
import numpy as np
from astropy.io import ascii
from htof.parse import HipparcosRereductionJavaTool

# Directory containing the residual records corresponding to the Java tool data
test_data_directory = os.path.join(os.getcwd(), '/home/dmichalik/HIPPARCOS_REREDUCTION/DATASETS/ASCII_res')

# Read list of the 6617 discrepant sources discussed in Brandt et al. 2021, Section 4
discrepant = ascii.read("htof/data/hip21_java_nobs_discrepant.txt", names=["HIP", "diffNobs"])
numDis = len(discrepant[discrepant['diffNobs'] > 0])

# Instantiate a data parser
data = HipparcosRereductionJavaTool()

# For each source, Let's store:
# HIP
# catalog_f2
# htof_f2 without ad-hoc correction
# htof_f2 with ad-hoc correction
# The difference of the htof_f2 with ad-hoc correction and the catalog f2
results = np.zeros((numDis,5))

for idx, hip_id in enumerate(discrepant[discrepant['diffNobs'] > 0]["HIP"]):
    results[idx][0] = hip_id
    # parse data without ad-hoc correction, compute and store f2
    data.parse(star_id=hip_id, intermediate_data_directory=test_data_directory, attempt_adhoc_rejection=False)
    results[idx][1] = data.catalog_f2
    results[idx][2] = data.f2
    # parse data with ad-hoc correction, compute and store f2
    data.parse(star_id=hip_id, intermediate_data_directory=test_data_directory, attempt_adhoc_rejection=True)
    results[idx][3] = data.f2
    results[idx][4] = data.f2 - data.catalog_f2

# write out results
ascii.write(results, 'f2_comparison_results.csv', format='csv',names=["HIP",
        "catalog_f2", "htof_f2_without", "htof_f2", "difference"])
