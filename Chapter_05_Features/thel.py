import os
import pandas as pd
import numpy as np
 
adult_filename = os.path.join(os.path.dirname(__file__), "Ch5_ExtractingFeatures_Chi2PearsonPCA_adult.data")
print(adult_filename)
 
adult = pd.read_csv(adult_filename, header=None, names=["Age", "Work-Class", "fnlwgt", "Education",
                                                        "Education-Num", "Marital-Status", "Occupation",
                                                        "Relationship", "Race", "Sex", "Capital-gain",
                                                        "Capital-loss", "Hours-per-week", "Native-Country",
                                                        "Earnings-Raw"])
print(adult[:10])
 