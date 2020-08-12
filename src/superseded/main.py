#####################################################################################################################
##### Open-Source Seismic Risk Assessment, OpenSRA(TM)
##### 
##### Copyright(c) 2020-2022 The Regents of the University of California and 
##### Slate Geotechnical Consultants. All Rights Reserved.
##### 
##### Main function
##### 
##### Created: April 27, 2020
##### @author: Barry Zheng (Slate Geotechnical Consultants)
#####################################################################################################################


#####################################################################################################################
##### Required packages
import model
# import pandas as pd
# from scipy.stats import norm, multivariate_normal
# import matplotlib.pyplot as plt
# import openpyxl, time


#####################################################################################################################
##### main function
#####################################################################################################################				  
def main():
	assess = model.assessment()
	print('Created the object "assess" under the class "assessment".')
	return assess
	
	
#####################################################################################################################
##### cue run
#####################################################################################################################				  
if __name__ == '__main__':
	main()