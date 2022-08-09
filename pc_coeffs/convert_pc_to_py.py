# python modules
import os

# directory to PC code
pc_base_code_dir = os.path.join(os.getcwd(),'To Share New')
# file = os.path.join(pc_base_code_dir, 'PC_Coeffs_CDF_Risk_array_function.m')
mat_file = os.path.join(pc_base_code_dir, 'PC_Coeffs_Simpler_CDF_Single_Integral_array_function.m')
out_dir = os.getcwd()
# out_file = os.path.join(out_dir,'pc.py') # file saved as pc.py in the same folder
out_file = os.path.join(out_dir,os.path.basename(mat_file).replace('.m','.py')) # file saved as pc.py in the same folder

# initialize list for storing converted linestrings
txt_list = []
txt_list.append('import numpy as np') # need numpy for vector operation
txt_list.append('from scipy.special import erf') # need erf function
txt_list.append('')
line_prev = ''
add_loop_flag = False
# loop through each line in the matlab code
with open(mat_file, 'r') as f:  # Open file for read
    for line in f:           # Read line-by-line
        line = line.strip()
        line = line.replace('%','#')
        line = '    '+line
        if '    #' in line:
            pass
        else:
            line = line.replace('...','\\') # nextline continuation operator
            line = line.replace(';','')
            line = line.replace('sigmaV) #','sigmaV):') # need colon for end of function header
            if '\\' in line_prev:
                line = '    '+line # python spacing
            line = line.replace('    function P_output=','def ') # functions start with def in python
            line = line.replace('.^','**') # power operator
            line = line.replace('.*','*') # element-wise multiply
            line = line.replace('./','/') # element-wise divide
            line = line.replace('=',' = ')
            line = line.replace('(1,:)','[0,:]') # python indicing start at 1
            line = line.replace('(2,:)','[1,:]')
            line = line.replace('(3,:)','[2,:]')
            line = line.replace('(4,:)','[3,:]')
            line = line.replace('(5,:)','[4,:]')
            line = line.replace('sigmaV) #','sigmaV):') # need colon for end of function header
            line = line.replace('end','') # python doesn't use end to end loops
            line = line.replace('length(muY)','muY.shape[1]') # length of y-dimension for matrix
            line = line.replace('length(v)','len(v)') # length of vector
            line = line.replace('zeros(1, n_sites)','np.zeros((1, n_sites))') # initializing zero matrix
            line = line.replace('zeros(2, n_sites)','np.zeros((2, n_sites))')
            line = line.replace('zeros(3, n_sites)','np.zeros((3, n_sites))')
            line = line.replace('zeros(4, n_sites)','np.zeros((4, n_sites))')
            line = line.replace('zeros(5, n_sites)','np.zeros((5, n_sites))')
            if 'P_output_1 = ' in line or 'P_output_2 = ' in line or 'P_output_3 = ' in line or 'P_output = ' in line or ', \\' in line_prev: # specific to lines that start with P_output...
                line = line.replace(' = ','=') 
                line = line.replace('    ','')
                line = line.replace(' ',', ')
                line = line.replace('=',' = ')
                line = line.replace('[','np.vstack((') # for stacking arrays vertically (e.g., 1x10 stacked on 1x10 to make 2x10)
                line = line.replace(']','))') # end of vstack function
                line = line.replace(', , ',', ')
                line = line.replace(', , , ',', ')
                line = line.replace(', , , , ',', ')
                line = '    '+line
                if ', \\' in line_prev:
                    line = '    '+line
            if 'P_output_1(' in line or 'P_output_2(' in line or 'P_output_3(' in line:
                line = line.replace('(','[')
                line = line.replace(')',']')
                line = line.replace(line[line.find('['):line.find(',')],'['+str(int(line[line.find('[')+1:line.find(',')])-1))
            if 'reshape' in line:
                line = line.replace('reshape(v, [], 1)','v.T') # transpose
            line = line.replace('v(i)','v[i]') # index calling
            line = line.replace('pi','np.pi') # pi value
            line = line.replace('zeros(n_pts_v, n_sites)','np.zeros((n_pts_v, n_sites))') # initializing zero matrix
            if 'for' in line:
                line = line.replace('for i  =  1:n_pts_v','for i in range(n_pts_v):') # for loop
                add_loop_flag = True
            if add_loop_flag and not 'for' in line:
                line = '    '+line
            line = line.replace('(i, :)','[i, :]') # calling index
            line = line.replace('exp(','np.exp(') # exponential function
        txt_list.append(line)
        if '\\' in line:
            line_prev = line
        else:
            line_prev = ''    
    txt_list.append('    return P_output') # add return parameter
	
# write to output file line by line
with open(out_file, 'w') as f:
    for row in txt_list:
        f.write(row + '\n')