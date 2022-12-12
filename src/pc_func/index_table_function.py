import numpy as np
from numba import njit, float64
from .pc_util import num_pc_terms


# def get_num_pc_terms(x, kl_dim, pc_order):
    # num_pc_terms_i = []
    # for i in range(pc_order):
        # num_pc_terms_i = [
    # return 

# def get_index_table_x(x, list_inc, kl_dim, table):
    # count = 0
    


def index_table_function(kl_dim, pc_order):
    """"""
    if pc_order > 10:
        raise ValueError('This pc_order is not available yet!!')
    
    else:
        ##########################################################################
        ######################## PC ORDER 0   ####################################
        ##########################################################################
        if pc_order == 0:
            table  =  np.zeros((1, kl_dim))
        
        ##########################################################################
        ######################## PC ORDER 1   ####################################
        ##########################################################################
        elif pc_order == 1:
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
                
            table = np.vstack([np.zeros((1,kl_dim)), index_table_1])
            
        ##########################################################################
        ######################## PC ORDER 2   ####################################
        ##########################################################################
        elif pc_order == 2:
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i]  =  index_table_1[count, i] + 1
                count += 1
                
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2])
            
        ##########################################################################
        ######################## PC ORDER 3   ####################################
        ##########################################################################    
        elif pc_order == 3:
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
            
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
            
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2, index_table_3])
        
        ##########################################################################
        ######################## PC ORDER 4   ####################################
        ##########################################################################
        elif pc_order == 4:
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            index_table_4[count, i] += 1
                            index_table_4[count, j] += 1
                            index_table_4[count, k] += 1
                            index_table_4[count, l] += 1
                            count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1

            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2, index_table_3, index_table_4])

        ##########################################################################
        ######################## PC ORDER 5   ####################################
        ##########################################################################
        elif pc_order == 5:
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
           
            table  =  np.vstack([
                np.zeros((1,kl_dim)), index_table_1, index_table_2, \
                index_table_3, index_table_4, index_table_5
            ])
        
        ##########################################################################
        ######################## PC ORDER 6   ####################################
        ##########################################################################
        elif pc_order == 6:
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
           
            table  =  np.vstack([
                np.zeros((1,kl_dim)), index_table_1, index_table_2, \
                index_table_3, index_table_4, index_table_5, index_table_6
            ])
           
        ##########################################################################
        ######################## PC ORDER 7   ####################################
        ##########################################################################
        elif pc_order == 7:
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])

            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7])
           
        ##########################################################################
        ######################## PC ORDER 8   ####################################
        ##########################################################################
        elif pc_order == 8:
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1
            
            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
             
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8])
             
        ##########################################################################
        ######################## PC ORDER 9   ####################################
        ##########################################################################
        elif pc_order == 9:
            num_pc_terms_9 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-8) - num_pc_terms(kl_dim,pc_order-9)
            index_table_9 = np.zeros([num_pc_terms_9, kl_dim])
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 9
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                index_table_9[count, i] += 1
                                                index_table_9[count, j] += 1
                                                index_table_9[count, k] += 1
                                                index_table_9[count, l] += 1
                                                index_table_9[count, m] += 1
                                                index_table_9[count, n] += 1
                                                index_table_9[count, o] += 1
                                                index_table_9[count, p] += 1
                                                index_table_9[count, q] += 1
                                                count += 1
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1
            
            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
             
            table  =  np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8, \
                index_table_9])
             
        ##########################################################################
        ######################## PC ORDER 10  ####################################
        ##########################################################################
        elif pc_order == 10:
            num_pc_terms_10 = num_pc_terms(kl_dim,pc_order) - num_pc_terms(kl_dim,pc_order-1)
            num_pc_terms_9 = num_pc_terms(kl_dim,pc_order-1) - num_pc_terms(kl_dim,pc_order-2)
            num_pc_terms_8 = num_pc_terms(kl_dim,pc_order-2) - num_pc_terms(kl_dim,pc_order-3)
            num_pc_terms_7 = num_pc_terms(kl_dim,pc_order-3) - num_pc_terms(kl_dim,pc_order-4)
            num_pc_terms_6 = num_pc_terms(kl_dim,pc_order-4) - num_pc_terms(kl_dim,pc_order-5)
            num_pc_terms_5 = num_pc_terms(kl_dim,pc_order-5) - num_pc_terms(kl_dim,pc_order-6)
            num_pc_terms_4 = num_pc_terms(kl_dim,pc_order-6) - num_pc_terms(kl_dim,pc_order-7)
            num_pc_terms_3 = num_pc_terms(kl_dim,pc_order-7) - num_pc_terms(kl_dim,pc_order-8)
            num_pc_terms_2 = num_pc_terms(kl_dim,pc_order-8) - num_pc_terms(kl_dim,pc_order-9)
            num_pc_terms_1 = num_pc_terms(kl_dim,pc_order-9) - num_pc_terms(kl_dim,pc_order-10)
            index_table_10 = np.zeros([num_pc_terms_10, kl_dim])
            index_table_9 = np.zeros([num_pc_terms_9, kl_dim])
            index_table_8 = np.zeros([num_pc_terms_8, kl_dim])
            index_table_7 = np.zeros([num_pc_terms_7, kl_dim])
            index_table_6 = np.zeros([num_pc_terms_6, kl_dim])
            index_table_5 = np.zeros([num_pc_terms_5, kl_dim])
            index_table_4 = np.zeros([num_pc_terms_4, kl_dim])
            index_table_3 = np.zeros([num_pc_terms_3, kl_dim])
            index_table_2 = np.zeros([num_pc_terms_2, kl_dim])
            index_table_1 = np.zeros([num_pc_terms_1, kl_dim])
            
            #Order 10
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                for r in range(q,kl_dim):
                                                    index_table_10[count, i] += 1
                                                    index_table_10[count, j] += 1
                                                    index_table_10[count, k] += 1
                                                    index_table_10[count, l] += 1
                                                    index_table_10[count, m] += 1
                                                    index_table_10[count, n] += 1
                                                    index_table_10[count, o] += 1
                                                    index_table_10[count, p] += 1
                                                    index_table_10[count, q] += 1
                                                    index_table_10[count, r] += 1
                                                    count += 1
                                                    
            #Order 9
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            for q in range(p,kl_dim):
                                                index_table_9[count, i] += 1
                                                index_table_9[count, j] += 1
                                                index_table_9[count, k] += 1
                                                index_table_9[count, l] += 1
                                                index_table_9[count, m] += 1
                                                index_table_9[count, n] += 1
                                                index_table_9[count, o] += 1
                                                index_table_9[count, p] += 1
                                                index_table_9[count, q] += 1
                                                count += 1
            
            #Order 8
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        for p in range(o,kl_dim):
                                            index_table_8[count, i] += 1
                                            index_table_8[count, j] += 1
                                            index_table_8[count, k] += 1
                                            index_table_8[count, l] += 1
                                            index_table_8[count, m] += 1
                                            index_table_8[count, n] += 1
                                            index_table_8[count, o] += 1
                                            index_table_8[count, p] += 1
                                            count += 1

            #Order 7
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                    for o in range(n,kl_dim):
                                        index_table_7[count, i] += 1
                                        index_table_7[count, j] += 1
                                        index_table_7[count, k] += 1
                                        index_table_7[count, l] += 1
                                        index_table_7[count, m] += 1
                                        index_table_7[count, n] += 1
                                        index_table_7[count, o] += 1
                                        count += 1
           
            #Order 6
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                for n in range(m,kl_dim):
                                   index_table_6[count, i] += 1
                                   index_table_6[count, j] += 1
                                   index_table_6[count, k] += 1
                                   index_table_6[count, l] += 1
                                   index_table_6[count, m] += 1
                                   index_table_6[count, n] += 1
                                   count += 1
           
            #Order 5
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                            for m in range(l,kl_dim):
                                index_table_5[count, i] += 1
                                index_table_5[count, j] += 1
                                index_table_5[count, k] += 1
                                index_table_5[count, l] += 1
                                index_table_5[count, m] += 1
                                count += 1
           
            #Order 4
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        for l in range(k,kl_dim):
                           index_table_4[count, i] += 1
                           index_table_4[count, j] += 1
                           index_table_4[count, k] += 1
                           index_table_4[count, l] += 1
                           count += 1
            
            #Order 3
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    for k in range(j,kl_dim):
                        index_table_3[count, i] += 1
                        index_table_3[count, j] += 1
                        index_table_3[count, k] += 1
                        count += 1
                        
            #Order 2
            count = 0
            for i in range(kl_dim):
                for j in range(i,kl_dim):
                    index_table_2[count, i] += 1
                    index_table_2[count, j] += 1
                    count += 1
                    
            #Order 1
            count = 0
            for i in range(kl_dim):
                index_table_1[count, i] += 1
                count += 1
            
            table = np.vstack([np.zeros((1,kl_dim)), index_table_1, index_table_2,\
                index_table_3, index_table_4, index_table_5,\
                index_table_6, index_table_7, index_table_8, \
                index_table_9, index_table_10])
        
        ##########################################################################
        ######################## END          ####################################
        ##########################################################################
        return table