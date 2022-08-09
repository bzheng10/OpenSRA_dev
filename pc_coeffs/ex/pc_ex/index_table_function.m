function table = index_table_function(KL_dim, PC_order)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 0   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PC_order == 0

    table = zeros(1,KL_dim);
    



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif PC_order == 1

      
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
  
    
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    
    table = vertcat(zeros(1,KL_dim), index_table_1);

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 2   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif PC_order == 2

    
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
     
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
 
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 3   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


elseif PC_order == 3

    
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3); 
    
    
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
   
    

    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2, index_table_3);

    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 4   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  
  
  
elseif PC_order == 4



    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    
    
    
    
    
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2, index_table_3, index_table_4);
    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 6   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif PC_order == 6



    Num_PCterms_6 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_5 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-4) - num_pc_terms(KL_dim,PC_order-5);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-5) - num_pc_terms(KL_dim,PC_order-6);
    
    
    
    
    
    index_table_6 = zeros(Num_PCterms_6, KL_dim);
    
    index_table_5 = zeros(Num_PCterms_5, KL_dim);
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    
    
    
    
    
    %Order 6
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            index_table_6(count, i) = index_table_6(count, i) + 1;
                            index_table_6(count, j) = index_table_6(count, j) + 1;
                            index_table_6(count, k) = index_table_6(count, k) + 1;
                            index_table_6(count, l) = index_table_6(count, l) + 1;
                            index_table_6(count, m) = index_table_6(count, m) + 1;
                            index_table_6(count, n) = index_table_6(count, n) + 1;
                            
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    
    
    
    %Order 5
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        index_table_5(count, i) = index_table_5(count, i) + 1;
                        index_table_5(count, j) = index_table_5(count, j) + 1;
                        index_table_5(count, k) = index_table_5(count, k) + 1;
                        index_table_5(count, l) = index_table_5(count, l) + 1;
                        index_table_5(count, m) = index_table_5(count, m) + 1;
                        
                        count = count + 1;
                        
                    end
                end
            end
        end
    end
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2, index_table_3, index_table_4, index_table_5, index_table_6);


    
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 7   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif PC_order == 7



    Num_PCterms_7 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_6 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_5 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-4) - num_pc_terms(KL_dim,PC_order-5);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-5) - num_pc_terms(KL_dim,PC_order-6);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-6) - num_pc_terms(KL_dim,PC_order-7);

    
    
    index_table_7 = zeros(Num_PCterms_7, KL_dim);
    
    index_table_6 = zeros(Num_PCterms_6, KL_dim);
    
    index_table_5 = zeros(Num_PCterms_5, KL_dim);
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    
    
    %Order 7
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                            
                                index_table_7(count, i) = index_table_7(count, i) + 1;
                                index_table_7(count, j) = index_table_7(count, j) + 1;
                                index_table_7(count, k) = index_table_7(count, k) + 1;
                                index_table_7(count, l) = index_table_7(count, l) + 1;
                                index_table_7(count, m) = index_table_7(count, m) + 1;
                                index_table_7(count, n) = index_table_7(count, n) + 1;
                                index_table_7(count, o) = index_table_7(count, o) + 1;

                                                     
                            count = count + 1;
                            
                            end
                        end
                    end
                end
            end
        end
    end
    
    
    
   
    
    
    
    
    %Order 6
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            index_table_6(count, i) = index_table_6(count, i) + 1;
                            index_table_6(count, j) = index_table_6(count, j) + 1;
                            index_table_6(count, k) = index_table_6(count, k) + 1;
                            index_table_6(count, l) = index_table_6(count, l) + 1;
                            index_table_6(count, m) = index_table_6(count, m) + 1;
                            index_table_6(count, n) = index_table_6(count, n) + 1;
                            
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    
    
    
    %Order 5
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        index_table_5(count, i) = index_table_5(count, i) + 1;
                        index_table_5(count, j) = index_table_5(count, j) + 1;
                        index_table_5(count, k) = index_table_5(count, k) + 1;
                        index_table_5(count, l) = index_table_5(count, l) + 1;
                        index_table_5(count, m) = index_table_5(count, m) + 1;
                        
                        count = count + 1;
                        
                    end
                end
            end
        end
    end
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2,...
                    index_table_3, index_table_4, index_table_5,...
                    index_table_6, index_table_7);


                
                
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 8   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif PC_order == 8



    Num_PCterms_8 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_7 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_6 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_5 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order-4) - num_pc_terms(KL_dim,PC_order-5);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-5) - num_pc_terms(KL_dim,PC_order-6);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-6) - num_pc_terms(KL_dim,PC_order-7);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-7) - num_pc_terms(KL_dim,PC_order-8);
    

    
    
    index_table_8 = zeros(Num_PCterms_8, KL_dim);   
    
    index_table_7 = zeros(Num_PCterms_7, KL_dim);
    
    index_table_6 = zeros(Num_PCterms_6, KL_dim);
    
    index_table_5 = zeros(Num_PCterms_5, KL_dim);
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    
    
    %Order 8
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                            
                                    index_table_8(count, i) = index_table_8(count, i) + 1;
                                    index_table_8(count, j) = index_table_8(count, j) + 1;
                                    index_table_8(count, k) = index_table_8(count, k) + 1;
                                    index_table_8(count, l) = index_table_8(count, l) + 1;
                                    index_table_8(count, m) = index_table_8(count, m) + 1;
                                    index_table_8(count, n) = index_table_8(count, n) + 1;
                                    index_table_8(count, o) = index_table_8(count, o) + 1;
                                    index_table_8(count, p) = index_table_8(count, p) + 1;

                                                     
                                    count = count + 1;
                            
                                end
                            end
                        end
                    end
                end
            end
        end
    end 
    
    
    
    
    
    %Order 7
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                            
                            index_table_7(count, i) = index_table_7(count, i) + 1;
                            index_table_7(count, j) = index_table_7(count, j) + 1;
                            index_table_7(count, k) = index_table_7(count, k) + 1;
                            index_table_7(count, l) = index_table_7(count, l) + 1;
                            index_table_7(count, m) = index_table_7(count, m) + 1;
                            index_table_7(count, n) = index_table_7(count, n) + 1;
                            index_table_7(count, o) = index_table_7(count, o) + 1;

                                                     
                            count = count + 1;
                            
                            end
                        end
                    end
                end
            end
        end
    end
    
    
    
   
    
    
    
    
    %Order 6
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            index_table_6(count, i) = index_table_6(count, i) + 1;
                            index_table_6(count, j) = index_table_6(count, j) + 1;
                            index_table_6(count, k) = index_table_6(count, k) + 1;
                            index_table_6(count, l) = index_table_6(count, l) + 1;
                            index_table_6(count, m) = index_table_6(count, m) + 1;
                            index_table_6(count, n) = index_table_6(count, n) + 1;
                            
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    
    
    
    %Order 5
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        index_table_5(count, i) = index_table_5(count, i) + 1;
                        index_table_5(count, j) = index_table_5(count, j) + 1;
                        index_table_5(count, k) = index_table_5(count, k) + 1;
                        index_table_5(count, l) = index_table_5(count, l) + 1;
                        index_table_5(count, m) = index_table_5(count, m) + 1;
                        
                        count = count + 1;
                        
                    end
                end
            end
        end
    end
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2,...
                    index_table_3, index_table_4, index_table_5,...
                    index_table_6, index_table_7, index_table_8);
                
                
                
       
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 9   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif PC_order == 9



    Num_PCterms_9 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_8 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_7 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_6 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    Num_PCterms_5 = num_pc_terms(KL_dim,PC_order-4) - num_pc_terms(KL_dim,PC_order-5);
    
    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order-5) - num_pc_terms(KL_dim,PC_order-6);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-6) - num_pc_terms(KL_dim,PC_order-7);
    
    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-7) - num_pc_terms(KL_dim,PC_order-8);

    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-8) - num_pc_terms(KL_dim,PC_order-9);
    

    
    
    index_table_9 = zeros(Num_PCterms_9, KL_dim);   
    
    index_table_8 = zeros(Num_PCterms_8, KL_dim);   
    
    index_table_7 = zeros(Num_PCterms_7, KL_dim);
    
    index_table_6 = zeros(Num_PCterms_6, KL_dim);
    
    index_table_5 = zeros(Num_PCterms_5, KL_dim);
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    %Order 9
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                                    
                                    for q = p:KL_dim
                            
                                        index_table_9(count, i) = index_table_9(count, i) + 1;
                                        index_table_9(count, j) = index_table_9(count, j) + 1;
                                        index_table_9(count, k) = index_table_9(count, k) + 1;
                                        index_table_9(count, l) = index_table_9(count, l) + 1;
                                        index_table_9(count, m) = index_table_9(count, m) + 1;
                                        index_table_9(count, n) = index_table_9(count, n) + 1;
                                        index_table_9(count, o) = index_table_9(count, o) + 1;
                                        index_table_9(count, p) = index_table_9(count, p) + 1;
                                        index_table_9(count, q) = index_table_9(count, q) + 1;
                                        
                                        
                                        count = count + 1;
                            
                                    end
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end    
    
    
    
    
    
    
    
    %Order 8
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                            
                                    index_table_8(count, i) = index_table_8(count, i) + 1;
                                    index_table_8(count, j) = index_table_8(count, j) + 1;
                                    index_table_8(count, k) = index_table_8(count, k) + 1;
                                    index_table_8(count, l) = index_table_8(count, l) + 1;
                                    index_table_8(count, m) = index_table_8(count, m) + 1;
                                    index_table_8(count, n) = index_table_8(count, n) + 1;
                                    index_table_8(count, o) = index_table_8(count, o) + 1;
                                    index_table_8(count, p) = index_table_8(count, p) + 1;

                                                     
                                    count = count + 1;
                            
                                end
                            end
                        end
                    end
                end
            end
        end
    end 
    
    
    
    
    
    %Order 7
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                            
                            index_table_7(count, i) = index_table_7(count, i) + 1;
                            index_table_7(count, j) = index_table_7(count, j) + 1;
                            index_table_7(count, k) = index_table_7(count, k) + 1;
                            index_table_7(count, l) = index_table_7(count, l) + 1;
                            index_table_7(count, m) = index_table_7(count, m) + 1;
                            index_table_7(count, n) = index_table_7(count, n) + 1;
                            index_table_7(count, o) = index_table_7(count, o) + 1;

                                                     
                            count = count + 1;
                            
                            end
                        end
                    end
                end
            end
        end
    end
    
    
    
   
    
    
    
    
    %Order 6
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            index_table_6(count, i) = index_table_6(count, i) + 1;
                            index_table_6(count, j) = index_table_6(count, j) + 1;
                            index_table_6(count, k) = index_table_6(count, k) + 1;
                            index_table_6(count, l) = index_table_6(count, l) + 1;
                            index_table_6(count, m) = index_table_6(count, m) + 1;
                            index_table_6(count, n) = index_table_6(count, n) + 1;
                            
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    
    
    
    %Order 5
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        index_table_5(count, i) = index_table_5(count, i) + 1;
                        index_table_5(count, j) = index_table_5(count, j) + 1;
                        index_table_5(count, k) = index_table_5(count, k) + 1;
                        index_table_5(count, l) = index_table_5(count, l) + 1;
                        index_table_5(count, m) = index_table_5(count, m) + 1;
                        
                        count = count + 1;
                        
                    end
                end
            end
        end
    end
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2,...
                    index_table_3, index_table_4, index_table_5,...
                    index_table_6, index_table_7, index_table_8, ...
                    index_table_9);              
                
                

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% PC ORDER 10   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif PC_order == 10



    Num_PCterms_10 = num_pc_terms(KL_dim,PC_order) - num_pc_terms(KL_dim,PC_order-1);
    
    Num_PCterms_9 = num_pc_terms(KL_dim,PC_order-1) - num_pc_terms(KL_dim,PC_order-2);
    
    Num_PCterms_8 = num_pc_terms(KL_dim,PC_order-2) - num_pc_terms(KL_dim,PC_order-3);
    
    Num_PCterms_7 = num_pc_terms(KL_dim,PC_order-3) - num_pc_terms(KL_dim,PC_order-4);
    
    Num_PCterms_6 = num_pc_terms(KL_dim,PC_order-4) - num_pc_terms(KL_dim,PC_order-5);
    
    Num_PCterms_5 = num_pc_terms(KL_dim,PC_order-5) - num_pc_terms(KL_dim,PC_order-6);
    
    Num_PCterms_4 = num_pc_terms(KL_dim,PC_order-6) - num_pc_terms(KL_dim,PC_order-7);
    
    Num_PCterms_3 = num_pc_terms(KL_dim,PC_order-7) - num_pc_terms(KL_dim,PC_order-8);

    Num_PCterms_2 = num_pc_terms(KL_dim,PC_order-8) - num_pc_terms(KL_dim,PC_order-9);
    
    Num_PCterms_1 = num_pc_terms(KL_dim,PC_order-9) - num_pc_terms(KL_dim,PC_order-10);


    
    index_table_10 = zeros(Num_PCterms_10, KL_dim);   
    
    index_table_9 = zeros(Num_PCterms_9, KL_dim);   
    
    index_table_8 = zeros(Num_PCterms_8, KL_dim);   
    
    index_table_7 = zeros(Num_PCterms_7, KL_dim);
    
    index_table_6 = zeros(Num_PCterms_6, KL_dim);
    
    index_table_5 = zeros(Num_PCterms_5, KL_dim);
    
    index_table_4 = zeros(Num_PCterms_4, KL_dim);
    
    index_table_3 = zeros(Num_PCterms_3, KL_dim);
    
    index_table_2 = zeros(Num_PCterms_2, KL_dim);
    
    index_table_1 = zeros(Num_PCterms_1, KL_dim);
    
    
    
    %Order 10
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                                    
                                    for q = p:KL_dim
                            
                                        for r = q:KL_dim
                                        
                                        index_table_10(count, i) = index_table_10(count, i) + 1;
                                        index_table_10(count, j) = index_table_10(count, j) + 1;
                                        index_table_10(count, k) = index_table_10(count, k) + 1;
                                        index_table_10(count, l) = index_table_10(count, l) + 1;
                                        index_table_10(count, m) = index_table_10(count, m) + 1;
                                        index_table_10(count, n) = index_table_10(count, n) + 1;
                                        index_table_10(count, o) = index_table_10(count, o) + 1;
                                        index_table_10(count, p) = index_table_10(count, p) + 1;
                                        index_table_10(count, q) = index_table_10(count, q) + 1;
                                        index_table_10(count, r) = index_table_10(count, r) + 1;
                         
                                        count = count + 1;
                            
                                        end
                                    end
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end    
    
    
    
    
    
    
    
    %Order 9
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                                    
                                    for q = p:KL_dim
                            
                                        index_table_9(count, i) = index_table_9(count, i) + 1;
                                        index_table_9(count, j) = index_table_9(count, j) + 1;
                                        index_table_9(count, k) = index_table_9(count, k) + 1;
                                        index_table_9(count, l) = index_table_9(count, l) + 1;
                                        index_table_9(count, m) = index_table_9(count, m) + 1;
                                        index_table_9(count, n) = index_table_9(count, n) + 1;
                                        index_table_9(count, o) = index_table_9(count, o) + 1;
                                        index_table_9(count, p) = index_table_9(count, p) + 1;
                                        index_table_9(count, q) = index_table_9(count, q) + 1;
                                        
                                        
                                        count = count + 1;
                            
                                    end
                                    
                                end
                            end
                        end
                    end
                end
            end
        end
    end    
    
    
    
    
    
    
    
    %Order 8
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                                
                                for p = o:KL_dim
                            
                                    index_table_8(count, i) = index_table_8(count, i) + 1;
                                    index_table_8(count, j) = index_table_8(count, j) + 1;
                                    index_table_8(count, k) = index_table_8(count, k) + 1;
                                    index_table_8(count, l) = index_table_8(count, l) + 1;
                                    index_table_8(count, m) = index_table_8(count, m) + 1;
                                    index_table_8(count, n) = index_table_8(count, n) + 1;
                                    index_table_8(count, o) = index_table_8(count, o) + 1;
                                    index_table_8(count, p) = index_table_8(count, p) + 1;

                                                     
                                    count = count + 1;
                            
                                end
                            end
                        end
                    end
                end
            end
        end
    end 
    
    
    
    
    
    %Order 7
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            for o = n:KL_dim
                            
                            index_table_7(count, i) = index_table_7(count, i) + 1;
                            index_table_7(count, j) = index_table_7(count, j) + 1;
                            index_table_7(count, k) = index_table_7(count, k) + 1;
                            index_table_7(count, l) = index_table_7(count, l) + 1;
                            index_table_7(count, m) = index_table_7(count, m) + 1;
                            index_table_7(count, n) = index_table_7(count, n) + 1;
                            index_table_7(count, o) = index_table_7(count, o) + 1;

                                                     
                            count = count + 1;
                            
                            end
                        end
                    end
                end
            end
        end
    end
    
    
    
   
    
    
    
    
    %Order 6
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        for n = m:KL_dim
                            
                            index_table_6(count, i) = index_table_6(count, i) + 1;
                            index_table_6(count, j) = index_table_6(count, j) + 1;
                            index_table_6(count, k) = index_table_6(count, k) + 1;
                            index_table_6(count, l) = index_table_6(count, l) + 1;
                            index_table_6(count, m) = index_table_6(count, m) + 1;
                            index_table_6(count, n) = index_table_6(count, n) + 1;
                            
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
    
    
    
    
    
    
    
    
    
    %Order 5
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    for m = l:KL_dim
                        
                        index_table_5(count, i) = index_table_5(count, i) + 1;
                        index_table_5(count, j) = index_table_5(count, j) + 1;
                        index_table_5(count, k) = index_table_5(count, k) + 1;
                        index_table_5(count, l) = index_table_5(count, l) + 1;
                        index_table_5(count, m) = index_table_5(count, m) + 1;
                        
                        count = count + 1;
                        
                    end
                end
            end
        end
    end
    
    
    
    
    %Order 4
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                for l = k:KL_dim
                    
                    index_table_4(count, i) = index_table_4(count, i) + 1;
                    index_table_4(count, j) = index_table_4(count, j) + 1;
                    index_table_4(count, k) = index_table_4(count, k) + 1;
                    index_table_4(count, l) = index_table_4(count, l) + 1;
                    count = count + 1;
                    
                end
            end
        end
    end
    
    
    
    
    
    
    
    %Order 3
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            for k = j:KL_dim
                
                index_table_3(count, i) = index_table_3(count, i) + 1;
                index_table_3(count, j) = index_table_3(count, j) + 1;
                index_table_3(count, k) = index_table_3(count, k) + 1;
                
                count = count + 1;
                
            end
        end
    end
    
    %Order 2
    
    count = 1;
    
    for i = 1:KL_dim
        
        for j = i:KL_dim
            
            index_table_2(count, i) = index_table_2(count, i) + 1;
            index_table_2(count, j) = index_table_2(count, j) + 1;
            
            count = count +1;
        end
        
    end
    
    
    %Order 1
    
    count = 1;
    
    for i = 1:KL_dim
        
        index_table_1(count, i) = index_table_1(count, i) + 1;
        
        count = count + 1;
        
    end
    
    table = vertcat(zeros(1,KL_dim), index_table_1, index_table_2,...
                    index_table_3, index_table_4, index_table_5,...
                    index_table_6, index_table_7, index_table_8, ...
                    index_table_9, index_table_10);              
                
                              
                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%%%%%%%%%%%%%%%%%%%%%%%%%     END     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       



else 
    
    print('This PC_order is not available yet!!')


        
    
    
    
end


