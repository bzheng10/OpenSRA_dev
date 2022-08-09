function Num_pc_terms = num_pc_terms(KL_dim, PC_order)

Num_pc_terms = nchoosek(KL_dim + PC_order, PC_order);

end

