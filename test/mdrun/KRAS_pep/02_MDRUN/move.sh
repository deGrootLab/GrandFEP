src="/home/chui/E29Project-2023-04-11/139-Peptide/2025-10-30-openff/03-KRAS/02_edges/edge_22_to_edge_INT_22_23_1/08-GrandFEP/peptide/OPT_sqrH_50_win12/opt_2"
# copy npt_eq.rst7
for win in 0 1 2 ; do
    mkdir -p $win
    cp $src/$win/npt_eq.rst7 $win/
done

for win in 5 6 ; do
    mkdir -p $(($win - 2))
    cp $src/$win/npt_eq.rst7 $(($win - 2))/
done

for win in 9 10 11 ; do
    mkdir -p $(($win - 4))
    cp $src/$win/npt_eq.rst7 $(($win - 4))/
done