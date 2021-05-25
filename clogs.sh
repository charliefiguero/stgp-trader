rm Test00balances.csv
rm Test00blotters.csv
rm Test00prices.csv
rm Test00tapes.csv

find ./stgp_csvs/gen_records/ ! -name '.gitkeep' -type f -exec rm -f {} +
find ./stgp_csvs/hall_of_fame/ ! -name '.gitkeep' -type f -exec rm -f {} +
find ./stgp_csvs/improvements/ ! -name '.gitkeep' -type f -exec rm -f {} +