#
set -e


#
python main_hmc.py --sbatch scholar --device cpu


#
python main_ddi.py --sbatch gpu --device cuda
python main_ddi.py --sbatch gpu --device cuda --positional


#
python main_ptb.py --sbatch scholar --device cpu --model markov --order 1
python main_ptb.py --sbatch scholar --device cpu --model markov --order 2
python main_ptb.py --sbatch scholar --device cpu --model markov --order 10
python main_ptb.py --sbatch gpu --device cuda --model lstm --trunc 5
python main_ptb.py --sbatch gpu --device cuda --model lstm --trunc 35
python main_ptb.py --sbatch gpu --device cuda --model lstm --trunc 80


#
sbatch sbatch/hmc.sh
sbatch sbatch/ddi_structure.sh
sbatch sbatch/ddi_position.sh
sbatch sbatch/ptb_markov~1.sh
sbatch sbatch/ptb_markov~2.sh
sbatch sbatch/ptb_markov~10.sh
sbatch sbatch/ptb_lstm~5.sh
sbatch sbatch/ptb_lstm~35.sh
sbatch sbatch/ptb_lstm~80.sh
