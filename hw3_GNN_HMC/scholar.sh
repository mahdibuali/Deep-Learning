#
module purge
module load learning/conda-5.1.0-py36-gpu
module load cudnn
module load cuda/10.1
source activate DPLClass


#
conda install python=3.8 ipython ipykernel -y
conda install pytorch=1.8.0 cudatoolkit=10.1 -c pytorch -y
conda install torchvision -c pytorch -y
conda install seaborn -y
pip install more_itertools --no-cache-dir
conda clean --all -y