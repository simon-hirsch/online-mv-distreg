set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n reproduction python=3.10
conda activate reproduction
pip install -r req.txt
python experiments/epf_germany/00_run_study.py