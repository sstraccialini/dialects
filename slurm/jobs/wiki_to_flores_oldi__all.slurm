#!/bin/bash
#SBATCH --job-name=ltp_wiki_to_flores_oldi_all
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Single SLURM job that runs Word2Vec and FastText sequentially on the
# same allocation, so we wait the queue only once.
# Total runtime expected: ~70-90 min on 8 CPU cores.

set -uo pipefail
source "$SLURM_SUBMIT_DIR/slurm/tools/env.sh"

echo ""
echo "=================================================="
echo " Step 1/2: Word2Vec wiki_to_flores_oldi"
echo "=================================================="
python -u analysis/word2vec/experiments/wiki_to_flores_oldi/run.py
W2V_RC=$?
echo "  word2vec exit code: $W2V_RC"

echo ""
echo "=================================================="
echo " Step 2/2: FastText wiki_to_flores_oldi"
echo "=================================================="
python -u analysis/fasttext/experiments/wiki_to_flores_oldi/run.py
FT_RC=$?
echo "  fasttext exit code: $FT_RC"

echo ""
echo "=================================================="
echo " Summary: word2vec=$W2V_RC  fasttext=$FT_RC"
echo "=================================================="

# fail the job if either step failed
[ $W2V_RC -eq 0 ] && [ $FT_RC -eq 0 ]
