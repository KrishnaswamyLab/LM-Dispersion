#!/bin/bash

#SBATCH --job-name=m-disp4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100|a100'
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00
#SBATCH --mem=80G
#SBATCH --mail-type=ALL

### 1 GPU: nodes=1, ntasks=1, gpus=1, cpus-per-task=10, mem=60G (if --num-workers 8)
### 2 GPU: nodes=1, ntasks=2, gpus=2, cpus-per-task=18, mem=80G (if --num-workers 8)

### For Misha
module purge
module load miniconda
module load CUDA/12.2.2 cuDNN/8.9.2.26-CUDA-12.2.2
module load GCC/12.2.0
module load git/2.38.1-GCCcore-12.2.0-nodocs

conda activate transformer

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || { echo "No GPUs visible (nvidia-smi failed)"; exit 1; }

cd /gpfs/radev/home/cl2482/project/Transformer-Dispersion/transformer_dispersion/midtrain_gpt2_huggingface
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --hf_token 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

# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 0.01 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 0.1 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 0.25 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 0.5 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 1.0 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Hinge' --dispersion_loc 'all' --dispersion_coeff 5.0 --hf_token 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

# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 0.01 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 0.1 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 0.25 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 0.5 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 1.0 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'Covariance' --dispersion_loc 'all' --dispersion_coeff 5.0 --hf_token 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

accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 0.01 --hf_token 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
accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 0.1 --hf_token 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
accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 0.25 --hf_token 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
accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 0.5 --hf_token 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
accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 1.0 --hf_token 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
accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_l2' --dispersion_loc 'all' --dispersion_coeff 5.0 --hf_token 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

# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 0.01 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 0.1 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 0.25 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 0.5 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 1.0 --hf_token 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
# accelerate launch midtrain_gpt2.py --train_tokens 300_000_000 --dispersion 'InfoNCE_cosine' --dispersion_loc 'all' --dispersion_coeff 5.0 --hf_token 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

