export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS="ignore::DeprecationWarning"

python run_summarization.py \
--mode=train \
--data_path=./data/$1 \
--vocab_path=./data/$2 \
--log_root=logroot \
--exp_name=$3 \
--min_dec_steps=15 \
--max_dec_steps=80 \
--max_enc_steps=400 \
--num_sections=20 \
--max_section_len=50 \
--min_section_len=1 \
--min_abstract_len=1 \
--batch_size=1 \
--vocab_size=50000 \
--use_do=True \
--optimizer=adagrad \
--do_prob=0.25 \
--hier=True \
--split_intro=False \
--fixed_attn=True \
--legacy_encoder=False \
--coverage=False
