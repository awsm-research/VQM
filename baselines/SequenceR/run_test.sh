python sequencer_main.py \
    --beam_size 1 \
    --model_name=fine_tuned_sequencer.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam1_test.log

python sequencer_main.py \
    --beam_size 3 \
    --model_name=fine_tuned_sequencer.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam3_test.log

python sequencer_main.py \
    --beam_size 5 \
    --model_name=fine_tuned_sequencer.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee beam5_test.log