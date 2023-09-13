python sequencer_main.py \
    --beam_size 1 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam1_test.log

python sequencer_main.py \
    --beam_size 3 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam3_test.log

python sequencer_main.py \
    --beam_size 5 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=./tokenizer \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --decoder_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam5_test.log