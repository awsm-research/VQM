python run.py \
    --num_beams 1 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --vul_repair_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam1test.log

python run.py \
    --num_beams 3 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --vul_repair_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam3test.log

python run.py \
    --num_beams 5 \
    --model_name=no_bug_model.bin \
    --output_dir=./saved_models \
    --tokenizer_name=Salesforce/codet5-base \
    --model_name_or_path=Salesforce/codet5-base \
    --do_test \
    --test_data_file=../../data/cve_fixes_and_big_vul/test.csv \
    --encoder_block_size 512 \
    --vul_repair_block_size 256 \
    --eval_batch_size 1 \
    --seed 123456  2>&1 | tee no_bug_beam5test.log