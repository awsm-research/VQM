python run.py \
--beam_size 1 \
--model_name no_bug_fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/cve_fixes_and_big_vul/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee no_bug_beam1_test.log

python run.py \
--beam_size 3 \
--model_name no_bug_fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/cve_fixes_and_big_vul/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee no_bug_beam3_test.log

python run.py \
--beam_size 5 \
--model_name no_bug_fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/cve_fixes_and_big_vul/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee no_bug_beam5_test.log