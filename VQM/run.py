from __future__ import absolute_import, division, print_function
import argparse
from ast import And
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from tqdm import tqdm
import pandas as pd
from VQM import VQM
from loc_bert import LocModel

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 vul_query_label,
                 repair_input_ids):
        self.input_ids = input_ids
        self.vul_query_label = vul_query_label
        self.repair_input_ids = repair_input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        source = df["source"].tolist()
        repair_target = df["target"].tolist()
        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(source[i], repair_target[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("vul_query_label: {}".format(' '.join(map(str, example.vul_query_label))))
                    logger.info("repair_input_ids: {}".format(' '.join(map(str, example.repair_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].vul_query_label), torch.tensor(self.examples[i].repair_input_ids)

def convert_examples_to_features(source, repair_target, tokenizer, args):
    # encode - subword tokenize
    input_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length')
    repair_input_ids = tokenizer.encode(repair_target, truncation=True, max_length=args.vul_repair_block_size, padding='max_length')

    vul_query = []
    is_vul = False
    for n in range(512):
        if input_ids[n] == tokenizer.start_bug_id:
            is_vul = True
            vul_query.append(1)
        elif input_ids[n] == tokenizer.end_bug_id:
            is_vul = False
            vul_query.append(1)
        elif is_vul:
            vul_query.append(1)
        else:
            vul_query.append(0)
    return InputFeatures(input_ids, vul_query, repair_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, loc_model, eval_dataset, train_with_mask):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader)*args.epochs*0.1, num_training_steps=len(train_dataloader)*args.epochs)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100000

    model.zero_grad()
    early_stop = 0
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            model.train()
            (input_ids, _, repair_input_ids) = [x.to(args.device) for x in batch]
            if train_with_mask:
                vul_query_mask = loc_model(input_ids)
                loss = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)
            else:
                loss = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, loc_model, eval_dataset, eval_when_training=True, eval_with_mask=train_with_mask)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        early_stop += 1
                        if not train_with_mask and early_stop >= 5:
                            print("Early stopping for warm-up training without mask.")
                            break

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, loc_model, eval_dataset, eval_when_training=False, eval_with_mask=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, _, repair_input_ids) = [x.to(args.device) for x in batch]
        if eval_with_mask:
            vul_query_mask = loc_model(input_ids)
            loss = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)        
        else:
            loss = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, loc_model, tokenizer, test_dataset):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, vq, repair_input_ids)=[x.to(args.device) for x in batch]
        vul_query_mask = loc_model(input_ids)
        with torch.no_grad():
            beam_outputs = model(input_ids=input_ids, repair_input_ids=repair_input_ids, vul_query_mask=vul_query_mask, generate_repair=True)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        repair_input_ids = repair_input_ids.detach().cpu().tolist()
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            # truth
            ground_truth = tokenizer.decode(repair_input_ids[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        nb_eval_steps += 1
        t = str(round(sum(accuracy) / len(accuracy), 4))
        bar.set_description(f"test acc: {t}")
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    parser.add_argument("--vul_repair_block_size", default=256, type=int,
                        help="")

    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_pretrained_t5", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--pretrained_model_name", default="pretrained_model.bin", type=str,
                        help="")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--warmup", action='store_true',
                        help="")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()
    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = "cuda:0"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
    set_seed(args)


    tok = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tok.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    encoder.resize_token_embeddings(len(tok))
    loc_model = LocModel(encoder=encoder, config=encoder.config, tokenizer=tok, args=args, num_labels=2)
    loc_model.load_state_dict(torch.load("./saved_models/checkpoint-best-loss/loc_fine_tuned_model.bin", map_location=args.device))
    loc_model.to(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    start_bug_id = tokenizer.encode("<S2SV_StartBug>",add_special_tokens=False)[0]
    end_bug_id = tokenizer.encode("<S2SV_EndBug>",add_special_tokens=False)[0]
    tokenizer.start_bug_id = start_bug_id
    tokenizer.end_bug_id = end_bug_id

    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    t5.resize_token_embeddings(len(tokenizer))
    t5.config.use_encoder_vul_mask = True
    t5.config.use_decoder_vul_mask = True

    if args.load_pretrained_t5:
        t5.load_state_dict(torch.load(f"saved_models/checkpoint-best-loss/{args.pretrained_model_name}", map_location=args.device))
    
    model = VQM(t5=t5, tokenizer=tokenizer, args=args)

    if args.load_pretrained_model:
        checkpoint_prefix = f'checkpoint-best-loss/{args.pretrained_model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        if args.warmup:
            train(args, train_dataset, model, loc_model, eval_dataset, train_with_mask=False)
        train(args, train_dataset, model, loc_model, eval_dataset, train_with_mask=True)

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, loc_model, tokenizer, test_dataset)

if __name__ == "__main__":
    main()