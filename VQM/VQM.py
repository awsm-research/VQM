import torch.nn as nn


class VQM(nn.Module):
    def __init__(self, t5, tokenizer, args):
        super(VQM, self).__init__()
        self.t5 = t5
        self.tokenizer = tokenizer
        self.args = args

    def forward(self,
                input_ids,
                vul_query_mask=None,
                repair_input_ids=None,
                generate_repair=False):
        if generate_repair:
            beam_outputs = self.t5.generate(input_ids=input_ids,
                                          attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                                          do_sample=False, # disable sampling to test if batching affects output
                                          num_beams=self.args.num_beams,
                                          num_return_sequences=self.args.num_beams,
                                          max_length=self.args.vul_repair_block_size,
                                          vul_query_mask=vul_query_mask)
            return beam_outputs
        else:
            loss = self.t5(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), vul_query_mask=vul_query_mask, labels=repair_input_ids).loss
            return loss
