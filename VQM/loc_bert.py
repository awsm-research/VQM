import torch
import torch.nn as nn

        
class LocModel(nn.Module): 
    def __init__(self, encoder, config, tokenizer, args, num_labels):
        super(LocModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
    
    def forward(self, input_ids, vul_query_label=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        outputs = torch.amax(outputs, dim=-1)
        vul_query_prob = torch.sigmoid(outputs)
        if vul_query_label is not None:
            loc_loss_fct = nn.CrossEntropyLoss()
            vq_loss = loc_loss_fct(vul_query_prob, vul_query_label)
            return vq_loss
        else:
            vul_query_mask = self.mask_activation(vul_query_prob)
            vul_query_mask = vul_query_mask.unsqueeze(-1).expand(vul_query_mask.shape[0], vul_query_mask.shape[1], 768)
            return vul_query_mask

    def mask_activation(self, prob, beta=0.1, alpha=1000):
        x = beta / (1 + torch.exp(-alpha * (prob-0.5))).float()
        x = torch.where(x==torch.tensor(beta, device=self.args.device).float(), x, torch.tensor(0, device=self.args.device).float())
        return x