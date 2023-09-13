import torch
import torch.nn as nn
from seq2seq.models.EncoderRNN import EncoderRNN
from seq2seq.models.DecoderRNN import DecoderRNN
from seq2seq.models.TopKDecoder import TopKDecoder


torch.cuda.set_device("cuda:0")


class Seq2Seq(nn.Module):
    def __init__(self, tokenizer, args):
        super(Seq2Seq, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        vocab_size = len(tokenizer)
        self.vocab_size = vocab_size
        self.encoder = EncoderRNN(vocab_size=vocab_size, max_len=512, hidden_size=512, n_layers=2, dropout_p=0.3, rnn_cell='lstm', bidirectional=True)
        self.decoder = DecoderRNN(vocab_size=vocab_size, max_len=256, hidden_size=512, sos_id=0, eos_id=2, n_layers=2, dropout_p=0.3, rnn_cell='lstm', use_attention=True)

    def beam_search(self, src, tgt):
        encoder_outputs, encoder_hidden = self.encoder(src)

        # summarize bidirectional output
        encoder_outputs = (encoder_outputs[:, :, :self.encoder.hidden_size] +
                           encoder_outputs[:, :, self.encoder.hidden_size:])
        # summarize bidirectional hidden & cell state
        new_enc_hid = []
        new_enc_hid.append(torch.cat((encoder_hidden[0][0].unsqueeze(0) + encoder_hidden[0][1].unsqueeze(0),
                           encoder_hidden[0][2].unsqueeze(0) + encoder_hidden[0][3].unsqueeze(0))))
        new_enc_hid.append(torch.cat((encoder_hidden[1][0].unsqueeze(0) + encoder_hidden[1][1].unsqueeze(0),
                           encoder_hidden[1][2].unsqueeze(0) + encoder_hidden[1][3].unsqueeze(0))))
        new_enc_hid = tuple(new_enc_hid)

        beam_search_decoder = TopKDecoder(decoder_rnn=self.decoder, k=self.args.beam_size)

        hid = encoder_hidden[0][0].unsqueeze(0) + encoder_hidden[0][1].unsqueeze(0)
        _, _, ret_dict = beam_search_decoder(inputs=src,
                                             encoder_hidden=new_enc_hid,
                                             encoder_outputs=encoder_outputs)
        beam_output = ret_dict["topk_sequence"]
        beam_output = torch.stack(beam_output).squeeze(-1).permute(1,2,0)
        return beam_output

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden = self.encoder(src)
        # summarize bidirectional output
        encoder_outputs = (encoder_outputs[:, :, :self.encoder.hidden_size] +
                           encoder_outputs[:, :, self.encoder.hidden_size:])
        # summarize bidirectional hidden & cell state
        new_enc_hid = []
        new_enc_hid.append(torch.cat((encoder_hidden[0][0].unsqueeze(0) + encoder_hidden[0][1].unsqueeze(0),
                           encoder_hidden[0][2].unsqueeze(0) + encoder_hidden[0][3].unsqueeze(0))))
        new_enc_hid.append(torch.cat((encoder_hidden[1][0].unsqueeze(0) + encoder_hidden[1][1].unsqueeze(0),
                           encoder_hidden[1][2].unsqueeze(0) + encoder_hidden[1][3].unsqueeze(0))))
        new_enc_hid = tuple(new_enc_hid)
        decoder_outputs, decoder_hidden, ret_dict = self.decoder(inputs=tgt,
                                                                 encoder_hidden=new_enc_hid,
                                                                 encoder_outputs=encoder_outputs,
                                                                 teacher_forcing_ratio=teacher_forcing_ratio)
        decoder_outputs = torch.stack(decoder_outputs, dim=0).permute(1, 0, 2)
        return decoder_outputs
