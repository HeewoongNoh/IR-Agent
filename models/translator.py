import torch
import math
import numpy as np
import torch.nn.functional as F


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, len_seq):
        super(EmbeddingLayer, self).__init__()
        self.len_seq = len_seq
        self.fc = torch.nn.Linear(dim_in, dim_emb)
        self.pos_emb_layer = torch.nn.Embedding(len_seq, dim_emb) 
        self.ln = torch.nn.LayerNorm(dim_emb, eps=1e-6)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        h = self.fc(x)
        pos_emb = self.pos_emb_layer(torch.arange(0, self.len_seq).unsqueeze(0).cuda())
        out = h + pos_emb
        out = self.dropout(self.ln(out))
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_emb, dim_head, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dim_model = self.dim_head * self.num_heads
        self.fc_q = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_k = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_v = torch.nn.Linear(dim_emb, self.dim_model)
        self.fc_out = torch.nn.Linear(self.dim_model, dim_emb)

    def forward(self, q, k, v, mask=None):
        # x: (num_batch, len_seq, dim_emb)
        # q, k, v: (num_batch, num_heads, len_seq, dim_head)
        # attn: (num_batch, num_heads, len_seq, len_seq)
        # out: (num_batch, len_seq, dim_emb)

        # Transform the input query, key, and value.
        q = self.fc_q(q).view(q.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.fc_k(k).view(k.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.fc_v(v).view(v.shape[0], -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Calculate self-attentions.
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=3)

        # Generate new latent embeddings of the value based on the calculated attentions.
        h = torch.matmul(attn, v).transpose(1, 2).contiguous().view(q.shape[0], -1, self.dim_model)
        out = self.fc_out(h)

        return out


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dim_emb):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = torch.nn.Linear(dim_emb, 64)
        self.fc2 = torch.nn.Linear(64, dim_emb)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        out = self.dropout(h)

        return out


class EncoderBlock(torch.nn.Module):
    def __init__(self, dim_emb, dim_head, num_heads):
        super(EncoderBlock, self).__init__()
        self.attn_layer = MultiHeadAttention(dim_emb, dim_head, num_heads)
        self.ff_layer = FeedForwardLayer(dim_emb)
        self.layer_norm1 = torch.nn.LayerNorm(dim_emb)
        self.layer_norm2 = torch.nn.LayerNorm(dim_emb)

    def forward(self, x):
        out = self.layer_norm1(x + self.attn_layer(x, x, x))
        out = self.layer_norm2(out + self.ff_layer(out))

        return out


class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, len_spect, num_layers, dim_head, num_heads):
        super(Encoder, self).__init__()
        self.emb_layer = EmbeddingLayer(dim_in, dim_emb, len_spect)
        self.layers = torch.nn.ModuleList([EncoderBlock(dim_emb, dim_head, num_heads) for _ in range(0, num_layers)])

    def forward(self, x):
        z = self.emb_layer(x)
        for layer in self.layers:
            z = layer(z)

        return z


class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_emb, dim_head, num_heads):
        super(DecoderBlock, self).__init__()
        self.masked_attn_layer = MultiHeadAttention(dim_emb, dim_head, num_heads)
        self.attn_layer = MultiHeadAttention(dim_emb, dim_head, num_heads)
        self.ff_layer = FeedForwardLayer(dim_emb)
        self.layer_norm1 = torch.nn.LayerNorm(dim_emb)
        self.layer_norm2 = torch.nn.LayerNorm(dim_emb)
        self.layer_norm3 = torch.nn.LayerNorm(dim_emb)

    def forward(self, x, context):
        mask = np.tril(np.ones((x.shape[1], x.shape[1])))
        mask = torch.tensor(mask, dtype=torch.bool, requires_grad=False).cuda()

        h = self.layer_norm1(x + self.masked_attn_layer(x, x, x, mask))
        h = self.layer_norm2(h + self.attn_layer(h, context, context))
        out = self.layer_norm3(h + self.ff_layer(h))

        return out


class Decoder(torch.nn.Module):
    def __init__(self, num_syms, dim_emb, max_len, num_layers, dim_head, num_heads):
        super(Decoder, self).__init__()
        self.max_len = max_len
        self.sym_emb_layer = torch.nn.Embedding(num_syms, dim_emb)
        self.pos_emb_layer = torch.nn.Embedding(max_len, dim_emb)
        self.ln = torch.nn.LayerNorm(dim_emb, eps=1e-6)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layers = torch.nn.ModuleList([DecoderBlock(dim_emb, dim_head, num_heads) for _ in range(0, num_layers)])
        self.fc = torch.nn.Linear(dim_emb, num_syms)

    # def forward(self, tgt, context):
    #     sym_emb = self.sym_emb_layer(tgt)
    #     pos_emb = self.pos_emb_layer(torch.arange(0, self.max_len).unsqueeze(0).cuda())
    #     z = sym_emb + pos_emb
    #     z = self.dropout(self.ln(z))

    #     for layer in self.layers:
    #         z = layer(z, context)
    #     out = self.fc(z)

    #     return out
    # For Decoding
    def forward(self, tgt, context):
        seq_len = tgt.size(1)
        sym_emb = self.sym_emb_layer(tgt)
        pos_emb = self.pos_emb_layer(torch.arange(0, seq_len).unsqueeze(0).cuda())
        z = sym_emb + pos_emb
        z = self.dropout(self.ln(z))
        
        for layer in self.layers:
            z = layer(z, context)
        out = self.fc(z)
        
        return out

class TransformerModel(torch.nn.Module):
    def __init__(self, dataset_config, dim_emb, num_enc_layers, num_dec_layers, dim_head, num_heads):
        super(TransformerModel, self).__init__()
        self.encoder = Encoder(dataset_config['dim_spect'],
                               dim_emb,
                               dataset_config['len_spect'],
                               num_enc_layers,
                               dim_head,
                               num_heads)
        self.decoder = Decoder(dataset_config['num_syms'],
                               dim_emb,
                               dataset_config['len_smiles'],
                               num_dec_layers,
                               dim_head,
                               num_heads)

    def forward(self, x, tgt):
        context = self.encoder(x)
        out = self.decoder(tgt, context)

        return out
    
    def _train(self, data_loader, optimizer, loss_func, scheduler):
        sum_losses = 0
        self.train()
        for batch in data_loader:
            src = batch[0].cuda()
            tgt = batch[1].cuda()

            preds = self(src, tgt)[:, :-1, :]
            loss = loss_func(preds.reshape(-1, preds.shape[2]), tgt[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    @staticmethod
    def idx_to_smiles(sym_idx, sym_set):
        smiles = ''

        for i in range(0, sym_idx.shape[0]):
            if sym_idx[i].item() == 1:
                continue
            elif sym_idx[i].item() == 2:
                break
            else:
                smiles += sym_set[sym_idx[i].item()]

        return smiles
    
def inference(model, data_loader, syms, sym_set, max_len):
    list_targets = list()
    list_preds = list()
    token_list = list()

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            src = batch[0].cuda()
            tgt = batch[1].cuda()
            pred_tgt = torch.full((src.shape[0], max_len), sym_set['[nop]'], dtype=torch.long).cuda()
            context = model.encoder(src)

            pred_tgt[:, 0] = sym_set['[start]']
            for i in range(0, max_len - 1):
                preds = torch.argmax(F.softmax(model.decoder(pred_tgt, context), dim=2), dim=2) 
                pred_tgt[:, i + 1] = preds[:, i]

            for i in range(0, tgt.shape[0]):
                list_targets.append(model.idx_to_smiles(tgt[i], syms))
                list_preds.append(model.idx_to_smiles(pred_tgt[i], syms))

    return list_targets, list_preds

def inference_multiple(model, data_loader, syms, sym_set, max_len, decoding_method='beam', beam_width=3, topk=10):
    list_targets = []
    list_preds = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            src = batch[0].cuda()
            tgt = batch[1].cuda()
            context = model.encoder(src)
            batch_preds = []
            for i in range(src.shape[0]):
                initial_seq = [sym_set['[start]']]
                if decoding_method == 'greedy':
                    preds = greedy_decode(model, context[i:i+1], initial_seq, max_len, sym_set)
                    batch_preds.append([model.idx_to_smiles(torch.tensor(preds).cpu(), syms)])
                elif decoding_method == 'beam':
                    beams = beam_search_decode(model, context[i:i+1], initial_seq, max_len, sym_set, beam_width, topk)
                    smiles_candidates = [model.idx_to_smiles(torch.tensor(beam).cpu(), syms) for beam in beams]
                    batch_preds.append(smiles_candidates)
                elif decoding_method == 'topk':
                    samples = topk_sampling_decode(model, context[i:i+1], initial_seq, max_len, sym_set, topk)
                    smiles_candidates = [model.idx_to_smiles(torch.tensor(seq).cpu(), syms) for seq in samples]
                    batch_preds.append(smiles_candidates)
                else:
                    raise ValueError("Unknown decoding_method: {}".format(decoding_method))
            for i in range(tgt.shape[0]):
                list_targets.append(model.idx_to_smiles(tgt[i], syms))
                list_preds.append(batch_preds[i])
    return list_targets, list_preds

def greedy_decode(model, context, initial_seq, max_len, sym_set):
    current_seq = initial_seq[:]
    for _ in range(max_len - 1):
        pred_input = torch.tensor([current_seq]).cuda()
        logits = model.decoder(pred_input, context)
        next_token_logits = logits[0, len(current_seq) - 1]
        next_token = torch.argmax(torch.softmax(next_token_logits, dim=0)).item()
        current_seq.append(next_token)
        if next_token == sym_set.get('[end]', -1):
            break
    return current_seq


def beam_search_decode(model, context, initial_seq, max_len, sym_set,
                       beam_width, top_k):
    beams = [(initial_seq, 0)]
    completed_beams = []

    for _ in range(max_len - 1):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == sym_set.get('[end]', -1):
                completed_beams.append((seq, score))
                continue

            pred_input = torch.tensor([seq]).cuda()
            logits = model.decoder(pred_input, context)
            next_token_logits = logits[0, len(seq) - 1]
            log_probs = torch.log_softmax(next_token_logits, dim=0)
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)

            for j in range(beam_width):
                token = topk_indices[j].item()
                new_seq = seq + [token]
                new_score = score + topk_log_probs[j].item()
                new_beams.append((new_seq, new_score))

        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        if not beams:
            break

    completed_beams.extend(beams)
    completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)

    if len(completed_beams) < top_k:
        end_id = sym_set.get('[end]', -1)
        extra = []
        for seq, score in completed_beams:
            if len(extra) + len(completed_beams) >= top_k:
                break
            if seq[-1] == end_id:          
                continue
            cur_seq, cur_score = seq[:], score
            while len(cur_seq) < max_len and cur_seq[-1] != end_id:
                pred_input = torch.tensor([cur_seq]).cuda()
                logits = model.decoder(pred_input, context)
                next_token_logits = logits[0, len(cur_seq) - 1]
                next_token = torch.argmax(next_token_logits).item()
                cur_seq.append(next_token)
                cur_score += torch.log_softmax(next_token_logits, dim=0)[next_token].item()
            extra.append((cur_seq, cur_score))
        completed_beams.extend(extra)
        completed_beams = sorted(completed_beams, key=lambda x: x[1], reverse=True)

    if len(completed_beams) < top_k:
        completed_beams += completed_beams[-1:] * (top_k - len(completed_beams))

    return [seq for seq, _ in completed_beams[:top_k]]





def topk_sampling_decode(model, context, initial_seq, max_len, sym_set, num_samples):
    sequences = []
    for _ in range(num_samples):
        seq = initial_seq.copy()
        for _ in range(max_len - 1):
            if seq[-1] == sym_set.get('[end]', -1):
                break
            pred_input = torch.tensor([seq]).cuda()
            logits = model.decoder(pred_input, context)
            next_token_logits = logits[0, len(seq) - 1]
            topk_probs, topk_indices = torch.topk(torch.softmax(next_token_logits, dim=0), num_samples)
            topk_probs = topk_probs / topk_probs.sum()
            token = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
            seq.append(token)
        sequences.append(seq)
    return sequences

