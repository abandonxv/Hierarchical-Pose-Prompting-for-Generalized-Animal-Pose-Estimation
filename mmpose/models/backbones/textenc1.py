import clip
import torch
from typing import List
import torch.nn as nn
from collections import OrderedDict
from ..builder import BACKBONES, HEADS
from mmpose.registry import MODELS

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import functools
from typing import Union
from typing import List, Tuple
@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class CLIPTokenizer:
    def __init__(self):
        bpe_path = "/root/mmpose/bpe_simple_vocab_16e6.txt.gz"
        self.tokenizer = SimpleTokenizer(bpe_path)
        # 特殊 token IDs
        self.sot_token = self.tokenizer.encoder["<|startoftext|>"]
        self.eot_token = self.tokenizer.encoder["<|endoftext|>"]
        self.pad_token = 0  # 填充 token
        self.context_length = 24  # CLIP 固定序列长度
        
    def tokenize(self, texts: list):
        """
        模拟 clip.tokenize 输出: [B, 77]
        """
        all_tokens = []
        for text in texts:
            # 1. 添加起始 token
            tokens = [self.sot_token]

            # 2. 编码文本内容
            text_tokens = self.tokenizer.encode(text)
            tokens.extend(text_tokens)

            # 3. 添加结束 token
            tokens.append(self.eot_token)

            # 4. 截断或填充
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length - 1] + [self.eot_token]
            else:
                tokens += [self.pad_token] * (self.context_length - len(tokens))

            all_tokens.append(tokens)

        return torch.tensor(all_tokens, dtype=torch.long)


class PromptInserter(nn.Module):
    def __init__(self, ctx_lens: Tuple[int, int, int], embed_dim: int, use_pos_shift: bool = False):
        super().__init__()
        self.ctx_lens = ctx_lens
        self.total_ctx = sum(ctx_lens)
        self.embed_dim = embed_dim
        self.use_pos_shift = use_pos_shift

        # 三段可学习 ctx token
        self.ctx_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(l, embed_dim) * 0.02) if l > 0 else None
            for l in ctx_lens
        ])

        if use_pos_shift:
            self.ctx_pos_offsets = nn.Parameter(torch.zeros(self.total_ctx))

    def forward(self, embedded_tokens: torch.Tensor, tokenized_ids: torch.Tensor,
                insert_pos: Tuple[int, int] = (1, -1), position_ids: torch.Tensor = None):
        """
        embedded_tokens: (B, L, D)
        tokenized_ids: (B, L)
        insert_pos: tuple of (start_idx, end_idx) for class tokens, e.g. (1, -1)
        position_ids: (L, D) or (B, L, D) positional embedding (optional)
        """
        B, L, D = embedded_tokens.shape
        i1, i2 = insert_pos

        # 展开ctx参数，batch复制
        ctx_front = self.ctx_tokens[0].unsqueeze(0).expand(B, -1, -1) if self.ctx_tokens[0] is not None else None
        ctx_middle = self.ctx_tokens[1].unsqueeze(0).expand(B, -1, -1) if self.ctx_tokens[1] is not None else None
        ctx_end = self.ctx_tokens[2].unsqueeze(0).expand(B, -1, -1) if self.ctx_tokens[2] is not None else None

        # 拆分原embedding
        x0 = embedded_tokens[:, :i1, :]  # prefix before class token
        x1 = embedded_tokens[:, i1:i2, :]  # class token (middle)
        x2 = embedded_tokens[:, i2:, :]  # suffix after class token

        parts = []
        if ctx_front is not None:
            parts.append(ctx_front)
        parts.append(x0)
        if ctx_middle is not None:
            parts.append(ctx_middle)
        parts.append(x1)
        if ctx_end is not None:
            parts.append(ctx_end)
        parts.append(x2)

        new_tokens = torch.cat(parts, dim=1)  # [B, L+ctx, D]

        if position_ids is not None:
            # position_ids 形状为 (L, D) 或 (B, L, D)，要构造新的 position embedding
            # 这里默认 position_ids 是 positional_embedding[:seq_len], (seq_len, D)
            # 先拆分 position embedding
            if position_ids.dim() == 2:
                pos = position_ids.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)
            else:
                pos = position_ids

            pos0 = pos[:, :i1, :]
            pos1 = pos[:, i1:i2, :]
            pos2 = pos[:, i2:, :]

            ctx_pos_list = []
            for ctx in [ctx_front, ctx_middle, ctx_end]:
                if ctx is not None:
                    ctx_pos_list.append(torch.zeros_like(ctx))  # 用零初始化，或者你可以定义可学习的位置偏置

            # 拼接
            pos_parts = []
            idx = 0
            if ctx_front is not None:
                pos_parts.append(ctx_pos_list[idx])
                idx += 1
            pos_parts.append(pos0)
            if ctx_middle is not None:
                pos_parts.append(ctx_pos_list[idx])
                idx += 1
            pos_parts.append(pos1)
            if ctx_end is not None:
                pos_parts.append(ctx_pos_list[idx])
                idx += 1
            pos_parts.append(pos2)

            new_pos = torch.cat(pos_parts, dim=1)
        else:
            new_pos = None

        return new_tokens, new_pos
class PromptSimplifier:
    def __init__(self):
        self.stop_words = set([
            "a", "an", "the", "with", "and", "or", "but", "as", "of", "at", "by",
            "for", "from", "in", "on", "to", "up", "off", "is", "its", "it's", "that",
            "this", "was", "are", "were", "has", "have", "had", "be", "being", "been",
            "slightly", "very", "just"
        ])

    def simplify(self, text: str) -> str:
        # 基本清洗：转小写、去标点、分词
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点
        tokens = text.strip().split()  # 简单分词（按空格）

        # 移除停用词
        cleaned = [word for word in tokens if word not in self.stop_words]

        return ' '.join(cleaned)

@MODELS.register_module()
class CLIPTextEncoder2(nn.Module):
    def __init__(self, context_length=24,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None,
                 ctx_lens: Tuple[int, int, int] = (1, 2, 1),
                 use_pos_shift: bool = False,
                 **kwargs):
        super().__init__()
        self.ctxlen=sum(ctx_lens)
        self.pretrained = pretrained

        self.context_length = context_length
        self.transformer_width = transformer_width
        self.embed_dim = embed_dim

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Linear(transformer_width, embed_dim)  # 默认有 bias

        # 新增 prompt inserter 子模块
        self.prompt_inserter = PromptInserter(ctx_lens=ctx_lens, embed_dim=transformer_width, use_pos_shift=use_pos_shift)

        # 初始化参数
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.prompt_simplifier = PromptSimplifier()
    def safe_tokenize(self, texts: List[str]):
        out = clip.tokenize(texts, context_length=self.context_length)
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.long)
        return out

    def build_attention_mask(self):
        ctxlen = self.ctxlen
        mask = torch.empty(self.context_length+ctxlen, self.context_length+ctxlen)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text: Union[List[str], torch.LongTensor]) -> torch.Tensor:
        if isinstance(text, list):
            # ✅ 在这里调用简化器
            text = [self.prompt_simplifier.simplify(t) for t in text]
        tokens = self.safe_tokenize(text)
        tokens = tokens.to(self.token_embedding.weight.device)

        # token embedding
        x = self.token_embedding(tokens)  # [B, L, D]
        seq_len = x.shape[1]

        pos_emb = self.positional_embedding[:seq_len, :]  # [L, D]

        # 用 prompt inserter 动态插入可学习 ctx tokens 和对应位置编码
        if self.prompt_inserter is not None:
            x, pos_emb = self.prompt_inserter(x, tokens, insert_pos=(1, -1),
                                              position_ids=pos_emb)
        
        print(x.shape)
        #print(f"pos shape = {pos_emb.shape}")
        # 加位置编码
        x = x + pos_emb # [B, L+ctx, D]
        #print(f"x shape = {x.shape}")
        x = x.permute(1, 0, 2)  # (L+ctx, B, D)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, L+ctx, D)

        x = self.ln_final(x)
        x = self.text_projection(x)
        
        return x

