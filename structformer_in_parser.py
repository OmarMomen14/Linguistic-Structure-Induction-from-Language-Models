
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

##########################################
def _get_activation_fn(activation):
  """Get specified activation function."""
  if activation == "relu":
    return nn.ReLU()
  elif activation == "gelu":
    return nn.GELU()
  elif activation == "leakyrelu":
    return nn.LeakyReLU()

  raise RuntimeError(
      "activation should be relu/gelu, not {}".format(activation))


class Conv1d(nn.Module):
  """1D convolution layer."""

  def __init__(self, hidden_size, kernel_size, dilation=1):
    """Initialization.

    Args:
      hidden_size: dimension of input embeddings
      kernel_size: convolution kernel size
      dilation: the spacing between the kernel points
    """
    super(Conv1d, self).__init__()

    if kernel_size % 2 == 0:
      padding = (kernel_size // 2) * dilation
      self.shift = True
    else:
      padding = ((kernel_size - 1) // 2) * dilation
      self.shift = False
    self.conv = nn.Conv1d(
        hidden_size,
        hidden_size,
        kernel_size,
        padding=padding,
        dilation=dilation)

  def forward(self, x):
    """Compute convolution.

    Args:
      x: input embeddings
    Returns:
      conv_output: convolution results
    """

    if self.shift:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)[:, 1:]
    else:
      return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiheadAttention(nn.Module):
  """Multi-head self-attention layer."""

  def __init__(self,
               embed_dim,
               num_heads,
               dropout=0.,
               bias=True,
               v_proj=True,
               out_proj=True,
               relative_bias=True):
    """Initialization.

    Args:
      embed_dim: dimension of input embeddings
      num_heads: number of self-attention heads
      dropout: dropout rate
      bias: bool, indicate whether include bias for linear transformations
      v_proj: bool, indicate whether project inputs to new values
      out_proj: bool, indicate whether project outputs to new values
      relative_bias: bool, indicate whether use a relative position based
        attention bias
    """

    super(MultiheadAttention, self).__init__()
    self.embed_dim = embed_dim

    self.num_heads = num_heads
    self.drop = nn.Dropout(dropout)
    self.head_dim = embed_dim // num_heads
    assert self.head_dim * num_heads == self.embed_dim, ("embed_dim must be "
                                                         "divisible by "
                                                         "num_heads")

    self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    if v_proj:
      self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    else:
      self.v_proj = nn.Identity()

    if out_proj:
      self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
    else:
      self.out_proj = nn.Identity()

    if relative_bias:
      self.relative_bias = nn.Parameter(torch.zeros((self.num_heads, 512)))
    else:
      self.relative_bias = None

    self._reset_parameters()

  def _reset_parameters(self):
    """Initialize attention parameters."""

    init.xavier_uniform_(self.q_proj.weight)
    init.constant_(self.q_proj.bias, 0.)

    init.xavier_uniform_(self.k_proj.weight)
    init.constant_(self.k_proj.bias, 0.)

    if isinstance(self.v_proj, nn.Linear):
      init.xavier_uniform_(self.v_proj.weight)
      init.constant_(self.v_proj.bias, 0.)

    if isinstance(self.out_proj, nn.Linear):
      init.xavier_uniform_(self.out_proj.weight)
      init.constant_(self.out_proj.bias, 0.)

  def forward(self, query, key_padding_mask=None, attn_mask=None):
    """Compute multi-head self-attention.

    Args:
      query: input embeddings
      key_padding_mask: 3D mask that prevents attention to certain positions
      attn_mask: 3D mask that rescale the attention weight at each position
    Returns:
      attn_output: self-attention output
    """

    length, bsz, embed_dim = query.size()
    assert embed_dim == self.embed_dim

    head_dim = embed_dim // self.num_heads
    assert head_dim * self.num_heads == embed_dim, ("embed_dim must be "
                                                    "divisible by num_heads")
    scaling = float(head_dim)**-0.5

    q = self.q_proj(query)
    k = self.k_proj(query)
    v = self.v_proj(query)

    q = q * scaling

    if attn_mask is not None:
      assert list(attn_mask.size()) == [bsz * self.num_heads,
                                        query.size(0), query.size(0)]

    q = q.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)
    k = k.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)
    v = v.contiguous().view(length, bsz * self.num_heads,
                            head_dim).transpose(0, 1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(
        attn_output_weights.size()) == [bsz * self.num_heads, length, length]

    if self.relative_bias is not None:
      pos = torch.arange(length, device=query.device)
      relative_pos = torch.abs(pos[:, None] - pos[None, :]) + 256
      relative_pos = relative_pos[None, :, :].expand(bsz * self.num_heads, -1,
                                                     -1)

      relative_bias = self.relative_bias.repeat_interleave(bsz, dim=0)
      relative_bias = relative_bias[:, None, :].expand(-1, length, -1)
      relative_bias = torch.gather(relative_bias, 2, relative_pos)
      attn_output_weights = attn_output_weights + relative_bias

    if key_padding_mask is not None:
      attn_output_weights = attn_output_weights + key_padding_mask

    if attn_mask is None:
      attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
    else:
      attn_output_weights = torch.sigmoid(attn_output_weights) * attn_mask

    attn_output_weights = self.drop(attn_output_weights)

    attn_output = torch.bmm(attn_output_weights, v)

    assert list(attn_output.size()) == [bsz * self.num_heads, length, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(
        length, bsz, embed_dim)
    attn_output = self.out_proj(attn_output)

    return attn_output


class TransformerLayer(nn.Module):
  """TransformerEncoderLayer is made up of self-attn and feedforward network."""

  def __init__(self,
               d_model,
               nhead,
               dim_feedforward=2048,
               dropout=0.1,
               dropatt=0.1,
               activation="leakyrelu",
               relative_bias=True):
    """Initialization.

    Args:
      d_model: dimension of inputs
      nhead: number of self-attention heads
      dim_feedforward: dimension of hidden layer in feedforward layer
      dropout: dropout rate
      dropatt: drop attention rate
      activation: activation function
      relative_bias: bool, indicate whether use a relative position based
        attention bias
    """

    super(TransformerLayer, self).__init__()
    self.self_attn = MultiheadAttention(
        d_model, nhead, dropout=dropatt, relative_bias=relative_bias)
    # Implementation of Feedforward model
    self.feedforward = nn.Sequential(
        nn.LayerNorm(d_model), nn.Linear(d_model, dim_feedforward),
        _get_activation_fn(activation), nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model))

    self.norm = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.nhead = nhead

  def forward(self, src, attn_mask=None, key_padding_mask=None):
    """Pass the input through the encoder layer.

    Args:
      src: the sequence to the encoder layer (required).
      attn_mask: the mask for the src sequence (optional).
      key_padding_mask: the mask for the src keys per batch (optional).
    Returns:
      src3: the output of transformer layer, share the same shape as src.
    """
    src2 = self.self_attn(
        self.norm(src), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    src2 = src + self.dropout1(src2)
    src3 = self.feedforward(src2)
    src3 = src2 + self.dropout2(src3)

    return src3

##########################################
def cumprod(x, reverse=False, exclusive=False):
  """cumulative product."""
  if reverse:
    x = x.flip([-1])

  if exclusive:
    x = F.pad(x[:, :, :-1], (1, 0), value=1)

  cx = x.cumprod(-1)

  if reverse:
    cx = cx.flip([-1])
  return cx


def cumsum(x, reverse=False, exclusive=False):
  """cumulative sum."""
  bsz, _, length = x.size()
  device = x.device
  if reverse:
    if exclusive:
      w = torch.ones([bsz, length, length], device=device).tril(-1)
    else:
      w = torch.ones([bsz, length, length], device=device).tril(0)
    cx = torch.bmm(x, w)
  else:
    if exclusive:
      w = torch.ones([bsz, length, length], device=device).triu(1)
    else:
      w = torch.ones([bsz, length, length], device=device).triu(0)
    cx = torch.bmm(x, w)
  return cx


def cummin(x, reverse=False, exclusive=False, max_value=1e9):
  """cumulative min."""
  if reverse:
    if exclusive:
      x = F.pad(x[:, :, 1:], (0, 1), value=max_value)
    x = x.flip([-1]).cummin(-1)[0].flip([-1])
  else:
    if exclusive:
      x = F.pad(x[:, :, :-1], (1, 0), value=max_value)
    x = x.cummin(-1)[0]
  return x


class Transformer_Front(nn.Module):
  """Transformer model."""

  def __init__(self,
               hidden_size,
               nlayers,
               ntokens,
               nhead=8,
               dropout=0.1,
               dropatt=0.1,
               relative_bias=True,
               pos_emb=False,
               pad=0):
    """Initialization.

    Args:
      hidden_size: dimension of inputs and hidden states
      nlayers: number of layers
      ntokens: number of output categories
      nhead: number of self-attention heads
      dropout: dropout rate
      dropatt: drop attention rate
      relative_bias: bool, indicate whether use a relative position based
        attention bias
      pos_emb: bool, indicate whether use a learnable positional embedding
      pad: pad token index
    """

    super(Transformer_Front, self).__init__()

    self.drop = nn.Dropout(dropout)

    self.emb = nn.Embedding(ntokens, hidden_size)
    if pos_emb:
      self.pos_emb = nn.Embedding(500, hidden_size)

    self.layers = nn.ModuleList([
        TransformerLayer(hidden_size, nhead, hidden_size * 4, dropout,
                                dropatt=dropatt, relative_bias=relative_bias)
        for _ in range(nlayers)])

    self.norm = nn.LayerNorm(hidden_size)

    self.init_weights()

    self.nlayers = nlayers
    self.nhead = nhead
    self.ntokens = ntokens
    self.hidden_size = hidden_size
    self.pad = pad

  def init_weights(self):
    """Initialize token embedding and output bias."""
    initrange = 0.1
    self.emb.weight.data.uniform_(-initrange, initrange)
    if hasattr(self, 'pos_emb'):
      self.pos_emb.weight.data.uniform_(-initrange, initrange)
    

  def visibility(self, x, device):
    """Mask pad tokens."""
    visibility = (x != self.pad).float()
    visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
    visibility = torch.repeat_interleave(visibility, self.nhead, dim=0)
    return visibility.log()

  def encode(self, x, pos):
    """Standard transformer encode process."""
    h = self.emb(x)
    if hasattr(self, 'pos_emb'):
      h = h + self.pos_emb(pos)
    h_list = []
    visibility = self.visibility(x, x.device)

    for i in range(self.nlayers):
      h_list.append(h)
      h = self.layers[i](
          h.transpose(0, 1), key_padding_mask=visibility).transpose(0, 1)

    output = h
    h_array = torch.stack(h_list, dim=2)

    return output, h_array

  def forward(self, x, pos):
    """Pass the input through the encoder layer.

    Args:
      x: input tokens (required).
      pos: position for each token (optional).
    Returns:
      output: probability distributions for missing tokens.
      state_dict: parsing results and raw output
    """

    batch_size, length = x.size()

    raw_output, _ = self.encode(x, pos)
    raw_output = self.norm(raw_output)
    raw_output = self.drop(raw_output)

    return {'raw_output': raw_output}


class Transformer_Rear(nn.Module):
  """Transformer model."""

  def __init__(self,
               hidden_size,
               nlayers,
               ntokens,
               nhead=8,
               dropout=0.1,
               dropatt=0.1,
               relative_bias=True,
               pos_emb=False,
               pad=0):
    """Initialization.

    Args:
      hidden_size: dimension of inputs and hidden states
      nlayers: number of layers
      ntokens: number of output categories
      nhead: number of self-attention heads
      dropout: dropout rate
      dropatt: drop attention rate
      relative_bias: bool, indicate whether use a relative position based
        attention bias
      pos_emb: bool, indicate whether use a learnable positional embedding
      pad: pad token index
    """

    super(Transformer_Rear, self).__init__()

    self.drop = nn.Dropout(dropout)

    self.emb = nn.Embedding(ntokens, hidden_size)
    if pos_emb:
      self.pos_emb = nn.Embedding(500, hidden_size)

    self.layers = nn.ModuleList([
        TransformerLayer(hidden_size, nhead, hidden_size * 4, dropout,
                                dropatt=dropatt, relative_bias=relative_bias)
        for _ in range(nlayers)])

    self.norm = nn.LayerNorm(hidden_size)

    self.output_layer = nn.Linear(hidden_size, ntokens)

    self.init_weights()

    self.nlayers = nlayers
    self.nhead = nhead
    self.ntokens = ntokens
    self.hidden_size = hidden_size
    self.pad = pad

  def init_weights(self):
    """Initialize token embedding and output bias."""
    initrange = 0.1
    self.emb.weight.data.uniform_(-initrange, initrange)
    if hasattr(self, 'pos_emb'):
      self.pos_emb.weight.data.uniform_(-initrange, initrange)
    self.output_layer.bias.data.fill_(0)

  def visibility(self, x, device):
    """Mask pad tokens."""
    visibility = (x != self.pad).float()
    visibility = visibility[:, None, :].expand(-1, x.size(1), -1)
    visibility = torch.repeat_interleave(visibility, self.nhead, dim=0)
    return visibility.log()

  def encode(self, x, pos, att_mask, h):
    """Structformer encoding process."""

    visibility = self.visibility(x, x.device)
    
    if hasattr(self, 'pos_emb'):
      assert pos.max() < 500
      h = h + self.pos_emb(pos)
    for i in range(self.nlayers):
      h = self.layers[i](
          h.transpose(0, 1), attn_mask=att_mask[i],
          key_padding_mask=visibility).transpose(0, 1)
    return h

  def forward(self, x, pos):
    """Pass the input through the encoder layer.

    Args:
      x: input tokens (required).
      pos: position for each token (optional).
    Returns:
      output: probability distributions for missing tokens.
      state_dict: parsing results and raw output
    """

    batch_size, length = x.size()

    raw_output, _ = self.encode(x, pos)
    raw_output = self.norm(raw_output)
    raw_output = self.drop(raw_output)

    output = self.output_layer(raw_output)
    return output.view(batch_size * length, -1), {'raw_output': raw_output,}


class StructFormer_In_Parser(nn.Module):
  """StructFormer model."""

  def __init__(self,
               hidden_size,
               nlayers,
               ntokens,
               nhead=8,
               dropout=0.1,
               dropatt=0.1,
               relative_bias=False,
               pos_emb=False,
               front_layers=2,
               rear_layers=6,
               pad=0,
               n_parser_layers=4,
               conv_size=9,
               relations=('head', 'child'),
               weight_act='softmax'):
    """Initialization.

    Args:
      hidden_size: dimension of inputs and hidden states
      nlayers: number of layers
      ntokens: number of output categories
      nhead: number of self-attention heads
      dropout: dropout rate
      dropatt: drop attention rate
      relative_bias: bool, indicate whether use a relative position based
        attention bias
      pos_emb: bool, indicate whether use a learnable positional embedding
      pad: pad token index
      n_parser_layers: number of parsing layers
      conv_size: convolution kernel size for parser
      relations: relations that are used to compute self attention
      weight_act: relations distribution activation function
    """

    super(StructFormer_In_Parser, self).__init__()
    
    self.transformer_front = Transformer_Front(
        hidden_size,
        nlayers=front_layers,
        ntokens=ntokens,
        nhead=nhead,
        dropout=dropout,
        dropatt=dropatt,
        relative_bias=relative_bias,
        pos_emb=pos_emb,
        pad=pad
    )
    
    self.transformer_rear = Transformer_Rear(
        hidden_size,
        nlayers=rear_layers,
        ntokens=ntokens,
        nhead=nhead,
        dropout=dropout,
        dropatt=dropatt,
        relative_bias=relative_bias,
        pos_emb=pos_emb,
        pad=pad
    )
    self.transformer_rear.emb.weight = self.transformer_front.emb.weight
    self.transformer_rear.output_layer.weight = self.transformer_front.emb.weight
    if pos_emb:
        self.transformer_rear.pos_emb.weight = self.transformer_front.pos_emb.weight

    self.parser_layers = nn.ModuleList([
        nn.Sequential(Conv1d(hidden_size, conv_size),
                      nn.LayerNorm(hidden_size, elementwise_affine=False),
                      nn.Tanh()) for i in range(n_parser_layers)])

    self.distance_ff = nn.Sequential(
        Conv1d(hidden_size, 2),
        nn.LayerNorm(hidden_size, elementwise_affine=False), nn.Tanh(),
        nn.Linear(hidden_size, 1))

    self.height_ff = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size, elementwise_affine=False), nn.Tanh(),
        nn.Linear(hidden_size, 1))

    n_rel = len(relations)
    self._rel_weight = nn.Parameter(torch.zeros((self.transformer_rear.nlayers, nhead, n_rel)))
    self._rel_weight.data.normal_(0, 0.1)

    self._scaler = nn.Parameter(torch.zeros(2))

    self.n_parse_layers = n_parser_layers
    self.weight_act = weight_act
    self.relations = relations

  @property
  def scaler(self):
    return self._scaler.exp()

  @property
  def rel_weight(self):
    if self.weight_act == 'sigmoid':
      return torch.sigmoid(self._rel_weight)
    elif self.weight_act == 'softmax':
      return torch.softmax(self._rel_weight, dim=-1)

  def parse(self, x, h):
    """Parse input sentence.

    Args:
      x: input tokens (required).
      pos: position for each token (optional).
    Returns:
      distance: syntactic distance
      height: syntactic height
    """

    mask = (x != self.transformer_rear.pad)
    mask_shifted = F.pad(mask[:, 1:], (0, 1), value=0)

    for i in range(self.n_parse_layers):
      h = h.masked_fill(~mask[:, :, None], 0)
      h = self.parser_layers[i](h)

    height = self.height_ff(h).squeeze(-1)
    height.masked_fill_(~mask, -1e9)

    distance = self.distance_ff(h).squeeze(-1)
    distance.masked_fill_(~mask_shifted, 1e9)

    # Calbrating the distance and height to the same level
    length = distance.size(1)
    height_max = height[:, None, :].expand(-1, length, -1)
    height_max = torch.cummax(
        height_max.triu(0) - torch.ones_like(height_max).tril(-1) * 1e9,
        dim=-1)[0].triu(0)

    margin_left = torch.relu(
        F.pad(distance[:, :-1, None], (0, 0, 1, 0), value=1e9) - height_max)
    margin_right = torch.relu(distance[:, None, :] - height_max)
    margin = torch.where(margin_left > margin_right, margin_right,
                         margin_left).triu(0)

    margin_mask = torch.stack([mask_shifted] + [mask] * (length - 1), dim=1)
    margin.masked_fill_(~margin_mask, 0)
    margin = margin.max()

    distance = distance - margin

    return distance, height

  def compute_block(self, distance, height):
    """Compute constituents from distance and height."""

    beta_logits = (distance[:, None, :] - height[:, :, None]) * self.scaler[0]

    gamma = torch.sigmoid(-beta_logits)
    ones = torch.ones_like(gamma)

    block_mask_left = cummin(
        gamma.tril(-1) + ones.triu(0), reverse=True, max_value=1)
    block_mask_left = block_mask_left - F.pad(
        block_mask_left[:, :, :-1], (1, 0), value=0)
    block_mask_left.tril_(0)

    block_mask_right = cummin(
        gamma.triu(0) + ones.tril(-1), exclusive=True, max_value=1)
    block_mask_right = block_mask_right - F.pad(
        block_mask_right[:, :, 1:], (0, 1), value=0)
    block_mask_right.triu_(0)

    block_p = block_mask_left[:, :, :, None] * block_mask_right[:, :, None, :]
    block = cumsum(block_mask_left).tril(0) + cumsum(
        block_mask_right, reverse=True).triu(1)

    return block_p, block

  def compute_head(self, height):
    """Estimate head for each constituent."""

    _, length = height.size()
    head_logits = height * self.scaler[1]
    index = torch.arange(length, device=height.device)

    mask = (index[:, None, None] <= index[None, None, :]) * (
        index[None, None, :] <= index[None, :, None])
    head_logits = head_logits[:, None, None, :].repeat(1, length, length, 1)
    head_logits.masked_fill_(~mask[None, :, :, :], -1e9)

    head_p = torch.softmax(head_logits, dim=-1)

    return head_p

  def generate_mask(self, x, distance, height):
    """Compute head and cibling distribution for each token."""

    bsz, length = x.size()

    eye = torch.eye(length, device=x.device, dtype=torch.bool)
    eye = eye[None, :, :].expand((bsz, -1, -1))

    block_p, block = self.compute_block(distance, height)
    head_p = self.compute_head(height)
    head = torch.einsum('blij,bijh->blh', block_p, head_p)
    head = head.masked_fill(eye, 0)
    child = head.transpose(1, 2)
    cibling = torch.bmm(head, child).masked_fill(eye, 0)

    rel_list = []
    if 'head' in self.relations:
      rel_list.append(head)
    if 'child' in self.relations:
      rel_list.append(child)
    if 'cibling' in self.relations:
      rel_list.append(cibling)

    rel = torch.stack(rel_list, dim=1)

    rel_weight = self.rel_weight

    dep = torch.einsum('lhr,brij->lbhij', rel_weight, rel)
    att_mask = dep.reshape(self.transformer_rear.nlayers, bsz * self.transformer_rear.nhead, length, length)

    return att_mask, cibling, head, block

  def forward(self, x, pos):
    """Pass the input through the encoder layer.

    Args:
      x: input tokens (required).
      pos: position for each token (optional).
    Returns:
      output: probability distributions for missing tokens.
      state_dict: parsing results and raw output
    """

    batch_size, length = x.size()

    raw_output_1, _ = self.transformer_front.encode(x, pos)
    raw_output_1 = self.transformer_front.norm(raw_output_1)
    raw_output_1 = self.transformer_front.drop(raw_output_1)
    
    distance, height = self.parse(x, raw_output_1)
    att_mask, cibling, head, block = self.generate_mask(x, distance, height)

    raw_output_2 = self.transformer_rear.encode(x, pos, att_mask, raw_output_1)
    raw_output_2 = self.transformer_rear.norm(raw_output_2)
    raw_output_2 = self.transformer_rear.drop(raw_output_2)

    output = self.transformer_rear.output_layer(raw_output_2)

    return output.view(batch_size * length, -1), \
        {'raw_output': raw_output_2, 'distance': distance, 'height': height,
         'cibling': cibling, 'head': head, 'block': block}



##########################################
# Clasication Head For BabyLM Evaluation Tasks
##########################################
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

##########################################
# HuggingFace Config
##########################################
class StructFormer_In_ParserConfig(PretrainedConfig):
    model_type = "structformer_in_parser"

    def __init__(
        self,
        hidden_size=512,
        nlayers=8,
        ntokens=10_000,
        nhead=8,
        dropout=0.1,
        dropatt=0.1,
        relative_bias=False,
        pos_emb=False,
        pad=0,
        n_parser_layers=4,
        front_layers=2,
        rear_layers=6,
        conv_size=9,
        relations=('head', 'child'),
        weight_act='softmax',
        num_labels=1,
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.ntokens = ntokens
        self.nhead = nhead
        self.dropout = dropout
        self.dropatt = dropatt
        self.relative_bias = relative_bias
        self.pos_emb = pos_emb
        self.pad = pad
        self.n_parser_layers = n_parser_layers
        self.front_layers = front_layers
        self.rear_layers = rear_layers
        self.conv_size = conv_size
        self.relations = relations
        self.weight_act = weight_act
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range=initializer_range
        super().__init__(**kwargs)


##########################################
# HuggingFace Models
##########################################
class StructFormer_In_ParserModel(PreTrainedModel):
    config_class = StructFormer_In_ParserConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = StructFormer_In_Parser(
              hidden_size=config.hidden_size,
              nlayers=config.nlayers,
              ntokens=config.ntokens,
              nhead=config.nhead,
              dropout=config.dropout,
              dropatt=config.dropatt,
              relative_bias=config.relative_bias,
              pos_emb=config.pos_emb,
              pad=config.pad,
              n_parser_layers=config.n_parser_layers,
              front_layers=config.front_layers,
              rear_layers=config.rear_layers,
              conv_size=config.conv_size,
              relations=config.relations,
              weight_act=config.weight_act
        )
        self.config = config

    def parse(self, input_ids, **kwargs):
      x = input_ids
      batch_size, length = x.size()
      pos = kwargs['position_ids'] if 'position_ids' in kwargs.keys() else torch.arange(length, device=x.device).expand(batch_size, length)
      
      sf_output = self.model(x, pos)
      
      return sf_output[1]
    
    def forward(self, input_ids, labels=None, **kwargs):
        x = input_ids
        batch_size, length = x.size()
        pos = kwargs['position_ids'] if 'position_ids' in kwargs.keys() else torch.arange(length, device=x.device).expand(batch_size, length)
        
        sf_output = self.model(x, pos)
        
        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(sf_output[0], labels.reshape(-1))
        
        return MaskedLMOutput(
            loss=loss, # shape: 1
            logits=sf_output[0].view(batch_size, length, -1), # shape: (batch_size, length, ntokens)
            hidden_states=None,
            attentions=None
        )

class StructFormer_In_ParserModelForSequenceClassification(PreTrainedModel):
    config_class = StructFormer_In_ParserConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = StructFormer_In_Parser(
              hidden_size=config.hidden_size,
              nlayers=config.nlayers,
              ntokens=config.ntokens,
              nhead=config.nhead,
              dropout=config.dropout,
              dropatt=config.dropatt,
              relative_bias=config.relative_bias,
              pos_emb=config.pos_emb,
              pad=config.pad,
              n_parser_layers=config.n_parser_layers,
              front_layers=config.front_layers,
              rear_layers=config.rear_layers,
              conv_size=config.conv_size,
              relations=config.relations,
              weight_act=config.weight_act
        )
        self.config = config
        self.num_labels = config.num_labels
        self.model.classifier = ClassificationHead(config)
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, labels=None, **kwargs):
        x = input_ids
        batch_size, length = x.size()
        pos = kwargs['position_ids'] if 'position_ids' in kwargs.keys() else torch.arange(length, device=x.device).expand(batch_size, length)
        
        sf_output = self.model(x, pos)
        
        logits = self.model.classifier(sf_output[1]['raw_output'])
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

