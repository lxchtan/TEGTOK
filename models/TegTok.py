import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

import warnings

from models.base_modify.bart_vae import PretrainedBartModel, BartConfig, BartModel, shift_tokens_right, _reorder_buffer, _make_linear_from_emb
# from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers.file_utils import ModelOutput


"""
  Encoder and Knowledge Encoder have the same config and initialize with same pre-trained params, but not tie weight.
"""

@dataclass
class Seq2SeqVAELMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
            used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    bow_logits: torch.FloatTensor = None
    bow_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class RAGModel(PretrainedBartModel):
  base_model_prefix = "model"
  authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

  def __init__(self, config: BartConfig, args):
    super().__init__(config)
    self.config = config
    self.args = args

    base_model = BartModel(config)
    self.model = base_model
    self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
    self.bow_predictor = nn.Sequential(
        nn.Linear(self.config.d_model, self.config.d_model),
        nn.Tanh(),
        nn.Linear(self.config.d_model, self.config.vocab_size)
      )

  def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
    old_num_tokens = self.model.shared.num_embeddings
    new_embeddings = super().resize_token_embeddings(new_num_tokens)
    self.model.shared = new_embeddings
    self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
    return new_embeddings

  def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
    if new_num_tokens <= old_num_tokens:
      new_bias = self.final_logits_bias[:, :new_num_tokens]
    else:
      extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
      new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    self.register_buffer("final_logits_bias", new_bias)

  def forward(
      self,
      input_ids,
      attention_mask=None,
      decoder_input_ids=None,
      decoder_attention_mask=None,
      encoder_outputs=None,
      knowledge_input_ids=None,
      knowledge_attention_mask=None,
      knowledge_encoder_outputs=None,
      past_key_values=None,
      labels=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=True,
      **unused,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
        Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
        config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
        (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

    Returns:

    Conditional generation example::

        >>> # Mask filling only works for bart-large
        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        >>> # ['good', 'great', 'all', 'really', 'very']
    """
    if "lm_labels" in unused:
      warnings.warn(
          "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
          FutureWarning,
      )
      labels = unused.pop("lm_labels")
    if "decoder_cached_states" in unused:
      warnings.warn(
          "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
          FutureWarning,
      )
      past_key_values = unused.pop("decoder_cached_states")
    if "decoder_past_key_values" in unused:
      warnings.warn(
          "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
          FutureWarning,
      )
      past_key_values = unused.pop("decoder_past_key_values")
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if labels is not None:
      use_cache = False
      if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        knowledge_input_ids=knowledge_input_ids,
        knowledge_attention_mask=knowledge_attention_mask,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        knowledge_encoder_outputs=knowledge_encoder_outputs,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

    # knowledge_decode_logits = F.linear(outputs.knowledge_cls_hidden, self.model.shared.weight, bias=self.final_logits_bias)
    knowledge_decode_logits = self.bow_predictor(outputs.knowledge_cls_hidden)
    bow_logits = knowledge_decode_logits

    masked_lm_loss = None
    bow_loss = None
    kl_loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      # TODO(SS): do we need to ignore pad tokens in labels?
      # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
      shifted_prediction_scores = lm_logits[:, :-1, :].contiguous()
      labels = labels[:, 1:].contiguous()
      masked_lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
      
      cal_bow_logits = bow_logits.unsqueeze(1).repeat((1, labels.size(-1), 1))
      bow_loss = loss_fct(cal_bow_logits.view(-1, cal_bow_logits.size(-1)), labels.view(-1))

      criterion = nn.KLDivLoss(reduction='batchmean')
      kl_loss = criterion(F.log_softmax(outputs.prior_logits, dim=-1), F.softmax(outputs.posterior_logits, dim=-1))
      

    if not return_dict:
      output = (lm_logits,) + outputs[1:]
      return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    return Seq2SeqVAELMOutput(
        loss=masked_lm_loss,
        logits=lm_logits,
        bow_logits=bow_logits,
        bow_loss=bow_loss,
        kl_loss=kl_loss,
        past_key_values=outputs.past_key_values,
        decoder_hidden_states=outputs.decoder_hidden_states,
        decoder_attentions=outputs.decoder_attentions,
        cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions,
    )

  def _prepare_decoder_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
    input_shape = input_ids.shape

    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
      attention_mask = input_ids.new_ones(input_shape)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

  @staticmethod
  def _expand_inputs_for_generation(
      input_ids: torch.LongTensor,
      expand_size: int = 1,
      is_encoder_decoder: bool = False,
      attention_mask: torch.LongTensor = None,
      encoder_outputs=None,
      knowledge_attention_mask=None,
      knowledge_encoder_outputs=None,
      **model_kwargs
  ):
    expanded_return_idx = (
        torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    )
    input_ids = input_ids.index_select(0, expanded_return_idx)

    if attention_mask is not None:
      model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
    
    if knowledge_attention_mask is not None:
      model_kwargs["knowledge_attention_mask"] = knowledge_attention_mask.index_select(0, expanded_return_idx)

    if is_encoder_decoder:
      assert encoder_outputs is not None
      encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
          0, expanded_return_idx
      )
      model_kwargs["encoder_outputs"] = encoder_outputs

      assert knowledge_encoder_outputs is not None
      knowledge_encoder_outputs["last_hidden_state"] = knowledge_encoder_outputs.last_hidden_state.index_select(
          0, expanded_return_idx
      )
      model_kwargs["knowledge_encoder_outputs"] = knowledge_encoder_outputs
    return input_ids, model_kwargs

  # First
  def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.LongTensor, model_kwargs):
    # retrieve encoder hidden states
    encoder = self.get_encoder()
    encoder_kwargs = {
        argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_") and not argument.startswith("knowledge_")
    }

    encoder_outputs = encoder(input_ids, return_dict=True, **encoder_kwargs)

    knowledge_input_ids = model_kwargs.get('knowledge_input_ids')
    knowledge_attention_mask = model_kwargs.get('knowledge_attention_mask')
    attention_mask = encoder_kwargs.pop('attention_mask')
    
    knowledge_encoder_outputs, knowledge_attention_mask, prior_logits, posterior_logits = self.model.cal_knowledge(
      encoder_outputs, knowledge_input_ids, knowledge_attention_mask,
      response_encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=True
    )
    # knowledge_encoder_outputs, knowledge_attention_mask, prior_logits, posterior_logits = self.model.cal_knowledge_top1(
    #   encoder_outputs, knowledge_input_ids, knowledge_attention_mask,
    #   response_encoder_outputs=None, output_attentions=None, output_hidden_states=None, return_dict=True
    # )
    self.knowledge_index = posterior_logits.argmax(dim=-1)
    encoder_kwargs['attention_mask'] = attention_mask

    model_kwargs["encoder_outputs"] = encoder_outputs
    model_kwargs["knowledge_encoder_outputs"] = knowledge_encoder_outputs
    model_kwargs["knowledge_attention_mask"] = knowledge_attention_mask

    return model_kwargs

  # Final
  def prepare_inputs_for_generation(self, input_ids, past=None, use_cache=None, attention_mask=None, encoder_outputs=None, knowledge_encoder_outputs=None, knowledge_attention_mask=None,  **kwargs):
    decoder_inputs = self._prepare_decoder_inputs_for_generation(input_ids)
    decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    input_dict = {
        "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "knowledge_attention_mask": knowledge_attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_input_ids": decoder_inputs["input_ids"],
        "encoder_outputs": encoder_outputs,
        "knowledge_encoder_outputs": knowledge_encoder_outputs,
        "use_cache": use_cache,
    }

    # Ideally all models should have a :obj:`use_cache`
    # leave following to ifs until all have it implemented
    if "use_cache" in decoder_inputs:
      input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

    if "past_key_values" in decoder_inputs:
      input_dict["past_key_values"] = decoder_inputs["past_key_values"]

    return input_dict

  def adjust_logits_during_generation(self, logits, cur_len, max_length):
    if cur_len == 1 and self.config.force_bos_token_to_be_generated:
      self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
    elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
      self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
    return logits

  @staticmethod
  def _force_token_id_to_be_generated(scores, token_id) -> None:
    """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
    scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

  @staticmethod
  def _reorder_cache(past, beam_idx):
    reordered_past = []
    for layer_past in past:
      # get the correct batch idx from decoder layer's batch dim for cross and self-attn
      layer_past_new = {
          attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
      }
      reordered_past.append(layer_past_new)
    return reordered_past

  def get_encoder(self):
    return self.model.encoder

  def get_output_embeddings(self):
    return _make_linear_from_emb(self.model.shared)  # make it on the fly

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
    # TODO: It is not an elegant solution.
    if kwargs["args"].share_encoder:
      model.model.knowledge_encoder = model.model.encoder
    else:
      model.model.knowledge_encoder.load_state_dict(
          model.model.encoder.state_dict()
      )  # Copy pre-trained weight to knowledge_encoder.
      model.model.knowledge_encoder.embed_tokens = model.model.encoder.embed_tokens  # Share embeddings.
    return model
