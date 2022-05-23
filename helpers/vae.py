import torch
from ignite.utils import convert_tensor

train_metrics = ["loss", "lm", "bow", "kl"]


def trainer_update(engine, batch, args, model, optimizer):
  """
    engine.state.epoch: start from 1
    engine.state.iteration: continue after epochs, not set back 0
    engine.state.epoch_length
  """
  model.train()
  batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
  input_ids, input_mask, knowledge_input_ids, knowledge_input_mask, decoder_input_ids, lm_labels, *_ = batch
  outputs = model(
    input_ids=input_ids, 
    attention_mask=input_mask, 
    decoder_input_ids=decoder_input_ids, 
    knowledge_input_ids=knowledge_input_ids,
    knowledge_attention_mask=knowledge_input_mask,
    labels=lm_labels
  )

  lm_loss, bow_loss, kl_loss = outputs.loss, outputs.bow_loss, outputs.kl_loss
  loss = lm_loss + bow_loss + kl_loss

  return_output = (loss.item(), lm_loss.item(), bow_loss.item(), kl_loss.item())
  if args.n_gpu > 1:
    loss = loss.mean()
  if args.gradient_accumulation_steps > 1:
    loss = loss / args.gradient_accumulation_steps

  loss.backward()
  if args.max_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
  if engine.state.iteration % args.gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
  return return_output


def evaluator_update(engine, batch, args, model):
  model.eval()
  with torch.no_grad():
    batch = tuple(convert_tensor(input_tensor, device=args.device, non_blocking=True) for input_tensor in batch)
    input_ids, input_mask, knowledge_input_ids, knowledge_input_mask, decoder_input_ids, lm_labels, *_ = batch
    outputs = model(
      input_ids=input_ids, 
      attention_mask=input_mask, 
      decoder_input_ids=decoder_input_ids, 
      knowledge_input_ids=knowledge_input_ids,
      knowledge_attention_mask=knowledge_input_mask,
      use_cache=False, # To make sure the logits of whole sentence return.
    )
    lm_logits = outputs.logits

    lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
    lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

  return (lm_logits_flat_shifted, ), (lm_labels_flat_shifted, )


def greedy_sample(batch, args, model, dataset):
  bos_id = args._tokenizer.bos_token_id
  eos_id = args._tokenizer.eos_token_id
  pad_id = args._tokenizer.pad_token_id

  num_beams = args.beam_size if args.beam_search else 1

  input_ids, input_mask, knowledge_input_ids, knowledge_input_mask = tuple(convert_tensor(input_tensor, device=args.device, non_blocking=True) for input_tensor in batch[:4])
  dialog_id, history, response_text = batch[4]
  
  current_outputs = model.generate(
      input_ids=input_ids,
      attention_mask=input_mask,
      knowledge_input_ids=knowledge_input_ids,
      knowledge_attention_mask=knowledge_input_mask,
      max_length=args.max_length,
      min_length=args.min_length,
      do_sample=not args.no_sample,
      num_beams=num_beams,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      # repetition_penalty=1.2,
      # bad_words_ids: Optional[Iterable[int]] = None,
      bos_token_id=bos_id,
      pad_token_id=pad_id,
      eos_token_id=eos_id,
      # length_penalty: Optional[float] = None,
      # no_repeat_ngram_size=1,
      num_return_sequences=1,
      decoder_start_token_id=bos_id,  # To ensure start with bos
      use_cache=False,  # Now the code does not yet support generate using cache, change this to avoid caching
  )
  tg_ids = knowledge_input_ids.gather(dim=1, index=model.knowledge_index[:, None, None].expand(knowledge_input_ids.size(0), 1, knowledge_input_ids.size(-1))).squeeze(1)
  return current_outputs, response_text, dialog_id, history, tg_ids
