def evaluate_bert_like(model, eval, tokenizer, key):
    def tokenize(batch):
        return tokenizer(batch[key], truncation=True, max_length=128)
    eval_tok = eval.map(tokenize, batched=True, remove_columns=eval.column_names)
    return model.evaluate(eval_dataset=eval_tok)
max_range = 10000
for range_size in [100, 500, 1000, 2000, 3000, 5000]:
    step = int(range_size/2)
    for start in range(0, max_range, step):
        end = start + range_size
        latin_eval_data_subset = latin_eval_data.select(range(start, end))
        metrics = evaluate_bert_like(trainer, latin_eval_data_subset, latin_tokenizer, latin_key)
        loss = metrics["eval_loss"]
        if not math.isnan(loss):
            print(f"  loss = {loss:.4f}")
            print(start,end)
