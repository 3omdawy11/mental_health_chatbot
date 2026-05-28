from src.emotion_classifier import load_tokenizer, save_tokenizer, get_dataloaders, load_data


train_df, val_df, test_df = load_data()

tokenizer = load_tokenizer()
save_tokenizer(tokenizer)

train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer)

# Inspect one batch to verify shapes
sample_batch = next(iter(train_loader))
print(f"input_ids shape:      {sample_batch['input_ids'].shape}")
print(f"attention_mask shape: {sample_batch['attention_mask'].shape}")
print(f"labels shape:         {sample_batch['label'].shape}")
print(f"Sample labels:        {sample_batch['label'][:8]}")

# The output should be something like:
# input_ids shape:      torch.Size([32, 128])
# attention_mask shape: torch.Size([32, 128])
# labels shape:         torch.Size([32])
# Sample labels:        tensor([0, 1, 2, 3, 4, 5, 6, 7])


