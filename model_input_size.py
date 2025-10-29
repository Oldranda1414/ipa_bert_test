from transformers import AutoModelForSequenceClassification

ipa_model_name = {"pretrained_model_name_or_path":'phonemetransformers/ipa-childes-models-tiny', "subfolder":'EnglishUK'}
model = AutoModelForSequenceClassification.from_pretrained(**ipa_model_name, num_labels=2)
print(model.config.max_position_embeddings)


