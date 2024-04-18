from transformers import pipeline

# 明确指定模型名称和修订版本
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
model_revision = "af0f99b"  # 这是模型的具体修订哈希，确保使用特定版本

# 创建情感分析管道
classifier = pipeline('sentiment-analysis', model=model_name, revision=model_revision)

# 进行预测
result = classifier("What is love? Baby don't hurt me")[0]
print(f"label: {result['label']}, with score: {result['score']:.4f}")
