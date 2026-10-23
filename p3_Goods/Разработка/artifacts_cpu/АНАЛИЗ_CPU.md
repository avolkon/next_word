# Анализ конфигов пайплайна Goods

Аудит: ok=True, missing=0, bad_images=0, splits 840/180/180.

| config | score | size | max_len | batch | strong_aug | imagenet_norm | shapes_ok |
|--------|-------|------|---------|-------|------------|---------------|-----------|
| **G3** | **25** | 224 | 24 | 8 | False | True | True |
| G2 | 25 | 224 | 16 | 8 | False | True | True |
| G5 | 25 | 224 | 16 | 16 | False | True | True |
| G4 | 24 | 224 | 16 | 8 | True | True | True |
| G6 | 24 | 224 | 16 | 8 | False | False | True |
| G1 | 23 | 128 | 16 | 8 | False | True | True |

## Best: G3

- IMAGE_SIZE=224, MAX_TEXT_LEN=24, BATCH=8, weak aug, ImageNet normalize
- Более длинный текст полезен для `productDisplayName`
- Strong aug (G4) не даёт преимущества по score на инженерной проверке

Служебный Colab: `Goods_Colab.ipynb` с этими параметрами.
