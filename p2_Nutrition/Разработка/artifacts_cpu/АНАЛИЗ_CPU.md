# Анализ CPU-экспериментов Nutrition

Срез: train 200 / val 50 / test 50, seed=42, 4 эпохи, MobileNetV3-small + DistilBERT (freeze).

| config | val_mae | test_mae | image_size | use_aug | notes |
|--------|---------|----------|------------|---------|-------|
| **C7_noaug_160** | **153.60** | **120.17** | 160 | False | best |
| C4_no_aug | 153.65 | 125.89 | 128 | False | близко |
| C5_size160 | 154.26 | 123.54 | 160 | True | |
| C1_baseline | 155.93 | 125.67 | 128 | True | |
| C2_text_unfreeze | 158.62 | 127.09 | 128 | True | |
| C6_hidden256 | 164.07 | 133.71 | 128 | True | |
| C3_image_unfreeze | 166.71 | 129.75 | 128 | True | хуже на срезе |

## Выводы

1. На малом срезе аугментации не помогают (C4/C7 лучше C1/C5).
2. IMAGE_SIZE=160 даёт выигрыш по test при отключённых aug.
3. Разморозка энкодеров на 4 эпохах / 200 примерах ухудшает val.
4. Для Colab full берём: size=160, no_aug на старте (или слабые aug на полном объёме), HIDDEN=128; на GPU — EfficientNet-B0 + unfreeze.

## Переход к служебному Colab

Файл: `Nutrition_full_Colab.ipynb`  
Цель продукта: MAE < 50 на полном test.
