# Предсказание калорийности блюд

Фото + список ингредиентов → калории (ккал). Цель продукта: **MAE &lt; 50** на test.

На CPU-поиске (замороженные энкодеры, C7, 160px, без aug) получено **test MAE ≈ 120**. Полный прогон (EfficientNet-B0 + DistilBERT, `BATCH_SIZE=4`) нужно выполнить на GPU ВМ курса; цифру test MAE после прогона внесите в вывод этапа 4.

## Содержимое (структура ТЗ)

```text
.
├── data/                         # dish.csv, ingredients.csv, images/<dish_id>/rgb.png
├── nutrition.zip                 # архив датасета из ТЗ (опционально, не в git)
├── scripts/
│   ├── dataset.py                # Dataset, transforms, DataLoader
│   ├── utils.py                  # Config, модель, train(cfg), seed
│   ├── config_full.py            # конфиг полного обучения
│   └── bootstrap.py              # поиск / скачивание data/
├── avo_m25_555_nutrition.ipynb   # EDA + train(cfg) + валидация
├── requirements.txt
└── README.md
```

## Запуск

Откройте `avo_m25_555_nutrition.ipynb` из этой папки и выполните сверху вниз.

Обучение: `result = train(cfg)` из `scripts/utils.py`, конфиг — `get_full_config` в `scripts/config_full.py`.

Данные, по порядку: `data/` рядом с ноутбуком → `nutrition.zip` → [Яндекс.Диск](https://disk.yandex.ru/d/kz9g5msVqtahiw) → диалог папки или Upload (Colab).
