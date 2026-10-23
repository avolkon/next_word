# Пайплайн данных для классификации товаров

Аудит → transforms → текст → Dataset → DataLoader.  
Классы: Handbags, Shirts, Sports Shoes, Watches.

## Содержимое

```text
.
├── data/                      # metadata.csv, class_to_idx.csv, images/
├── avo_m25_555_goods.ipynb    # монолитный ноутбук (этапы 1–6 ТЗ)
└── README.md
```

## Запуск

Откройте `avo_m25_555_goods.ipynb` **из этой папки** и выполните сверху вниз.

Данные: ячейка 0.2 сначала ищет `data/` рядом с ноутбуком; если нет — диалог выбора папки (локально) или Upload zip (Colab).

Весь код — внутри ноутбука, без внешних `.py` проекта.
