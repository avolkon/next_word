# Предсказание калорийности блюд

Фото + список ингредиентов → калории (ккал). Цель: **MAE &lt; 50** на test.

## Содержимое

```text
.
├── nutrition/                    # dish.csv, ingredients.csv, images/
├── nutrition.zip                 # опционально (вместо/вместе с папкой)
├── avo_m25_555_nutrition.ipynb
└── README.md
```

## Запуск

Откройте `avo_m25_555_nutrition.ipynb` **из этой папки** и выполните сверху вниз.

Данные: ячейка 0.2 ищет папку **`nutrition/`** или файл **`nutrition.zip`** рядом с ноутбуком;
если нет — диалог выбора папки (локально) или Upload `nutrition.zip` (Colab).
