import pandas as pd

# Путь к файлу с метками классов
class_labels_file = 'data/test/_classes.csv'

# Загрузка и проверка меток классов
class_labels_df = pd.read_csv(class_labels_file)
print("Columns in the dataset:", class_labels_df.columns)
print("First few rows of the dataset:")
print(class_labels_df.head())
