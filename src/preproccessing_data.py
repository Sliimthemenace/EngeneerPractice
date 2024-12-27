import pandas as pd
import gdown


def extract(train_path, test_path):
    """
    Загружает данные из файлов train.csv, test.csv
    """
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except Exception:
        raise 'Нет файла'


def transform(train_data, test_data):
    """
    Очищает и преобразует данные:
    - Заполняет пропущенные значения.
    - Кодирует категориальные переменные.
    - Удаляет ненужные столбцы.
    """
    # Заполнение пропущенных значений
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    # Удаление ненужных столбцов
    train_data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    test_data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

    # Кодирование категориальных переменных
    train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'],
                                drop_first=True)
    test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'],
                               drop_first=True)

    # Убедимся, что столбцы совпадают между train и test
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

    return train_data, test_data


def load(train_data, test_data, output_train_path, output_test_path):
    """
    Сохраняет обработанные данные в новые файлы.
    """
    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)
    print(f"Обработанные данные сохранены в {output_train_path} и {output_test_path}")


def etl_pipeline(train_path, test_path,
                 output_train_path, output_test_path):
    """
    Основной ETL-pipeline для обработки данных Титаник.
    """
    # Extract
    print("Загрузка данных...")
    train_data, test_data = extract(train_path, test_path)

    # Transform
    print("Преобразование данных...")
    train_data, test_data = transform(train_data, test_data)

    # Load
    print("Сохранение обработанных данных...")
    load(train_data, test_data, output_train_path, output_test_path)

    print("ETL-pipeline завершен!")


if __name__ == "__main__":
    link = 'https://drive.google.com/drive/folders/1RUZVcnBYC5KEjB0ehhXp8J3UWQzWWyNc?usp=drive_link'
    gdown.download_folder(link)

    train_path = "data/train.csv"
    test_path = "data/test.csv"
    output_train_path = "data/train.csv"
    output_test_path = "data/test.csv"

    etl_pipeline(train_path, test_path,
                 output_train_path, output_test_path)
