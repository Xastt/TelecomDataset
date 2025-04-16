import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from tqdm import tqdm


def detect_anomalies(data_dir):
    try:
        #грузим данные
        clients = pd.read_parquet(Path(data_dir) / 'client.parquet')
        subscribers = pd.read_csv(Path(data_dir) / 'subscriber.csv')

        #грузим файлы статистики
        stats_files = list(Path(data_dir).glob('psx_stats_*.*'))
        stats_dfs = []

        for file in tqdm(stats_files, desc="Загрузка файлов статистики"):
            if file.suffix == '.csv':
                df = pd.read_csv(file)
            elif file.suffix == '.txt':
                df = pd.read_csv(file, sep='|')
            stats_dfs.append(df)

        if not stats_dfs:
            raise ValueError("Не найдены файлы статистики")

        stats = pd.concat(stats_dfs)

        #анализируем трафик
        stats['TotalTraffic'] = stats['UpTx'] + stats['DownTx']
        traffic_stats = stats.groupby('IdSubscriber').agg({
            'TotalTraffic': ['sum', 'mean', 'count']
        })
        traffic_stats.columns = ['TotalTraffic', 'AvgTraffic', 'Sessions']

        #вычисляем порог аномалии
        mean = traffic_stats['TotalTraffic'].mean()
        std = traffic_stats['TotalTraffic'].std()
        threshold = mean + 5 * std

        #ищем аномалии
        anomalies = traffic_stats[traffic_stats['TotalTraffic'] > threshold].index.tolist()
        anomaly_clients = subscribers[subscribers['IdOnPSX'].isin(anomalies)]['IdClient'].tolist()

        return anomaly_clients

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return []


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python checker.py <путь_к_данным>")
        print("Пример: python checker.py data/small")
        sys.exit(1)

    data_path = sys.argv[1]
    anomalies = detect_anomalies(data_path)

    print("\nОбнаружены аномальные клиенты:")
    for client_id in anomalies:
        print(client_id)