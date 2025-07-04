import pandas as pd
import numpy as np
import uuid
import json
import random
from datetime import datetime, timedelta
from faker import Faker
from tqdm import tqdm
from pathlib import Path

from data_generator.config import Config, DATA_DIR, RESULTS_DIR

#иниц Faker
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


class DataGenerator:
    def __init__(self):
        self.plans = self._generate_plans()
        self.psx_attrs = self._generate_psx_attrs()

    def _generate_plans(self):
        return [
            {"Id": 0, "Name": "Start", "Description": "Minimal plan", "CreatedAt": 1606089600000, "Enabled": True,
             "Attrs": "I,1000,24"},
            {"Id": 1, "Name": "Basic", "Description": "Basic plan", "CreatedAt": 1609459200000, "Enabled": True,
             "Attrs": "I,5000,24"},
            {"Id": 2, "Name": "Advanced", "Description": "Advanced plan", "CreatedAt": 1612137600000, "Enabled": True,
             "Attrs": "I,10000,24"},
            {"Id": 3, "Name": "Premium", "Description": "Premium plan", "CreatedAt": 1614556800000, "Enabled": True,
             "Attrs": "U,0,24"},
            {"Id": 4, "Name": "Business", "Description": "Business plan", "CreatedAt": 1617235200000, "Enabled": True,
             "Attrs": "I,20000,24"},
            {"Id": 5, "Name": "Enterprise", "Description": "Enterprise plan", "CreatedAt": 1619827200000,
             "Enabled": True, "Attrs": "U,0,24"}
        ]

    def _generate_psx_attrs(self):
        return [
            {"Id": 0, "PSX": "66.1", "TransmitUnits": "bits", "Delimiter": "|", "DateFormat": Config.DATE_FORMATS[0],
             "TZ": "GMT-5"},
            {"Id": 1, "PSX": "77.2", "TransmitUnits": "bytes", "Delimiter": ",", "DateFormat": Config.DATE_FORMATS[1],
             "TZ": "GMT+3"},
            {"Id": 2, "PSX": "88.3", "TransmitUnits": "bits", "Delimiter": ",", "DateFormat": Config.DATE_FORMATS[2],
             "TZ": "GMT+1"},
            {"Id": 3, "PSX": "99.4", "TransmitUnits": "bytes", "Delimiter": "|", "DateFormat": Config.DATE_FORMATS[3],
             "TZ": "GMT-8"}
        ]

    def generate_all_data(self, dataset_size, dataset_type='small'):
        print(f"\nГенерация {dataset_type} набора данных...")
        output_dir = DATA_DIR / dataset_type

        # Генерация клиентов
        clients, physicals, companies = self._generate_clients(dataset_size)

        # Сохранение данных
        self._save_data(clients, output_dir / 'client.parquet')
        self._save_data(physicals, output_dir / 'physical.parquet')
        self._save_data(companies, output_dir / 'company.parquet')

        # Тарифные планы и атрибуты PSX
        with open(output_dir / 'plan.json', 'w') as f:
            json.dump(self.plans, f, indent=2)

        pd.DataFrame(self.psx_attrs).to_csv(output_dir / 'psx_attrs.csv', index=False)

        #подписчики
        subscribers = self._generate_subscribers(clients)
        pd.DataFrame(subscribers).to_csv(output_dir / 'subscriber.csv', index=False)

        #ген статы
        psx_stats = self._generate_psx_stats(subscribers, dataset_type)
        self._save_psx_stats(psx_stats, output_dir)

        #поиск аномалий
        anomalies = self._detect_anomalies(clients, subscribers, psx_stats)
        with open(RESULTS_DIR / f'result_{dataset_type}.txt', 'w') as f:
            for anomaly in anomalies:
                f.write(f"{anomaly}\n")

        print(f"Генерация {dataset_type} набора завершена!")
        return anomalies

    def _generate_clients(self, num_clients):
        clients, physicals, companies = [], [], []
        num_companies = int(num_clients * Config.COMPANY_RATIO)

        for i in tqdm(range(num_clients), desc="Генерация клиентов"):
            is_company = i < num_companies
            client_id = str(uuid.uuid4())
            contract = f"GB{random.randint(10, 99)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000, 9999)}"

            if is_company:
                #юрлица
                clients.append({
                    "Id": client_id,
                    "Contract": contract,
                    "Documents": f"internal.store.com/clients/documents/{contract}",
                    "Email": f"info@{fake.domain_word()}.com",
                    "IdPlan": random.choice([4, 5])
                })

                companies.append({
                    "Id": client_id,
                    "Name": fake.company(),
                    "Address": fake.address().replace('\n', ', '),
                    "Phones": json.dumps([fake.phone_number() for _ in range(random.randint(1, 3))]),
                    "Contact": json.dumps([fake.name() for _ in range(random.randint(1, 2))])
                })
            else:
                #физлица
                clients.append({
                    "Id": client_id,
                    "Contract": contract,
                    "Documents": f"internal.store.com/clients/documents/{contract}",
                    "Email": fake.email(),
                    "IdPlan": random.choice([0, 1, 2, 3])
                })

                physicals.append({
                    "Id": client_id,
                    "Name": fake.name(),
                    "Address": fake.address().replace('\n', ', '),
                    "Passport": self._generate_passport(),
                    "Phones": json.dumps([fake.phone_number() for _ in range(random.randint(1, 2))])
                })

        return clients, physicals, companies

    def _generate_passport(self):
        return (f"{random.choice(['M', 'F'])} "
                f"{fake.date_of_birth(minimum_age=18, maximum_age=80).strftime('%Y-%m-%d')}, "
                f"{fake.date_between(start_date='-2y').strftime('%Y-%m-%d')}, "
                f"{fake.date_between(start_date='today', end_date='+2y').strftime('%Y-%m-%d')} "
                f"{random.choice(['A', 'B', 'C', 'D'])}{random.randint(1000, 9999)}")

    def _generate_subscribers(self, clients):

        available_psx_ids = random.sample(
            range(*Config.PSX_RANGE),
            len(clients)
        )

        data = np.zeros(len(clients), dtype=[
            ('IdClient', 'U36'),
            ('IdOnPSX', 'i4'),
            ('Status', 'U3')
        ])

        for i in tqdm(range(len(clients)), desc="Генерация подписчиков"):
            data[i] = (
                clients[i]["Id"],
                available_psx_ids[i],
                "ON" if random.random() > 0.05 else "OFF"
            )

        return pd.DataFrame(data)

    def _generate_psx_stats(self, subscribers, dataset_type):
        stats = []
        num_sessions_multiplier = 5 if dataset_type == 'small' else 10

        # Преобразуем subscribers в DataFrame, если это еще не сделано
        if not isinstance(subscribers, pd.DataFrame):
            subscribers = pd.DataFrame(subscribers)

        for _, sub in tqdm(subscribers.iterrows(), total=len(subscribers), desc="Генерация статистики"):
            if sub["Status"] == "OFF":
                continue

            num_sessions = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0] * num_sessions_multiplier

            for _ in range(num_sessions):
                psx_id = random.choice([0, 1, 2, 3])
                psx_config = self.psx_attrs[psx_id]

                start_date = fake.date_time_between(start_date="-1y", end_date="now")
                duration = random.randint(*Config.SESSION_DURATION)

                if random.random() < 0.8:
                    end_date = start_date + timedelta(seconds=duration)
                    end_session = end_date.strftime(psx_config["DateFormat"])
                else:
                    end_session = ""

                # Генерация трафика
                if psx_config["TransmitUnits"] == "bits":
                    base = 10_000_000_000 if random.random() < Config.ANOMALY_RATIO else 100_000_000
                    up_tx = random.randint(base, base * 100)
                    down_tx = random.randint(base, base * 100)
                else:
                    base = 1_000_000_000 if random.random() < Config.ANOMALY_RATIO else 10_000_000
                    up_tx = random.randint(base, base * 100)
                    down_tx = random.randint(base, base * 100)

                stats.append({
                    "IdSession": random.randint(1000, 999999),
                    "IdPSX": psx_id,
                    "IdSubscriber": sub["IdOnPSX"],
                    "StartSession": start_date.strftime(psx_config["DateFormat"]),
                    "EndSession": end_session,
                    "Duration": duration,
                    "UpTx": up_tx,
                    "DownTx": down_tx
                })

        return stats

    def _save_data(self, data, path):
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def _save_psx_stats(self, stats, output_dir):
        chunks = np.array_split(stats, 4)  # Разделяем на 4 части

        for i, chunk in enumerate(chunks, 1):
            df = pd.DataFrame(chunk)
            df.to_csv(output_dir / f'psx_stats_{i}.csv', index=False)
            df.to_csv(output_dir / f'psx_stats_{i}.txt', sep='|', index=False)

    def _detect_anomalies(self, clients, subscribers, psx_stats):
        print("Обнаружение аномалий...")

        # Конвертируем в pandas DataFrame
        df_subscribers = pd.DataFrame(subscribers)
        df_stats = pd.DataFrame(psx_stats)

        # Оптимизация 1: Работаем с chunk'ами
        chunk_size = 100_000
        anomalies = []

        for i in tqdm(range(0, len(df_stats), chunk_size),
                      desc="Анализ трафика"):
            chunk = df_stats.iloc[i:i + chunk_size]

            # Агрегируем трафик по подписчикам в chunk'е
            traffic_chunk = chunk.groupby('IdSubscriber').agg({
                'UpTx': 'sum',
                'DownTx': 'sum'
            })
            traffic_chunk['TotalTraffic'] = traffic_chunk['UpTx'] + traffic_chunk['DownTx']

            # Вычисляем статистики по всему набору (1 раз)
            if i == 0:
                mean = traffic_chunk['TotalTraffic'].mean()
                std = traffic_chunk['TotalTraffic'].std()
                threshold = mean + 5 * std

            # Находим аномалии в chunk'е
            chunk_anomalies = traffic_chunk[traffic_chunk['TotalTraffic'] > threshold].index.tolist()
            anomalies.extend(chunk_anomalies)

        # Убираем дубликаты (если подписчик попал в несколько chunk'ов)
        anomalies = list(set(anomalies))

        # Сопоставляем с клиентами
        anomaly_clients = df_subscribers[df_subscribers['IdOnPSX'].isin(anomalies)]['IdClient'].tolist()

        return anomaly_clients


def main():
    generator = DataGenerator()

    # Генерация малого набора
    generator.generate_all_data(Config.SMALL_DATASET_SIZE, 'small')

    # Генерация большого набора
    generator.generate_all_data(Config.BIG_DATASET_SIZE, 'big')


if __name__ == "__main__":
    main()