from pathlib import Path
import json
import sqlite3
import pandas as pd
import pickle


def parse_isotope_file(file_path):
    isotopes = {}
    with open(file_path, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('%'):
                continue
            parts = stripped_line.split()
            if len(parts) != 9:
                print(f"Skipping malformed line: {stripped_line}")
                continue
            try:
                protons = int(parts[0])
                nucleons = int(parts[1])
                stability = parts[2]
                symbol = parts[3]
                name = parts[4]
                spin = float(parts[5])
                gn = float(parts[6])
                abundance = float(parts[7])
                quadrupole_str = parts[8].lower()
                if quadrupole_str == 'nan':
                    quadrupole = float('nan')
                else:
                    quadrupole = float(quadrupole_str)
                key = f"{nucleons}{symbol}"
                isotopes[key] = {
                    'protons': protons,
                    'nucleons': nucleons,
                    'stability': stability,
                    'symbol': symbol,
                    'name': name,
                    'spin': spin,
                    'gn': gn,
                    'abundance': abundance,
                    'quadrupole': quadrupole
                }
            except ValueError as e:
                print(f"Error parsing line '{stripped_line}': {e}")
    return isotopes


def save_as_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def save_as_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_as_sqlite(data, db_file):
    conn = sqlite3.connect(db_file)
    df = pd.DataFrame(data)
    df.to_sql('nuclei', conn, if_exists='replace', index=False)
    conn.close()


def load_from_sqlite(db_file, query):
    conn = sqlite3.connect(db_file)
    result = pd.read_sql_query(query, conn)
    conn.close()
    return result


def create_files():
    file_path = Path("./nuclei_db/isotopedata.txt")
    nuclei_data = parse_isotope_file(file_path)
    save_as_json(nuclei_data, "./nuclei_db/nuclear_data.json")
    save_as_sqlite(nuclei_data, "./nuclei_db/nuclear_data.db")
    save_as_pickle(nuclei_data, "./nuclei_db/nuclear_data.pkl")
