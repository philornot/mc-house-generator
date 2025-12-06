"""
Generator domów w Minecraft - TWÓJ KOD
Ten plik rozumiesz i edytujesz!
"""

import numpy as np
import json
from pathlib import Path

# =============================================================================
# DEFINICJE TYPÓW BLOKÓW
# =============================================================================

# Typy bloków - możesz dodawać własne
BLOCK_TYPES = {
    'AIR': 0,
    'BLOCK': 1,
    'STAIR': 2,
}

# Rotacje dla schodów
ROTATIONS = {
    'NORTH': 0,
    'EAST': 1,
    'SOUTH': 2,
    'WEST': 3,
}


# =============================================================================
# GŁÓWNA FUNKCJA GENERUJĄCA DOM
# =============================================================================

def generate_house(width, height, depth, options=None):
    """
    Generuje dom używając numpy array.

    Args:
        width: Szerokość (X)
        height: Wysokość ścian (Y)
        depth: Głębokość (Z)
        options: Dodatkowe opcje (okna, kolumny, etc.)

    Returns:
        tuple: (house, rotation) - dwie tablice numpy
    """

    if options is None:
        options = {
            'add_windows': True,
            'add_columns': True,
            'add_stairs': True,
            'roof_type': 'flat'
        }

    # Tworzymy tablice z zapasem na zewnętrzne elementy
    # +2 dla kolumn z boków, +5 dla dachu na górze
    house = np.zeros((width + 2, height + 5, depth + 2), dtype=int)
    rotation = np.zeros((width + 2, height + 5, depth + 2), dtype=int)

    # Offset - przesunięcie żeby główny dom był w środku
    ox, oz = 1, 1  # offset X i Z

    # === PODŁOGA ===
    house[ox:ox + width, 0, oz:oz + depth] = BLOCK_TYPES['BLOCK']

    # === ŚCIANY ===
    # Przednia ściana (Z=0 w lokalnych koordynatach)
    house[ox:ox + width, 1:height + 1, oz] = BLOCK_TYPES['BLOCK']

    # Tylna ściana
    house[ox:ox + width, 1:height + 1, oz + depth - 1] = BLOCK_TYPES['BLOCK']

    # Lewa ściana (bez rogów - te są już w przedniej/tylnej)
    house[ox, 1:height + 1, oz + 1:oz + depth - 1] = BLOCK_TYPES['BLOCK']

    # Prawa ściana (bez rogów)
    house[ox + width - 1, 1:height + 1, oz + 1:oz + depth - 1] = BLOCK_TYPES['BLOCK']

    # === DRZWI (usuń bloki ze ściany) ===
    door_x = ox + width // 2
    house[door_x, 1, oz] = BLOCK_TYPES['AIR']
    house[door_x, 2, oz] = BLOCK_TYPES['AIR']

    # === OKNA ===
    if options['add_windows']:
        window_y = 1 + height // 2
        # Przednia ściana - co drugi blok
        for x in range(ox + 2, ox + width - 2, 2):
            house[x, window_y, oz] = BLOCK_TYPES['BLOCK']

        # Boczne ściany
        for z in range(oz + 2, oz + depth - 2, 2):
            house[ox, window_y, z] = BLOCK_TYPES['BLOCK']
            house[ox + width - 1, window_y, z] = BLOCK_TYPES['BLOCK']

    # === KOLUMNY W ROGACH ===
    if options['add_columns']:
        for y in range(1, height + 1):
            column_block = BLOCK_TYPES['BLOCK']
            house[ox - 1, y, oz - 1] = column_block
            house[ox + width, y, oz - 1] = column_block
            house[ox - 1, y, oz + depth] = column_block
            house[ox + width, y, oz + depth] = column_block

    # === SCHODY PRZED DRZWIAMI ===
    if options['add_stairs']:
        # Dwa schody przed drzwiami, skierowane na południe (w stronę drzwi)
        house[door_x, 0, oz - 1] = BLOCK_TYPES['STAIR']
        rotation[door_x, 0, oz - 1] = ROTATIONS['SOUTH']

        house[door_x - 1, 0, oz - 1] = BLOCK_TYPES['STAIR']
        rotation[door_x - 1, 0, oz - 1] = ROTATIONS['SOUTH']

    # === DACH ===
    roof_y = height + 1
    if options['roof_type'] == 'flat':
        # Prosty płaski dach
        house[ox:ox + width, roof_y, oz:oz + depth] = BLOCK_TYPES['BLOCK']
    elif options['roof_type'] == 'gable':
        # Dach dwuspadowy - TU DODASZ SWÓJ ALGORYTM!
        generate_gable_roof(house, ox, roof_y, width, depth)

    return house, rotation


# =============================================================================
# ALGORYTMY DACHÓW (do rozbudowy!)
# =============================================================================

def generate_gable_roof(house, ox, start_y, width, depth):
    """
    Prosty dach dwuspadowy (A-frame).
    TU MOŻESZ EKSPERYMENTOWAĆ!

    Args:
        house: Tablica numpy do modyfikacji
        ox: Offset X (gdzie zaczyna się dom)
        start_y: Wysokość startu dachu
        width: Szerokość domu
        depth: Głębokość domu
    """

    # Środek w osi Z
    mid_z = depth // 2

    # Wysokość dachu (połowa głębokości)
    roof_height = depth // 2

    # Budujemy warstwami
    for layer in range(roof_height):
        current_y = start_y + layer

        # Jak daleko od środka jesteśmy w tej warstwie
        z_offset = layer

        # Bloki dachu na tej wysokości
        z_start = mid_z - z_offset
        z_end = mid_z + z_offset + 1

        # Wypełnij całą szerokość na tym poziomie
        for x in range(ox, ox + width):
            for z in range(z_start, z_end):
                if 0 <= z < depth:
                    house[x + ox, current_y, z + ox] = BLOCK_TYPES['BLOCK']


# =============================================================================
# KONWERSJA DO JSON (dla renderera JavaScript)
# =============================================================================

def array_to_json(house, rotation):
    """
    Konwertuje numpy arrays na JSON dla JavaScript.
    TEN KOD MOŻESZ ZIGNOROWAĆ - to tylko konwersja.
    """

    type_names = {
        0: None,  # AIR nie eksportujemy
        1: 'block',
        2: 'stair'
    }

    rotation_names = ['north', 'east', 'south', 'west']

    blocks = []

    # Przejdź przez całą tablicę
    for x in range(house.shape[0]):
        for y in range(house.shape[1]):
            for z in range(house.shape[2]):
                block_type = house[x, y, z]

                # Pomiń powietrze
                if block_type == 0:
                    continue

                # Podstawowe dane bloku
                block_data = {
                    "x": int(x - 1),  # Odejmij offset
                    "y": int(y),
                    "z": int(z - 1),  # Odejmij offset
                    "type": type_names[block_type]
                }

                # Dodaj metadane dla schodów
                if block_type == BLOCK_TYPES['STAIR']:
                    rot = rotation[x, y, z]
                    block_data["metadata"] = {
                        "facing": rotation_names[rot],
                        "upsideDown": False
                    }

                blocks.append(block_data)

    return {
        "blocks": blocks,
        "dimensions": {
            "width": int(house.shape[0] - 2),
            "height": int(house.shape[1] - 5),
            "depth": int(house.shape[2] - 2)
        }
    }


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def visualize_layer(house, y_level, title=""):
    """
    Wyświetl warstwę poziomą w konsoli.
    SUPER PRZYDATNE DO DEBUGOWANIA!
    """

    symbols = {
        0: '·',  # AIR
        1: '█',  # BLOCK
        2: '≡',  # STAIR
    }

    if title:
        print(f"\n{title}")
    print(f"Warstwa Y={y_level}")
    print("  " + "".join(str(x % 10) for x in range(house.shape[0])) + " (X)")

    for z in range(house.shape[2]):
        line = f"{z:2d} "
        for x in range(house.shape[0]):
            block = house[x, y_level, z]
            line += symbols.get(block, '?')
        print(line)
    print("   " + " " * house.shape[0] + "(Z)")


def print_stats(house):
    """Wyświetl statystyki bloków"""

    unique, counts = np.unique(house, return_counts=True)

    type_names = {
        0: 'AIR',
        1: 'BLOCK',
        2: 'STAIR'
    }

    print("\nStatystyki bloków:")
    for block_type, count in zip(unique, counts):
        if block_type == 0:
            continue
        name = type_names.get(block_type, 'UNKNOWN')
        print(f"  {name:10s}: {count:4d}")


# =============================================================================
# MAIN - uruchom to żeby wygenerować dom!
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATOR DOMÓW MINECRAFT")
    print("=" * 60)

    # PARAMETRY - ZMIEŃ TO!
    WIDTH = 8
    HEIGHT = 4
    DEPTH = 6

    OPTIONS = {
        'add_windows': True,
        'add_columns': True,
        'add_stairs': True,
        'roof_type': 'flat'  # 'flat' lub 'gable'
    }

    print(f"\nGeneruję dom {WIDTH}×{HEIGHT}×{DEPTH}...")

    # GENERUJ DOM
    house, rotation = generate_house(WIDTH, HEIGHT, DEPTH, OPTIONS)

    # WIZUALIZUJ W KONSOLI
    print("\n" + "=" * 60)
    print("WIZUALIZACJA")
    print("=" * 60)

    visualize_layer(house, 0, "Podłoga (Y=0)")
    visualize_layer(house, HEIGHT // 2, f"Środkowa wysokość (Y={HEIGHT // 2})")
    visualize_layer(house, HEIGHT + 1, f"Dach (Y={HEIGHT + 1})")

    # STATYSTYKI
    print_stats(house)

    # ZAPISZ DO JSON
    print("\n" + "=" * 60)
    print("EKSPORT DO JSON")
    print("=" * 60)

    json_data = array_to_json(house, rotation)

    # Utwórz folder output jeśli nie istnieje
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'house.json'
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✓ Zapisano {len(json_data['blocks'])} bloków")
    print(f"✓ Plik: {output_file}")
    print("\n" + "=" * 60)
    print("GOTOWE!")
    print("=" * 60)
    print("\nTeraz:")
    print("1. Otwórz frontend/index.html w przeglądarce")
    print("2. Kliknij 'Load House' i wybierz output/house.json")
    print("3. Zobacz swój dom w 3D!")
    print("\nLUB po prostu otwórz index.html - automatycznie załaduje house.json")