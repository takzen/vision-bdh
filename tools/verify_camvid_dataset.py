import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter

# --- Konfiguracja ---
DATASET_PATH = Path("./data_camvid")  # Ścieżka do folderu CamVid
NUM_CLASSES = 11                       # Liczba klas w Twoim kodzie
IGNORE_INDEX = 11                       # ignore_index w CrossEntropyLoss
# --------------------

def verify_dataset(root_path: Path, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    print("=" * 60)
    print(f"🕵️  Rozpoczynam weryfikację zbioru danych w: {root_path.resolve()}")
    print("=" * 60)

    if not root_path.exists():
        print(f"❌ BŁĄD: Folder zbioru danych nie istnieje pod ścieżką: {root_path.resolve()}")
        return

    splits = ['train', 'val', 'test']
    all_mask_values = Counter()
    total_files_ok = True
    dataset_ok_for_training = True

    for split in splits:
        print(f"\n--- Weryfikacja podzbioru: '{split}' ---")
        images_dir = root_path / split

        # Sprawdzenie dwóch możliwych nazw folderów z maskami
        masks_dir_labels = root_path / f"{split}_labels"
        masks_dir_annot = root_path / f"{split}annot"

        if masks_dir_labels.exists():
            masks_dir = masks_dir_labels
        elif masks_dir_annot.exists():
            masks_dir = masks_dir_annot
        else:
            print(f"⚠️ OSTRZEŻENIE: Brak folderu masek dla '{split}' (szukano {masks_dir_labels} i {masks_dir_annot}). Pomijam ten podzbiór.")
            dataset_ok_for_training = False
            continue

        if not images_dir.exists():
            print(f"⚠️ OSTRZEŻENIE: Brak folderu obrazów '{images_dir}'. Pomijam ten podzbiór.")
            dataset_ok_for_training = False
            continue

        image_files = sorted(list(images_dir.glob('*.png')))
        mask_files = sorted(list(masks_dir.glob('*.png')))

        print(f"  🖼️ Znaleziono obrazów: {len(image_files)}")
        print(f"  🎭 Znaleziono masek: {len(mask_files)}")

        if len(image_files) != len(mask_files):
            print(f"⚠️ Liczba obrazów i masek się nie zgadza!")
            total_files_ok = False
            dataset_ok_for_training = False

        # Analiza masek
        missing_masks = 0
        split_mask_values = Counter()
        pbar = tqdm(image_files, desc=f"  Analizuję maski '{split}'", ncols=80)
        for img_path in pbar:
            # Sprawdzenie dwóch konwencji nazewnictwa
            mask_path_l = masks_dir / f"{img_path.stem}_L.png"
            mask_path_simple = masks_dir / img_path.name
            mask_path = mask_path_l if mask_path_l.exists() else (mask_path_simple if mask_path_simple.exists() else None)

            if mask_path is None:
                missing_masks += 1
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            unique_values = np.unique(mask)
            split_mask_values.update(unique_values)
            all_mask_values.update(unique_values)

        if missing_masks > 0:
            print(f"  ❌ Brak masek dla {missing_masks} obrazów!")
            total_files_ok = False
            dataset_ok_for_training = False
        else:
            print(f"  ✅ Wszystkie obrazy mają odpowiadające maski.")

        if split_mask_values:
            unique_sorted = sorted(split_mask_values.keys())
            print(f"  📊 Unikalne wartości pikseli w '{split}': {unique_sorted}")
            # Sprawdzenie zgodności z NUM_CLASSES
            if max(unique_sorted) > num_classes:
                print(f"⚠️ UWAGA: Wartości pikseli przekraczają NUM_CLASSES={num_classes}. Dataset NIE nadaje się do treningu bez mapowania!")
                dataset_ok_for_training = False
        else:
            print(f"  ⚠️ Nie znaleziono żadnych wartości pikseli w maskach '{split}'.")

    print("\n\n" + "=" * 60)
    print("🏁 Końcowy raport zbioru danych")
    print("=" * 60)

    if total_files_ok:
        print("✅ Struktura plików (obrazy i maski) wygląda poprawnie.")
    else:
        print("❌ Wykryto problemy ze strukturą plików.")

    if all_mask_values:
        unique_all = sorted(all_mask_values.keys())
        print(f"\n📈 Łącznie znaleziono {len(unique_all)} unikalnych wartości (klas) w całym zbiorze:")
        print(f"   {unique_all}")

    if dataset_ok_for_training:
        print("\n✅ Dataset jest gotowy do użycia z Twoim skryptem (NUM_CLASSES=11, ignore_index=11).")
    else:
        print("\n❌ Dataset NIE nadaje się do treningu z Twoim obecnym kodem! Wartości pikseli lub struktura są niezgodne.")

    print("=" * 60)


if __name__ == "__main__":
    if not DATASET_PATH.exists():
        print(f"❌ BŁĄD: Podana ścieżka do zbioru danych nie istnieje: {DATASET_PATH.resolve()}")
    else:
        verify_dataset(DATASET_PATH)
