# Thu du lieu train lai model nga

Chay collector rieng, khong anh huong workflow server:

```powershell
python collect_kfall_style_data.py
```

Mac dinh du lieu luu vao:

```text
storage/collection/kfall_style/
  labels.csv
  sensor_data/
    fall/
      fall_forward/
        S01/
          S01T01R01.csv
          S01T01R02.csv
    not_fall/
      sit_down_fast/
        S01/
          S01T22R01.csv
          S01T22R02.csv
```

`labels.csv` la file tong hop metadata cua moi trial. Moi file CSV trong `sensor_data/` la mot lan thuc hien thao tac.

## Cach test

1. Chay ESP32 nhu binh thuong de publish `sensor/fall_raw`.
2. Chay `python collect_kfall_style_data.py`.
3. Bam `Connect MQTT`.
4. Kiem tra `Storage root` dang la `storage\collection\kfall_style`.
5. Chon dung task tren GUI.
6. Thuc hien thao tac.
7. Khi collector hien `SAVED`, no da luu dung 1 file CSV va tu khoa lai.
8. Chon task tiep theo roi moi test tiep.

Neu ESP32 gui them `sensor/fall_raw` khi chua chon task, collector se bo qua va khong luu.

## Task

Train van chi co 2 nhan chinh:

```text
fall
not_fall
```

Task chi dung de tach folder va phan tich loi:

```text
T01 fall      fall_forward
T02 fall      fall_backward
T03 fall      fall_left
T04 fall      fall_right
T05 fall      fall_from_chair
T06 fall      fall_from_standing
T07 fall      fall_from_bed
T08 fall      fall_other

T21 not_fall  walk
T22 not_fall  sit_down_fast
T23 not_fall  stand_up_fast
T24 not_fall  lie_down_fast
T25 not_fall  pick_object
T26 not_fall  run
T27 not_fall  jump
T28 not_fall  stairs_up_down
T29 not_fall  device_bump
T30 not_fall  device_drop
T31 not_fall  turn_body_fast
T32 not_fall  not_fall_other
```

## Luu y ve ghi de

Collector dung che do tao file moi va khong ghi de. Moi lan cung mot task se tang so repeat:

```text
S01T22R01.csv
S01T22R02.csv
S01T22R03.csv
```

Neu ban da chay ban collector cu luu vao folder `data/`, folder do khong con duoc dung nua. Dataset chuan nam trong `storage/collection/kfall_style/`.
