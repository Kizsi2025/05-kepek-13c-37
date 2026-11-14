# Bolyai Labs – 5. blokk: A gép lát (CNN és objektumdetektálás)

## Feladatkártya #5

> Minőségellenőrként szeretném, ha a futószalag kamerája magától felismerné az alkatrészeket (kör, négyzet, háromszög, kereszt), és egy kész modell a raktári fotókon az embereket és a járműveket is megtalálná.

| Rész | Feladat |
| --- | --- |
| a) | konvolúció kézzel: `F.conv2d` egy 5 × 5-ös képen |
| b) | alapvonal: MLP a képpontokon (kész kód) |
| c) | konvolúciós háló (CNN) felépítése |
| d) | a CNN tanítása – **képosztályozási teszt:** legalább 93% |
| e) | objektumdetektálás előtanított YOLOv8 modellel |

- [ ] saját repó ebből a sablonból (pl. `kepek-13b-07`, Public)
- [ ] a `kepek.ipynb` TODO-jai kitöltve, az önellenőrzés lefut
- [ ] visszamentve Colabból, commitüzenet: `feat: képfelismerés CNN-nel és YOLO-val (feladatkártya #5)`
- [ ] az Actions fülön a **Feladatkártya 5** futás zöld
- [ ] a repó linkje beadva a Teams-feladatba

**Érték:** 5 sprintpont (zöld build 2, határidő 1, commitüzenet 1, kilépőkérdések 1). **Határidő:** a 2. óra vége.

## Az adat

`data/alkatreszek.npz`: 3000 tanító és 600 teszt kép, 28 × 28 képpont, szürkeárnyalatos, négy osztály (kör, négyzet, háromszög, kereszt) véletlen helyen, méretben, elforgatással és zajjal. Mesterségesen generált, a Bolyai Labs futószalagját utánozza.

## Megnyitás Colabban

[colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** fül → felhasználónév → saját repó → `kepek.ipynb`. GPU nem kell. Az e) rész telepíti az `ultralytics` csomagot, és letölti a `yolov8n.pt` modellt (kb. 6 MB).

## Ha piros a build

Nyisd meg a piros futást az Actions fülön; a tesztek nevéből látod, melyik rész (a–e) hibás. Az első futás 3–4 perc, mert a CI telepíti a PyTorch-ot és az ultralytics csomagot.
