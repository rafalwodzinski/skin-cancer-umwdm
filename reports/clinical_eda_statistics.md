# Wyniki EDA klinicznej ISIC 2019

- Źródło: [ISIC 2019](https://challenge.isic-archive.com/data/#2019), treningowe metadane i ground truth (licencja CC BY-NC 4.0).
- Rekordy wczytane: **25,331**.
- Rekordy w pięciu analizowanych klasach: **24,211**.
- Klasy: **NV, MEL, BCC, BKL, AK** (wybrane jako pięć najliczniejszych klas docelowych).

## Liczebność klas

| Diagnoza | Liczba | Udział całego zbioru |
|---|---:|---:|
| NV | 12,875 | 50.83% |
| MEL | 4,522 | 17.85% |
| BCC | 3,323 | 13.12% |
| BKL | 2,624 | 10.36% |
| AK | 867 | 3.42% |

## Braki danych

| Zmienna | Braki (%) |
|---|---:|
| diagnosis | 0.00% |
| age_approx | 1.73% |
| sex | 1.52% |
| anatom_site_general | 10.39% |

## Testy chi-kwadrat niezależności

| Cecha | χ² | df | p-value | V Craméra | N | Min. oczekiwana | Oczekiwane < 5 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| anatom_site_general | 5540.8101 | 32 | <1e-300 | 0.239 | 24,211 | 1.934 | 4.4% |
| sex | 159.2497 | 4 | 2.11761e-33 | 0.082 | 23,827 | 406.708 | 0.0% |

## Test Kruskala-Wallisa dla wieku

- H = **6572.4295**, df = **4**, p-value = **<1e-300**, N = **23,774**.
- Wielkość efektu (ε²) = **0.276**.
- Liczba kompletnych obserwacji według klasy: NV: 12,558; MEL: 4,437; BCC: 3,317; BKL: 2,598; AK: 864.

## Interpretacja i ograniczenia

Testy chi-kwadrat oceniają zależność między diagnozą a płcią/lokalizacją; V Craméra i ε² są miarami wielkości obserwowanych zależności/różnic. Dla chi-kwadrat przybliżenie jest zwyczajowo uznawane za wiarygodne, gdy mniej niż 20% komórek ma liczebność oczekiwaną < 5 i żadna nie ma liczebności < 1; raport podaje te wartości, aby umożliwić kontrolę warunku. Wielkości efektu są opisowe, nie dowodzą skuteczności predykcyjnej. Istotność statystyczna nie dowodzi przyczynowości ani wartości predykcyjnej cechy; przy dużym zbiorze nawet niewielkie różnice mogą dawać małe p-value. Przydatność cech w fuzji multimodalnej należy następnie ocenić w walidacji modelu bez przecieku danych.

## Wygenerowane wykresy

- `figures/eda/class_distribution.png`
- `figures/eda/age_distribution.png`
- `figures/eda/site_diagnosis_heatmap.png`
- `figures/eda/missingness.png`
