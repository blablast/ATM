# System Wykrywania Oszustw

## 📋 Spis Treści
- [Opis Projektu](#-opis-projektu)
- [Komponenty Systemu](#-komponenty-systemu)
  - [Notebook (Modele ML)](#-notebook-modele-ml)
  - [Backend (FastAPI)](#-backend-fastapi)
  - [Frontend (Streamlit)](#-frontend-streamlit)
- [Instalacja i Uruchomienie](#-instalacja-i-uruchomienie)
- [Szczegóły Implementacji](#-szczegóły-implementacji)
- [Użytkowanie](#-użytkowanie)

## 🎯 Opis Projektu
System wykrywania oszustw finansowych łączący zaawansowane modele uczenia maszynowego z interaktywnym interfejsem użytkownika. Umożliwia analizę transakcji w czasie rzeczywistym oraz symulację różnych scenariuszy oszustw.

## 🔧 Komponenty Systemu

### 📊 Notebook (Modele ML)
#### Preprocessing Danych
- Obsługa brakujących wartości
  - Kategoryczne: wypełnianie 'Unknown'
  - Numeryczne: mediana dla kwot, średnia dla ID
- Kodowanie zmiennych kategorycznych (one-hot encoding)
- Normalizacja (StandardScaler)
- Balansowanie klas (SMOTE)

#### Modele
1. **Gradient Boosting**
   - Klasyczny model ensemblowy
   - Próg klasyfikacji: 0.5

2. **HistGradientBoosting**
   - Zoptymalizowana wersja GB
   - Efektywna dla dużych zbiorów

3. **Sieć Neuronowa (DNN)**
   - Architektura wielowarstwowa
   - Dropout i normalizacja wsadowa
   - Próg klasyfikacji: 0.425
   - Early stopping i redukcja learning rate

### ⚙️ Backend (FastAPI)
#### API Endpoints
- `/predict` - predykcje w czasie rzeczywistym
  ```json
  {
    "ta": 1000.0,    // Transaction Amount
    "tt": 1,         // Transaction Type
    "tm": 14.5,      // Time (hours)
    "du": 2,         // Device Used
    "lc": 1,         // Location
    "pm": 3,         // Payment Method
    "ui": 1234,      // User ID
    "pf": 0,         // Previous Fraud
    "aa": 30,        // Account Age
    "nt": 5          // Num Transactions 24h
  }
  ```

#### Funkcjonalności
- Automatyczne ładowanie modeli
- Walidacja danych (Pydantic)
- Kolorowe logowanie predykcji
- Obsługa wielu modeli równocześnie

### 🖥️ Frontend (Streamlit)
#### Tryb Manualny
- Interaktywne suwaki i kontrolki
- Natychmiastowa aktualizacja predykcji
- Wizualizacja wyników w formie post-it notes

#### Tryb Symulacji
- Automatyczne generowanie transakcji
- Konfigurowalne opóźnienia (50-1000ms)
- Ciągła aktualizacja wyników

#### Wizualizacja
- Kolorowe oznaczenie predykcji
  - Czerwony: potencjalne oszustwo
  - Zielony: transakcja bezpieczna
- Wyświetlanie metryk i czasów inferencji

## 🚀 Instalacja i Uruchomienie

### Wymagania
- Python 3.8+
- Zależności:
  ```bash
  pip install -r requirements.txt
  ```

Główne zależności:
- **ML/DL**:
  - tensorflow>=2.10.0
  - scikit-learn>=1.0.2
  - pandas>=1.5.0
  - numpy>=1.21.0

- **API & Web**:
  - fastapi>=0.95.0
  - uvicorn>=0.21.0
  - streamlit>=1.22.0
  - requests>=2.28.0

- **Narzędzia**:
  - pydantic>=2.0.0
  - plotly>=5.13.0
  - python-dotenv>=0.21.0
  - colorama>=0.4.6

### Kroki Uruchomienia
1. **Przygotowanie Modeli**
   Uruchom `project.ipynb` w Jupyter Notebook lub Colab, aby przeprowadzić preprocessing i trenować modele.
   
2. **Mapowanie Kategorii**
   ```bash
   python mapping_script.py
   ```

2. **Backend**
   ```bash
   uvicorn server:app --reload --port 8000
   ```

3. **Frontend**
   ```bash
   streamlit run app.py
   ```

## 🔍 Szczegóły Implementacji

### Progi Decyzyjne
```python
FRAUD_THRESHOLD = {
    "Gradient Boosting": 0.5,
    "HistGradientBoosting": 0.5
}
```

### Mapowanie Kategorii
- Przechowywane w `categorical_mappings.csv`
- Automatyczna synchronizacja między komponentami
- Spójne kodowanie zmiennych

### Metryki Wydajności
- Accuracy
- Precision
- Recall (per klasa)
- F1 Score
- ROC AUC

## 💡 Użytkowanie

### Tryb Manualny
1. Wybierz parametry transakcji
2. Obserwuj predykcje w czasie rzeczywistym
3. Analizuj wyniki różnych modeli

### Tryb Symulacji
1. Ustaw zakres opóźnień
2. Obserwuj automatyczne predykcje
3. Monitoruj podejrzane transakcje

### Wskazówki
- Używaj różnych kombinacji parametrów
- Zwracaj uwagę na czasy inferencji
- Porównuj predykcje między modelami
