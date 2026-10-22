import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib
import os
import re
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# 1. УНИВЕРСАЛЬНАЯ ЗАГРУЗКА ДАННЫХ
# ================================================================

def load_data(csv_path):
    encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin-1', 'windows-1251']
    separators = [';', ',', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
                
                if len(df.columns) >= 2:
                    text_col = None
                    class_col = None
                    
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'текст' in col_lower or 'фрагмент' in col_lower:
                            text_col = col
                        if 'класс' in col_lower:
                            class_col = col
                    
                    if text_col is None:
                        text_col = df.columns[0]
                    if class_col is None:
                        class_col = df.columns[1]
                    
                    df = df.rename(columns={text_col: 'текст_фрагмента', class_col: 'класс'})
                    df['класс'] = pd.to_numeric(df['класс'], errors='coerce')
                    df = df.dropna(subset=['текст_фрагмента', 'класс'])
                    df = df[df['класс'].isin([0, 1, 2, 3])]
                    df['класс'] = df['класс'].astype(int)
                    df = df[df['текст_фрагмента'].str.len() > 5]
                    
                    if len(df) > 10:
                        print(f"✅ Прочитано: разделитель '{sep}', кодировка '{encoding}'")
                        print(f"   ✅ Загружено {len(df)} строк")
                        print(f"   📊 Распределение классов: {df['класс'].value_counts().sort_index().to_dict()}")
                        texts = df['текст_фрагмента'].astype(str).tolist()
                        labels = df['класс'].astype(int).tolist()
                        return texts, labels
                        
            except Exception as e:
                continue
    
    raise ValueError("❌ Не удалось прочитать файл ни в одной кодировке/разделителе")

# ================================================================
# 2. АУГМЕНТАЦИЯ ДАННЫХ (СОЗДАНИЕ СИНОНИМОВ)
# ================================================================

def augment_text(text, class_label):
    """Создает вариации текста для аугментации"""
    variations = [text]
    
    # Замены ключевых слов на синонимы
    synonym_map = {
        'коррозия': ['ржавчина', 'коррозионное повреждение', 'ржавление'],
        'трещина': ['трещина', 'разрыв', 'раскрытие', 'щель'],
        'скол': ['откол', 'скол', 'повреждение'],
        'износ': ['изнашивание', 'изношенность', 'деградация'],
        'обрушение': ['обвал', 'разрушение', 'падение'],
        'усиление': ['укрепление', 'армирование', 'восстановление'],
        'ремонт': ['восстановление', 'реставрация', 'исправление'],
        'аварийный': ['опасный', 'критический', 'предаварийный'],
        'работоспособное': ['исправное', 'нормальное', 'штатное'],
        'ограниченно': ['частично', 'неполностью', 'условно'],
    }
    
    # Создаем вариации заменой ключевых слов
    for word, synonyms in synonym_map.items():
        if word in text:
            for syn in synonyms[:1]:  # Берем только первый синоним для разнообразия
                new_text = text.replace(word, syn)
                if new_text != text and len(new_text.split()) > 3:
                    variations.append(new_text)
    
    # Для класса 3 (аварийное) добавляем больше вариаций
    if class_label == 3:
        danger_phrases = [
            'требуется срочное вмешательство',
            'необходимы противоаварийные мероприятия',
            'существует угроза обрушения',
            'эксплуатация недопустима'
        ]
        for phrase in danger_phrases:
            if len(text) < 200:
                new_text = text + '. ' + phrase
                variations.append(new_text)
    
    # Для класса 1 (работоспособное) добавляем подтверждения
    if class_label == 1:
        safe_phrases = [
            'состояние удовлетворительное',
            'дефекты не выявлены',
            'конструкция работает штатно'
        ]
        for phrase in safe_phrases:
            if len(text) < 200 and phrase not in text:
                new_text = text + '. ' + phrase
                variations.append(new_text)
    
    return variations

def augment_dataset(texts, labels, augment_factor=2):
    """Аугментация всего датасета"""
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts, labels):
        # Оригинал
        augmented_texts.append(text)
        augmented_labels.append(label)
        
        # Аугментации
        augs = augment_text(text, label)
        for aug in augs[:augment_factor]:
            if aug != text and len(aug.split()) > 3:
                augmented_texts.append(aug)
                augmented_labels.append(label)
    
    print(f"📊 Аугментация: {len(texts)} → {len(augmented_texts)} примеров")
    return augmented_texts, augmented_labels

# ================================================================
# 3. РАСШИРЕННОЕ ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ================================================================

DANGER_WORDS = [
    'сквозн', 'обруш', 'не обеспечен', 'ктс', 'прогиб', 'превыша', 
    'недопустим', 'исчерпан', 'срез', 'обрыв', 'аварийн', 'критическ',
    'разруш', 'потеря устойчивости', 'опасн', 'критический',
    'несущая способность не', 'превышает допустимый',
    'прогнил', 'насквозь', 'угроза', 'коэффициент использования',
    'предельное состояние', 'недопустимое состояние', 'предаварийн'
]

LIMITED_WORDS = [
    'коррози', 'скол', 'усадочн', 'износ', 'замачиван', 'высол',
    'выветриван', 'недостаточн', 'фрагментаци', 'ремонт', 'усилени',
    'поврежден', 'дефект', 'протечк', 'увлажн', 'каверн', 'пор',
    'некачествен', 'низкое качество', 'отслоени', 'следы', 'ржавч'
]

SAFE_WORDS = [
    'работоспособное', 'удовлетворительное', 'обеспечена',
    'соответствует', 'норме', 'допустимого', 'стабилизировались',
    'запас прочности', 'несущая способность обеспечена',
    'условие выполнено', 'осадки стабилизировались',
    'качество кладки удовлетворительное', 'штатное'
]

PURE_DESC = [
    'сечение', 'шаг', 'высота', 'длина', 'ширина', 'толщина',
    'изготовлен', 'применен', 'выполнены', 'армированы',
    'соответствует', 'по серии', 'класса', 'марки', 'класс бетона',
    'плотность', 'удельный вес', 'габариты', 'рассчитан',
    'замоделирована', 'принята', 'нормативные значения', 'план',
    'схема', 'расположен', 'имеются', 'составляет', 'здание', 'этаж',
    'подполье', 'убежище', 'предусмотрен', 'разрез', 'отметка',
    'уровень', 'мощность', 'глубина', 'обмерные', 'чертежи',
    'конструктивно', 'запроектирована', 'предусмотрено', 'типовое'
]

def extract_enhanced_features(text):
    """Извлекает дополнительные признаки из текста"""
    text_lower = text.lower()
    
    # Считаем количество ключевых слов
    danger_count = sum(1 for w in DANGER_WORDS if w in text_lower)
    limited_count = sum(1 for w in LIMITED_WORDS if w in text_lower)
    safe_count = sum(1 for w in SAFE_WORDS if w in text_lower)
    pure_count = sum(1 for w in PURE_DESC if w in text_lower)
    
    # Числовые признаки
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    mm_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*мм', text_lower)
    percent_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_lower)
    
    has_number = len(numbers) > 0
    has_mm = len(mm_numbers) > 0
    has_percent = len(percent_numbers) > 0
    
    # Максимальные числовые значения
    max_mm = max([float(x) for x in mm_numbers]) if mm_numbers else 0
    max_percent = max([float(x) for x in percent_numbers]) if percent_numbers else 0
    
    # Соотношения
    ratio_danger_limited = danger_count / (limited_count + 1)
    ratio_safe_danger = safe_count / (danger_count + 1)
    ratio_limited_pure = limited_count / (pure_count + 1)
    
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'danger_count': danger_count,
        'limited_count': limited_count,
        'safe_count': safe_count,
        'pure_count': pure_count,
        'has_number': int(has_number),
        'has_mm': int(has_mm),
        'has_percent': int(has_percent),
        'max_mm': max_mm,
        'max_percent': max_percent,
        'ratio_danger_limited': ratio_danger_limited,
        'ratio_safe_danger': ratio_safe_danger,
        'ratio_limited_pure': ratio_limited_pure
    }

def extract_features_batch(texts):
    """Извлекает признаки для списка текстов"""
    features_list = []
    for text in texts:
        feat = extract_enhanced_features(text)
        features_list.append([
            feat['length'],
            feat['word_count'],
            feat['danger_count'],
            feat['limited_count'],
            feat['safe_count'],
            feat['pure_count'],
            feat['has_number'],
            feat['has_mm'],
            feat['has_percent'],
            feat['max_mm'],
            feat['max_percent'],
            feat['ratio_danger_limited'],
            feat['ratio_safe_danger'],
            feat['ratio_limited_pure']
        ])
    return np.array(features_list)

# ================================================================
# 4. УЛУЧШЕННЫЕ ПРАВИЛА
# ================================================================

def apply_rules(text, ml_pred, proba=None):
    """Правила-фильтры с учетом уверенности модели"""
    text_lower = text.lower()
    
    # === ЖЕСТКИЕ ПРАВИЛА ===
    danger_strong = [
        'аварийн', 'обруш', 'сквозн трещин', 'прогнил', 'насквозь',
        'угроза обруш', 'предельное состояние', 'недопустимое состояние',
        'несущая способность не обеспечена', 'несущая способность полностью исчерпана',
        'ктс = 0', 'критический прогиб', 'предаварийн'
    ]
    for word in danger_strong:
        if word in text_lower:
            return 3
    
    working_strong = [
        'работоспособное состояние', 'несущая способность обеспечена',
        'осадки стабилизировались', 'условие выполнено',
        'несущая способность основания обеспечена', 'запас прочности'
    ]
    for word in working_strong:
        if word in text_lower:
            return 1
    
    # === ОТРИЦАНИЯ ===
    if 'не выявл' in text_lower or 'отсутств' in text_lower or 'не обнаруж' in text_lower:
        if 'трещин' in text_lower or 'дефект' in text_lower:
            return 0
        return 0
    
    # === ЧИСЛОВЫЕ ПОРОГИ ===
    numbers = re.findall(r'(\d+(?:\.\d+)?)\s*(мм|%)', text_lower)
    for num, unit in numbers:
        try:
            num = float(num)
            if unit == 'мм' and num > 2.0:
                if 'не выявл' in text_lower or 'отсутств' in text_lower:
                    continue
                return 3
            if unit == '%' and num > 25:
                if 'не выявл' in text_lower or 'отсутств' in text_lower:
                    continue
                return 3
        except:
            pass
    
    # === КОЭФФИЦИЕНТ ИСПОЛЬЗОВАНИЯ ===
    if 'коэффициент использования' in text_lower:
        nums = re.findall(r'0\.\d+', text_lower)
        for n in nums:
            try:
                if float(n) > 0.9:
                    return 3
                if float(n) < 0.8:
                    return 1
            except:
                pass
    
    # === ЕСЛИ МОДЕЛЬ УВЕРЕНА ===
    if proba is not None and max(proba) > 0.85:
        return ml_pred
    
    return ml_pred

# ================================================================
# 5. ПОСТ-ОБРАБОТКА
# ================================================================

def post_process(text, pred_class, proba):
    """Пост-обработка для исправления очевидных ошибок"""
    text_lower = text.lower()
    
    # === ЖЕСТКИЕ ПРАВИЛА ===
    hard_rules = {
        # АВАРИЯ (3)
        'аварийн': 3,
        'обруш': 3,
        'сквозн': 3,
        'прогнил': 3,
        'насквозь': 3,
        'угроза': 3,
        'несущая способность не': 3,
        'ктс = 0': 3,
        'предаварийн': 3,
        'напряжени': 3,
        'превыша': 3,
        'расчетное сопротивление': 3,
        'предельное состояние': 3,
        'недопустимое состояние': 3,
        
        # РАБОТОСПОСОБНОЕ (1)
        'работоспособное': 1,
        'несущая способность обеспечена': 1,
        'осадки стабилизировались': 1,
        'условие выполнено': 1,
        'запас прочности': 1,
        'удовлетворительное': 1,        # <-- ДОБАВЛЕНО!
        
        # ОГРАНИЧЕННО (2)
        'коррози': 2,
        'скол': 2,
        'износ': 2,
        'требуется усил': 2,
        'ремонт': 2,
        'протечк': 2,
        'высол': 2,
        'усадочн': 2,
        'замачиван': 2,
        'выветриван': 2,
        'недостаточн': 2,
        'фрагментаци': 2,
        'отслоени': 2,
        'каверн': 2,
        'пор': 2,
        'ржавч': 2,
        'следы': 2,
    }
    
    # Проверяем жесткие правила
    for key, value in hard_rules.items():
        if key in text_lower:
            return value
    
    # Отрицания (если есть отрицание — это НОРМА)
    if 'не выявл' in text_lower or 'отсутств' in text_lower or 'не обнаруж' in text_lower:
        return 0
    
    # Числовые пороги (трещина > 2 мм → авария)
    mm_nums = re.findall(r'(\d+)\s*мм', text_lower)
    for num in mm_nums:
        if int(num) > 2:
            return 3
    
    # Если модель уверена > 85% — доверяем
    if max(proba) > 0.85:
        return pred_class
    
    return pred_class



# ================================================================
# 6. УЛУЧШЕННОЕ ОБУЧЕНИЕ С АУГМЕНТАЦИЕЙ
# ================================================================

def train_ensemble(texts, labels):
    print("\n" + "="*60)
    print("🔄 ОБУЧЕНИЕ УЛУЧШЕННОГО АНСАМБЛЯ С АУГМЕНТАЦИЕЙ")
    print("="*60)
    
    # АУГМЕНТАЦИЯ
    print("\n🔄 Аугментируем данные...")
    augmented_texts, augmented_labels = augment_dataset(texts, labels, augment_factor=2)
    
    # Разделяем
    X_train, X_test, y_train, y_test = train_test_split(
        augmented_texts, augmented_labels, test_size=0.2, random_state=42, stratify=augmented_labels
    )
    
    print(f"📊 Размер обучающей выборки: {len(X_train)}")
    print(f"📊 Размер тестовой выборки: {len(X_test)}")
    
    # === 1. SBERT ===
    print("\n🔄 Обучаем SBERT...")
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    train_emb = sbert_model.encode(X_train, show_progress_bar=True)
    test_emb = sbert_model.encode(X_test, show_progress_bar=True)
    
    sbert_clf = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight='balanced',
        random_state=42,
        solver='saga'
    )
    sbert_clf = CalibratedClassifierCV(sbert_clf, cv=3, method='isotonic')
    sbert_clf.fit(train_emb, y_train)
    
    sbert_pred = sbert_clf.predict(test_emb)
    sbert_acc = accuracy_score(y_test, sbert_pred)
    print(f"   ✅ SBERT точность: {sbert_acc*100:.2f}%")
    
    # === 2. TF-IDF с LinearSVC ===
    print("\n🔄 Обучаем TF-IDF с LinearSVC...")
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        use_idf=True,
        max_df=0.85
    )
    
    train_tfidf = vectorizer.fit_transform(X_train)
    test_tfidf = vectorizer.transform(X_test)
    
    svc_clf = LinearSVC(
        C=1.5,
        class_weight='balanced',
        random_state=42,
        max_iter=5000,
        dual=False
    )
    svc_clf = CalibratedClassifierCV(svc_clf, cv=3, method='isotonic')
    svc_clf.fit(train_tfidf, y_train)
    
    svc_pred = svc_clf.predict(test_tfidf)
    svc_acc = accuracy_score(y_test, svc_pred)
    print(f"   ✅ TF-IDF+LinearSVC точность: {svc_acc*100:.2f}%")
    
    # === 3. Дополнительные признаки ===
    print("\n🔄 Извлекаем дополнительные признаки...")
    train_features = extract_features_batch(X_train)
    test_features = extract_features_batch(X_test)
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    print(f"   ✅ Извлечено {train_features.shape[1]} признаков")
    
    # === 4. АНСАМБЛЬ ИЗ НЕСКОЛЬКИХ XGBoost ===
    print("\n🔄 Обучаем ансамбль XGBoost...")
    
    sbert_proba_train = sbert_clf.predict_proba(train_emb)
    sbert_proba_test = sbert_clf.predict_proba(test_emb)
    
    svc_proba_train = svc_clf.predict_proba(train_tfidf)
    svc_proba_test = svc_clf.predict_proba(test_tfidf)
    
    X_train_meta = np.hstack([sbert_proba_train, svc_proba_train, train_features_scaled])
    X_test_meta = np.hstack([sbert_proba_test, svc_proba_test, test_features_scaled])
    
    # Создаем несколько XGBoost с разными параметрами
    xgb_models = []
    
    params_list = [
        {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.02, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 12, 'learning_rate': 0.03, 'subsample': 0.7, 'colsample_bytree': 0.7},
        {'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.025, 'subsample': 0.9, 'colsample_bytree': 0.6},
    ]
    
    for params in params_list:
        clf = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            objective='multi:softprob',
            num_class=4,
            random_state=42,
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=0.05,
            reg_lambda=0.05,
            tree_method='hist',
            eval_metric='mlogloss'
        )
        clf.fit(X_train_meta, y_train)
        xgb_models.append(clf)
    
    # Ансамблевое предсказание (усреднение вероятностей)
    probas = []
    for clf in xgb_models:
        probas.append(clf.predict_proba(X_test_meta))
    
    avg_proba = np.mean(probas, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    acc = accuracy_score(y_test, y_pred)
    
    # Сохраняем основной классификатор (первый XGBoost)
    meta_clf = xgb_models[0]
    
    print("\n" + "="*60)
    print("📊 ОЦЕНКА КАЧЕСТВА (УЛУЧШЕННЫЙ АНСАМБЛЬ)")
    print("="*60)
    
    target_names = ['Нормативное', 'Работоспособное', 'Ограниченно', 'Аварийное']
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"✅ Точность ансамбля: {acc*100:.2f}%")
    
    # Сохраняем модели
    joblib.dump(sbert_model, 'embedding_model.pkl')
    joblib.dump(sbert_clf, 'sbert_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(svc_clf, 'svc_classifier.pkl')
    joblib.dump(meta_clf, 'meta_classifier_xgb.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("\n✅ Все компоненты сохранены")
    
    return meta_clf, sbert_clf, svc_clf, sbert_model, vectorizer, scaler, xgb_models

# ================================================================
# 7. ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# ================================================================

def predict_with_ensemble(text, meta_clf=None, sbert_clf=None, svc_clf=None, 
                         sbert_model=None, vectorizer=None, scaler=None, xgb_models=None):
    """Предсказывает класс с использованием ансамбля"""
    
    if meta_clf is None:
        try:
            meta_clf = joblib.load('meta_classifier_xgb.pkl')
            sbert_clf = joblib.load('sbert_classifier.pkl')
            svc_clf = joblib.load('svc_classifier.pkl')
            sbert_model = joblib.load('embedding_model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')
            scaler = joblib.load('feature_scaler.pkl')
        except Exception as e:
            print(f"❌ Ошибка загрузки моделей: {e}")
            return None
    
    # Получаем предсказания
    emb = sbert_model.encode([text])
    sbert_proba = sbert_clf.predict_proba(emb)
    
    tfidf = vectorizer.transform([text])
    svc_proba = svc_clf.predict_proba(tfidf)
    
    features = extract_features_batch([text])
    features_scaled = scaler.transform(features)
    
    X_meta = np.hstack([sbert_proba, svc_proba, features_scaled])
    
    # Предсказание
    if xgb_models is not None:
        probas = []
        for clf in xgb_models:
            probas.append(clf.predict_proba(X_meta))
        avg_proba = np.mean(probas, axis=0)
        pred = np.argmax(avg_proba, axis=0)[0]
        proba = avg_proba[0]
    else:
        pred = meta_clf.predict(X_meta)[0]
        proba = meta_clf.predict_proba(X_meta)[0]
    
    # Применяем правила
    filtered_pred = apply_rules(text, pred, proba)
    final_pred = post_process(text, filtered_pred, proba)
    
    class_names = {0: 'НОРМАТИВНОЕ', 1: 'РАБОТОСПОСОБНОЕ', 2: 'ОГРАНИЧЕННО', 3: 'АВАРИЙНОЕ'}
    
    return {
        'class': final_pred,
        'class_name': class_names[final_pred],
        'confidence': float(max(proba)),
        'raw_pred': pred,
        'raw_proba': proba.tolist()
    }

# ================================================================
# 8. ТЕСТИРОВАНИЕ
# ================================================================

def test_ensemble(xgb_models=None):
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ УЛУЧШЕННОГО АНСАМБЛЯ")
    print("="*60)
    
    test_texts = [
        ("сквозная трещина в колонне", 3),
        ("коррозия закладных деталей", 2),
        ("дефектов не выявлено", 0),
        ("прогиб 40 мм превышает допустимый", 3),
        ("работоспособное состояние", 1),
        ("несущая способность не обеспечена", 3),
        ("усадочные трещины в колоннах", 2),
        ("осадки стабилизировались", 1),
        ("напряжения превышают расчетное сопротивление", 3),
        ("коэффициент технического состояния Ктс = 0", 3),
        ("качество кладки удовлетворительное", 1),
        ("требуется усиление ригелей", 2),
        ("прогнил насквозь", 3),
        ("техническое состояние фундамента работоспособное", 1),
        ("сколы защитного слоя бетона", 2),
        ("несущая способность основания обеспечена", 1),
        ("сквозная коррозия стенки двутавра", 3),
        ("поверхностная коррозия", 2),
        ("здание 9-ти этажное", 0),
        ("высота здания 33,3 м", 0),
        ("продольные трещины до 1,8 мм", 3),
        ("коэффициент использования 0,99", 3),
        ("осадки стабилизировались, деформаций не зафиксировано", 1),
        ("ремонт и восстановление конструкций", 2),
        ("недостаточная толщина защитного слоя бетона", 2),
        ("отсутствие антикоррозионного покрытия", 2),
        ("проектное решение каркаса по серии ИИ-04", 0),
        ("трещины в сварных швах", 3),
        ("фрагментация бетона стаканов колонн", 2),
        ("следы коррозии на поверхности", 2),
        ("многочисленные проломы в кладке наружной стены", 3),
        ("запас прочности обеспечен", 1),
        ("сколы защитного слоя с обнажением арматуры", 2),
        ("предаварийное состояние конструкции", 3),
    ]
    
    class_names = {0: 'НОРМАТИВНОЕ', 1: 'РАБОТОСПОСОБНОЕ', 2: 'ОГРАНИЧЕННО', 3: 'АВАРИЙНОЕ'}
    
    correct = 0
    errors = []
    
    for text, expected in test_texts:
        result = predict_with_ensemble(text, xgb_models=xgb_models)
        if result is None:
            return
        
        is_correct = (result['class'] == expected)
        if is_correct:
            correct += 1
        else:
            errors.append((text, expected, result['class']))
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Текст: {text}")
        print(f"   Ожидалось: {class_names[expected]}")
        print(f"   → ИТОГ: {class_names[result['class']]} (уверенность: {result['confidence']*100:.1f}%)")
    
    print(f"\n📊 ИТОГО: {correct}/{len(test_texts)} правильно ({correct/len(test_texts)*100:.1f}%)")
    
    if errors:
        print("\n❌ ОШИБКИ:")
        for text, exp, pred in errors:
            print(f"   Текст: {text[:50]}...")
            print(f"      Ожидалось: {class_names[exp]}, Получено: {class_names[pred]}")

# ================================================================
# 9. ЗАПУСК
# ================================================================

if __name__ == "__main__":
    csv_file = 'data_for_labeling_ml_fixed.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ Файл '{csv_file}' не найден!")
        exit()
    
    print("📚 Загружаем размеченные данные...")
    try:
        texts, labels = load_data(csv_file)
        print(f"✅ Загружено {len(texts)} фрагментов")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        exit()
    
    if len(texts) < 10:
        print("❌ Слишком мало данных для обучения")
        exit()
    
    # Обучаем улучшенный ансамбль
    meta_clf, sbert_clf, svc_clf, sbert_model, vectorizer, scaler, xgb_models = train_ensemble(texts, labels)
    
    # Тестируем с передачей xgb_models для ансамблевого предсказания
    test_ensemble(xgb_models)
    
    print("\n🎉 Обучение завершено!")