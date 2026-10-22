import re
import pypdf  
import pandas as pd
import os
import joblib
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import json
from typing import Dict, List, Any, Tuple, Optional


try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"{e}")
except Exception as e:
    OCR_AVAILABLE = False
    print(f"Ошибка настройки OCR: {e}")


def load_russian_words():
    common_words = [
        'фундамент', 'стена', 'колонна', 'балка', 'ригель', 'перекрытие', 
        'плита', 'крыша', 'кровля', 'лестница', 'подвал', 'кладка',
        'кирпич', 'бетон', 'арматура', 'сталь', 'коррозия', 'трещина',
        'обрушение', 'аварийный', 'работоспособный', 'нормативный',
        'обследование', 'дефект', 'повреждение', 'усиление', 'ремонт',
        'здание', 'сооружение', 'этаж', 'помещение', 'конструкция',
        'несущий', 'ограждающий', 'внутренний', 'наружный', 'продольный',
        'поперечный', 'горизонтальный', 'вертикальный', 'монолитный',
        'сборный', 'железобетонный', 'стальной', 'деревянный'
    ]
    return set(common_words)

def smart_clean_text(text):
   
    if not text:
        return ""
    
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\"\'\№\d]', ' ', text)
    
    text = re.sub(r'(?<![.!?]\s)([а-яa-z])([А-ЯA-Z])', r'\1 \2', text)
    
    text = re.sub(r'\.([А-ЯA-Zа-яa-z])', r'. \1', text)
    text = re.sub(r',([А-ЯA-Zа-яa-z])', r', \1', text)
    
    text = re.sub(r'(\d)([А-ЯA-Zа-яa-z])', r'\1 \2', text)
    text = re.sub(r'([А-ЯA-Zа-яa-z])(\d)', r'\1 \2', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def fix_pdf_text(text):
  
    if not text:
        return ""
    
    text = re.sub(r'[^А-Яа-яA-Za-z0-9\s\.\,\-\:\;\(\)]', ' ', text)
    
    text = re.sub(r'(?<![.!?]\s)([а-яa-z])([А-ЯA-Z])', r'\1 \2', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_text(text):

    if not text:
        return ""
    
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\"\'\№\d]', ' ', text)
    
    text = re.sub(r'([а-яa-z])([А-ЯA-Z])', r'\1 \2', text)
    text = re.sub(r'([А-ЯA-Z])([а-яa-z])', r'\1 \2', text)
    
    text = re.sub(r'(\d)([А-ЯA-Zа-яa-z])', r'\1 \2', text)
    text = re.sub(r'([А-ЯA-Zа-яa-z])(\d)', r'\1 \2', text)
    
    text = re.sub(r'\.([А-ЯA-Zа-яa-z])', r'. \1', text)
    text = re.sub(r',([А-ЯA-Zа-яa-z])', r', \1', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_axis(text):
    
    if not text or text == 'не указана':
        return 'не указана'
    
    text = re.sub(r'[^А-ЯA-Zа-яa-z0-9\-/]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    garbage = [
        'имеется', 'усилены', 'находится', 'состояние', 'техническое',
        'кирпичных', 'несущих', 'стени', 'фундамента', 'здания',
        'строительных', 'конструкций', 'обследование', 'повреждение',
        'дефекты', 'ремонт', 'усиление', 'замена', 'восстановление'
    ]
    
    for word in garbage:
        if word in text.lower():
            return 'не указана'
    
    if len(text) > 15:
        return 'не указана'
    
    return text if len(text) > 0 else 'не указана'



def load_ml_model():
    try:
        sbert_model = joblib.load('embedding_model.pkl')
        sbert_clf = joblib.load('sbert_classifier.pkl')
        svc_clf = joblib.load('svc_classifier.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        meta_clf = joblib.load('meta_classifier_xgb.pkl')
        print("ML-модель загружена")
        return sbert_model, sbert_clf, svc_clf, vectorizer, scaler, meta_clf
    except Exception as e:
        print(f"ML-модель не загружена: {e}")
        return None, None, None, None, None, None



CONSTRUCT_CATEGORIES = {
    'Фундамент': ['фундамент', 'свая', 'ростверк', 'плита фундаментная', 'лента', 'подколонник', 'основание', 'грунт',  'свайный', 'буронабивной', 'забивной'],
    'Вертикальные': ['стена', 'колонна', 'пилон', 'пилястра', 'простенок', 'перемычка', 'стойка', 'опора', 
                     'ядро жесткости', 'ствол жесткости', 'диафрагма', 'панель', 'кладка'],
    'Перекрытия': ['плита перекрытия', 'плита покрытия', 'перекрытие', 'балка', 'ригель', 'консоль', 
                   'панель перекрытия', 'прогон', 'покрытие', 'свод'],
    'Крыша': ['крыша', 'кровля', 'стропила', 'стропильная нога', 'ферма', 'кровельный ковер'],
    'Лестница': ['лестница', 'лестничная площадка', 'лестничный марш', 'косоур', 'шахта лифта'],
    'Связи': ['цоколь', 'карниз', 'парапет', 'балкон', 'лоджия', 'эркер', 'обойма', 'подкос', 'связь', 'затяжка', 'арка', 'свод']
}

FALSE_CONSTRUCTS = ['сводная', 'сводный', 'сводной', 'сводных']

CONSTRUCT_TO_CATEGORY = {}
for category, keywords in CONSTRUCT_CATEGORIES.items():
    for keyword in keywords:
        CONSTRUCT_TO_CATEGORY[keyword] = category


def get_construct_category(construct_name):
    construct_lower = construct_name.lower()
    for false_word in FALSE_CONSTRUCTS:
        if false_word in construct_lower:
            return None
    for keyword, category in CONSTRUCT_TO_CATEGORY.items():
        if keyword in construct_lower:
            return category
    return 'Другое'


def extract_axis(text):
    
    if not text:
        return None
    axis_patterns = [
        r'ос[иь]\s*([А-Яа-яA-Za-z0-9]+)',
        r'в\s*осях\s*([0-9А-Яа-яA-Za-z\-/]+)',
        r'осях\s*([0-9А-Яа-яA-Za-z\-/]+)',
        r'по\s*оси\s*([А-Яа-яA-Za-z0-9]+)',
    ]
    for pattern in axis_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            axis = match.group(1).strip()
            return clean_axis(axis)
    return None

def restore_spaces(text: str) -> str:
    """Восстанавливает пробелы между русскими словами, потерянные при извлечении из PDF"""
    if not text:
        return text
    
    # Восстанавливаем пробелы между словами из букв
    # Шаблон: буква+буква -> буква буква
    text = re.sub(r'([а-яА-Я])([А-Я])', r'\1 \2', text)  # строчная + заглавная
    text = re.sub(r'([А-Я])([а-я])', r'\1 \2', text)     # заглавная + строчная
    
    # Специальные случаи для частых слов в отчетах
    replacements = {
        'сводовспереборкойотдельных': 'сводов переборкой отдельных',
        'участковкладки': 'участков кладки',
        'усилениебалок': 'усиление балок',
        'наличиевременных': 'наличие временных',
        'Перекрытиенад': 'Перекрытие над',
        'мэтажом': 'м этажом',
        'этажей': 'этажей',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Восстанавливаем пробелы после цифр
    text = re.sub(r'(\d)([А-Яа-я])', r'\1 \2', text)
    text = re.sub(r'([А-Яа-я])(\d)', r'\1 \2', text)
    
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


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


def extract_features_batch(texts):
    features_list = []
    for text in texts:
        text_lower = text.lower()
        danger_count = sum(1 for w in DANGER_WORDS if w in text_lower)
        limited_count = sum(1 for w in LIMITED_WORDS if w in text_lower)
        safe_count = sum(1 for w in SAFE_WORDS if w in text_lower)
        pure_count = sum(1 for w in PURE_DESC if w in text_lower)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        mm_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*мм', text_lower)
        percent_numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_lower)
        has_number = len(numbers) > 0
        has_mm = len(mm_numbers) > 0
        has_percent = len(percent_numbers) > 0
        max_mm = max([float(x) for x in mm_numbers]) if mm_numbers else 0
        max_percent = max([float(x) for x in percent_numbers]) if percent_numbers else 0
        ratio_danger_limited = danger_count / (limited_count + 1)
        ratio_safe_danger = safe_count / (danger_count + 1)
        ratio_limited_pure = limited_count / (pure_count + 1)
        features_list.append([
            len(text), len(text.split()), danger_count, limited_count,
            safe_count, pure_count, int(has_number), int(has_mm),
            int(has_percent), max_mm, max_percent,
            ratio_danger_limited, ratio_safe_danger, ratio_limited_pure
        ])
    return np.array(features_list)



def post_process(text, pred_class, proba):
    text_lower = text.lower()
    
    danger_strong = [
        'аварийн', 'обруш', 'сквозн', 'прогнил', 'насквозь',
        'несущая способность не', 'ктс = 0', 'критический прогиб',
        'угроза обруш', 'предельное состояние', 'недопустимое состояние'
    ]
    for word in danger_strong:
        if word in text_lower:
            return 3
    
    if 'не выявл' in text_lower or 'отсутств' in text_lower or 'не обнаруж' in text_lower:
        return 0
    
    mm_match = re.search(r'прогиб\s*(\d+)\s*мм', text_lower)
    if mm_match and int(mm_match.group(1)) > 30:
        return 3
    
    mm_match = re.search(r'трещин\w*\s*(\d+)\s*мм', text_lower)
    if mm_match and int(mm_match.group(1)) > 2:
        return 3
    
    if max(proba) > 0.85:
        return pred_class
    
    return pred_class

def fix_broken_pdf_text(text):
    """Восстановление текста из битых PDF"""
    if not text:
        return ""
    
    if re.match(r'^[\d\s]+$', text[:100]):
        return "" 
    
    # Удаляем мусорные символы
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\"\'\№\d]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


SECTION_PATTERNS = {
    'фундамент': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+фундамента',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+фундамента\s+и\s+подвала',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Фундамент',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Основание',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Подвал',
    ],
    'стены': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+несущих\s+стен',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+стен',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?(?:Несущие\s+)?Стены',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Колонны',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Вертикальные\s+конструкции',
    ],
    'перекрытия': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+междуэтажных\s+перекрытий',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Обследование\s+перекрытий',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Перекрытия',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Плиты\s+перекрытия',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Междуэтажные\s+перекрытия',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Покрытие',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Балки',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Ригели',
    ],
    'крыша': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?(?:Обследование\s+)?Крыша',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Кровля',
    ],
    'лестницы': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Лестницы',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Лестничные\s+клетки',
    ],
    'выводы': [
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Выводы',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Заключение',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?ЗАКЛЮЧЕНИЕ',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Рекомендации',
        r'(?:\n|^)\s*(?:\d+\.?\s*)?Результаты\s+обследования',
    ]
}


def split_into_sections(full_text):
    sections = {}
    text = re.sub(r'\s+', ' ', full_text)
    
    headers = []
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start_pos = match.start()
                if start_pos > 0 and text[start_pos-1].isalpha():
                    continue
                headers.append((start_pos, section_name, match.group(0)))
                break
    
    headers.sort(key=lambda x: x[0])
    
    if not headers:
        sections['выводы'] = text
        return sections
    
    for i, (start_pos, section_name, header_text) in enumerate(headers):
        section_start = start_pos + len(header_text)
        section_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        section_text = text[section_start:section_end].strip()
        section_text = re.sub(r'\s+', ' ', section_text)
        
        if len(section_text) > 30:
            if section_name not in sections:
                sections[section_name] = []
            sections[section_name].append(section_text)
    
    for name in sections:
        sections[name] = ' '.join(sections[name])
    
    return sections


def extract_reinforcement(text: str) -> Dict[str, Any]:
   
    if not text:
        return {'has_reinforcement': False, 'year': None, 'description': None}
    
    text_lower = text.lower()
    
    reinforcement_keywords = [
        'усилен', 'усиление', 'армирован', 'армирование',
        'восстановлен', 'восстановление', 'ремонт',
        'заменен', 'замена', 'реконструкци'
    ]
    
    year_pattern = r'(\d{4})\s*г(?:од)?\.?\s*(?:усилен|восстановлен|заменен|ремонт)'
    year_match = re.search(year_pattern, text, re.IGNORECASE)
    year = year_match.group(1) if year_match else None
    
    has_reinforcement = any(kw in text_lower for kw in reinforcement_keywords)
    
    desc_patterns = [
        r'(?:усилен[ие]|восстановлен[ие]|ремонт)\s*[^.!?]{10,150}',
        r'(?:выполнен[о]?\s+)?усилен[ие]\s+[^.!?]{10,150}',
    ]
    
    description = None
    for pattern in desc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            description = match.group(0).strip()
            break
    
    return {
        'has_reinforcement': has_reinforcement,
        'year': year,
        'description': description
    }


def add_reinforcement_info(all_constructs: List[Dict]) -> List[Dict]:

    for construct in all_constructs:
        citation = construct.get('Цитата', '')
        reinforcement = extract_reinforcement(citation)
        construct['Усиление'] = reinforcement.get('description', '')
        construct['Год_усиления'] = reinforcement.get('year', '')
        construct['Требует_усиления'] = reinforcement.get('has_reinforcement', False)
    
    return all_constructs

def analyze_pdf(pdf_path, use_ml=True, collect_all_citations=False):
    full_text = ""
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text() or ""
                if text:
                    full_text += fix_pdf_text(text) + " "
    except Exception as e:
        print(f"Ошибка чтения PDF: {e}")
    
    if len(full_text.strip()) < 500 and OCR_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), 30)): 
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang='rus')
                full_text += fix_pdf_text(page_text) + " "
            doc.close()
        except Exception as e:
            print(f" OCR ошибка: {e}")
    
    if len(full_text.strip()) < 100:
        print(f"  Не удалось извлечь текст из {pdf_path}")
        return {
            'sections': {},
            'all_constructs': [],
            'has_collapse': False,
            'total_constructs': 0
        }
    
    
    full_text = smart_clean_text(full_text)
    #full_text = restore_spaces(full_text)
   
    sbert_model, sbert_clf, svc_clf, vectorizer, scaler, meta_clf = load_ml_model() if use_ml else (None, None, None, None, None, None)
    ml_models = (sbert_model, sbert_clf, svc_clf, vectorizer, scaler, meta_clf)
    
    sections = split_into_sections(full_text)
    
    section_results = {}
    all_constructs = []
    total_collapse = False
    
    for section_name, section_text in sections.items():
        if len(section_text) < 20:
            continue
        
        result = analyze_section(
            section_text, 
            section_name, 
            ml_models, 
            collect_all_citations=collect_all_citations  
        )
        if result is None:
            continue
            
        section_results[section_name] = result
        all_constructs.extend(result['constructs'])
        if result['has_collapse']:
            total_collapse = True
    
    all_constructs = add_reinforcement_info(all_constructs)
    
    return {
        'sections': section_results,
        'all_constructs': all_constructs,
        'has_collapse': total_collapse,
        'total_constructs': len(all_constructs)
    }

def extract_section(text: str) -> Dict[str, str]:
   
    if not text:
        return {'type': None, 'value': None}
    
    text_lower = text.lower()
    
    patterns = {
        'подъезд': [
            r'подъезд\s*[№#]?\s*(\d+[а-я]?)',
            r'(\d+[а-я]?)\s*подъезд',
            r'п\.\s*(\d+[а-я]?)',
        ],
        'литер': [
            r'литер\s*([А-Я])',
            r'лит\.\s*([А-Я])',
            r'([А-Я])\s*литер',
        ],
        'секция': [
            r'секци[яи]\s*[№#]?\s*(\d+[а-я]?)',
            r'(\d+[а-я]?)\s*секци[яи]',
        ],
        'корпус': [
            r'корпус\s*[№#]?\s*(\d+[а-я]?)',
            r'корп\.\s*(\d+[а-я]?)',
            r'(\d+[а-я]?)\s*корпус',
        ],
        'этаж': [
            r'(\d+[а-я]?)\s*этаж',
            r'этаж\s*(\d+[а-я]?)',
        ],
        'крыло': [
            r'(лево[её]|право[её])\s*крыло',
            r'крыло\s*([А-Я])',
        ],
        'ось': [
            r'ос[иь]\s*([А-Яа-яA-Za-z0-9]+)',
            r'в\s*осях\s*([0-9А-Яа-яA-Za-z\-/]+)',
            r'по\s*оси\s*([А-Яа-яA-Za-z0-9]+)',
            r'([А-Яа-яA-Za-z])\s*-\s*([А-Яа-яA-Za-z])',
        ],
    }
    
    for section_type, section_patterns in patterns.items():
        for pattern in section_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = match.group(1).strip() if match.groups() else match.group(0).strip()
               
                value = re.sub(r'[^\w\-/]', '', value)
                if len(value) > 0 and len(value) < 20:
                   
                    if section_type == 'ось' and '-' in value:
                        return {'type': 'ось', 'value': value}
                    return {'type': section_type, 'value': value}
    
    return {'type': None, 'value': None}


class CitationSelector:
    """Класс для работы с цитатами"""
    
    DEFECT_KEYWORDS = [
        'трещин', 'разруш', 'коррози', 'деформ', 'скол', 'износ',
        'протечк', 'увлажн', 'высол', 'отслоени', 'выветриван',
        'сквозн', 'обруш', 'прогиб', 'гниль', 'ржавч', 'поврежд',
        'дефект', 'осадк', 'выкрашиван', 'отклонен', 'осыпани',
        'выветриван', 'замачиван', 'каверн', 'разрыв', 'отрыв',
        'недостаточн', 'превыша', 'недопустим', 'исчерпан'
    ]
    
    @classmethod
    def has_defect(cls, text: str) -> bool:
        if not text:
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in cls.DEFECT_KEYWORDS)
    
    @classmethod
    def count_defects(cls, text: str) -> int:
        if not text:
            return 0
        text_lower = text.lower()
        return sum(1 for kw in cls.DEFECT_KEYWORDS if kw in text_lower)
    
    @classmethod
    def select_best_citation(cls, citations: List[str]) -> tuple:
        if not citations:
            return "", []
        
        clean_citations = [c.strip() for c in citations if len(c.strip()) > 10]
        if not clean_citations:
            return "", []
        
        sorted_citations = sorted(
            clean_citations,
            key=lambda x: (cls.has_defect(x), cls.count_defects(x), len(x)),
            reverse=True
        )
        
        return sorted_citations[0], sorted_citations[:5]
    
    @classmethod
    def extract_defect_description(cls, citation: str) -> str:
        if not citation:
            return ""
        
        sentences = re.split(r'[.!?]', citation)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and cls.has_defect(sent):
                return sent[:200]
        
        return sentences[0].strip()[:200] if sentences else ""

def analyze_section(section_text, section_name, ml_models, collect_all_citations=False):
    sbert_model, sbert_clf, svc_clf, vectorizer, scaler, meta_clf = ml_models
    ml_available = sbert_model is not None
    
    # Разбиваем на предложения
    sentences = re.split(r'[.!?]', section_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    
    # ФИЛЬТРУЕМ МУСОРНЫЕ ПРЕДЛОЖЕНИЯ
    garbage_patterns = [
        r'^вес элемента',
        r'^табл\.?\s*\d+',
        r'^рис\.?\s*\d+',
        r'^фото\s*[А-Я]?\s*\d+',
        r'^\d+\s*[мс][м]?\s*',
        r'^п\s*п\s*\d+',
        r'^№\s*п/п',
        r'^\(на основании',
        r'^согласно\s+[А-Я]+[\d\.-]+',
        r'^Общая оценка',
        r'^Относительная надежность',
        r'^Срок эксплуатации',
    ]
    
    
    filtered_sentences = []
    for s in sentences:
        is_garbage = False
        for pattern in garbage_patterns:
            if re.match(pattern, s, re.IGNORECASE):
                is_garbage = True
                break
        if not is_garbage and len(s) > 20:
            filtered_sentences.append(s)
    
    sentences = filtered_sentences
    
    results = []
    has_collapse = False
    
    # Группируем предложения по конструкциям
    construct_sentences = defaultdict(list)
    
    for sentence in sentences:
        # Пропускаем технические
        if any(word in sentence.lower() for word in ['рис', 'табл', 'прилож', 'см', 'формул']):
            continue
        
        found_construct = None
        found_category = None
        construct_lower = sentence.lower()
        
        # Проверка на ложные конструкции
        is_false = False
        for false_word in FALSE_CONSTRUCTS:
            if false_word in construct_lower:
                is_false = True
                break
        if is_false:
            continue
        
        # Поиск конструкции
        for category, keywords in CONSTRUCT_CATEGORIES.items():
            for keyword in keywords:
                if keyword in construct_lower:
                    found_construct = keyword
                    found_category = category
                    break
            if found_construct:
                break
        
        if found_construct:
            construct_sentences[found_construct].append({
                'sentence': sentence,
                'category': found_category,
                'has_defect': CitationSelector.has_defect(sentence),
                'defect_count': CitationSelector.count_defects(sentence),
                'length': len(sentence)
            })
    
    # Для каждой конструкции выбираем лучшую цитату
    for construct_name, sentences_data in construct_sentences.items():
        # Выбираем предложение с наибольшим количеством дефектов
        best_sentence_data = select_best_sentence(sentences_data)
        best_sentence = best_sentence_data['sentence']
        #best_sentence = restore_spaces(best_sentence)
        category = best_sentence_data['category']
        
        # Все цитаты для этой конструкции
        all_citations = [s['sentence'] for s in sentences_data]
        
        # ML классификация
        if ml_available:
            try:
                emb = sbert_model.encode([best_sentence])
                sbert_proba = sbert_clf.predict_proba(emb)
                tfidf = vectorizer.transform([best_sentence])
                svc_proba = svc_clf.predict_proba(tfidf)
                features = extract_features_batch([best_sentence])
                features_scaled = scaler.transform(features)
                X_meta = np.hstack([sbert_proba, svc_proba, features_scaled])
                pred = meta_clf.predict(X_meta)[0]
                proba = meta_clf.predict_proba(X_meta)[0]
                final_pred = post_process(best_sentence, pred, proba)
                status_map = {0: 'НОРМАТИВНОЕ', 1: 'РАБОТОСПОСОБНОЕ', 
                             2: 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 3: 'АВАРИЙНОЕ'}
                status = status_map.get(final_pred, 'НЕ ОПРЕДЕЛЕНО')
            except Exception as e:
                status, _ = get_status_by_rules(best_sentence)
        else:
            status, _ = get_status_by_rules(best_sentence)
        
        # Извлечение разделения
        section_info = extract_section(best_sentence)
        section_type = section_info.get('type')
        section_value = section_info.get('value')
        
        if section_type and section_value:
            type_names = {
                'ось': 'Ось',
                'подъезд': 'Подъезд',
                'литер': 'Литер',
                'секция': 'Секция',
                'корпус': 'Корпус',
                'этаж': 'Этаж',
                'крыло': 'Крыло'
            }
            display_name = f"{type_names.get(section_type, section_type)} {section_value}"
        else:
            display_name = 'не указана'
        
        # Проверка на угрозу обрушения
        has_collapse_for_construct = any(
            word in best_sentence.lower() for word in ['обруш', 'угроз', 'аварийн', 'сквозн']
        )
        if has_collapse_for_construct:
            has_collapse = True
        
        results.append({
            "Конструкция": construct_name,
            "Категория": category,
            "Разделение": display_name,
            "Тип_разделения": section_type,
            "Статус": status,
            "Цитата": best_sentence,
            "all_citations": all_citations if collect_all_citations else [],
            "Секция": section_name,
            "Угроза_обрушения": has_collapse_for_construct
        })
    
    section_status = get_section_status(results)
    
    return {
        'section_name': section_name,
        'section_status': section_status,
        'constructs': results,
        'has_collapse': has_collapse,
        'total_constructs': len(results)
    }


def select_best_sentence(sentences_data: List[Dict]) -> Dict:
    """Выбор лучшего предложения - с приоритетом на дефекты"""
    # Сортируем по приоритету:
    # 1. Есть дефекты (has_defect=True)
    # 2. Больше слов-дефектов (defect_count)
    # 3. Длиннее (больше информации)
    # 4. Не начинается с мусорных фраз
    sorted_sentences = sorted(
        sentences_data,
        key=lambda x: (
            x['has_defect'],
            x['defect_count'],
            x['length']
        ),
        reverse=True
    )
    
    return sorted_sentences[0]

def get_section_status(results):
    if not results:
        return 'НЕ ОПРЕДЕЛЕНО'
    
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['Статус']] += 1
    
    if status_counts.get('АВАРИЙНОЕ', 0) > 0:
        return 'АВАРИЙНОЕ'
    if status_counts.get('ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 0) > 0:
        return 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'
    if status_counts.get('РАБОТОСПОСОБНОЕ', 0) > 0:
        return 'РАБОТОСПОСОБНОЕ'
    return 'НОРМАТИВНОЕ'


def get_status_by_rules(sentence):
    text_lower = sentence.lower()
    danger_words = ['сквозн', 'обруш', 'не обеспечен', 'ктс', 'прогиб', 'превыша',
                    'недопустим', 'исчерпан', 'срез', 'обрыв', 'аварийн', 'критическ',
                    'разруш', 'потеря устойчивости', 'угроза', 'предельное состояние']
    for word in danger_words:
        if word in text_lower:
            return 'АВАРИЙНОЕ', 'Обнаружено аварийное состояние'
    working_words = ['работоспособное', 'удовлетворительное', 'осадки стабилизировались',
                     'несущая способность обеспечена', 'запас прочности']
    for word in working_words:
        if word in text_lower:
            return 'РАБОТОСПОСОБНОЕ', 'Состояние работоспособное'
    limited_words = ['коррози', 'скол', 'усадочн', 'износ', 'замачиван', 'высол',
                     'выветриван', 'недостаточн', 'фрагментаци', 'ремонт', 'усилени',
                     'поврежден', 'дефект']
    for word in limited_words:
        if word in text_lower:
            return 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 'Требуется ремонт или усиление'
    if 'не выявл' in text_lower or 'отсутств' in text_lower or 'не обнаруж' in text_lower:
        return 'НОРМАТИВНОЕ', 'Дефектов не выявлено'
    return 'НОРМАТИВНОЕ', 'Состояние не определено как опасное'



SECTION_WEIGHTS = {
    'фундамент': 30,
    'стены': 35,
    'перекрытия': 20,
    'крыша': 10,
    'лестницы': 5,
    'выводы': 0,
}

STATUS_SCORES = {
    'НОРМАТИВНОЕ': 0,
    'РАБОТОСПОСОБНОЕ': 1,
    'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': 3,
    'АВАРИЙНОЕ': 10,
    'НЕ ОПРЕДЕЛЕНО': 2,
}


def calculate_priority(analysis_result, has_court_order=False):
    
    if analysis_result['has_collapse'] or has_court_order:
        return {
            'priority': 'КРИТИЧЕСКОЕ',
            'color': '🔴',
            'score': 100,
            'health_index': 0,
            'reason': 'Обнаружена угроза обрушения' if analysis_result['has_collapse'] else 'Судебное решение',
            'details': {'has_collapse': analysis_result['has_collapse'], 'has_court_order': has_court_order}
        }
    
    
    total_weight = 0
    weighted_score = 0
    
    for section_name, result in analysis_result['sections'].items():
        weight = SECTION_WEIGHTS.get(section_name, 0)
        if weight == 0:
            continue
        
        status = result['section_status']
        status_score = STATUS_SCORES.get(status, 2)
        
        dangerous = [c for c in result['constructs'] if c['Статус'] == 'АВАРИЙНОЕ']
        if dangerous:
            status_score = max(status_score, 10)
        
        weighted_score += weight * status_score
        total_weight += weight
    
    
    if total_weight == 0:
        all_constructs = analysis_result['all_constructs']
        if not all_constructs:
            return {'priority': 'НИЗКИЙ', 'color': '🟢', 'score': 0, 'health_index': 100, 'reason': 'Конструкции не найдены'}
        
        status_counts = defaultdict(int)
        for c in all_constructs:
            status_counts[c['Статус']] += 1
        
        avg_score = 0
        total = len(all_constructs)
        for status, count in status_counts.items():
            avg_score += STATUS_SCORES.get(status, 2) * count
        avg_score = avg_score / total
        
        raw_score = (avg_score / 10) * 100
        health_index = 100 - raw_score
        
        if raw_score >= 70:
            return {'priority': 'ВЫСОКИЙ', 'color': '🟠', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Средний балл: {avg_score:.1f}/10'}
        elif raw_score >= 40:
            return {'priority': 'СРЕДНИЙ', 'color': '🟡', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Средний балл: {avg_score:.1f}/10'}
        else:
            return {'priority': 'НИЗКИЙ', 'color': '🟢', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Средний балл: {avg_score:.1f}/10'}
    
    
    max_possible = total_weight * 10
    raw_score = weighted_score / max_possible * 100
    health_index = 100 - raw_score
    
    if raw_score >= 70:
        return {'priority': 'ВЫСОКИЙ', 'color': '🟠', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Общий коэффициент состояния здания: {health_index:.1f}%'}
    elif raw_score >= 40:
        return {'priority': 'СРЕДНИЙ', 'color': '🟡', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Общий коэффициент состояния здания: {health_index:.1f}%'}
    else:
        return {'priority': 'НИЗКИЙ', 'color': '🟢', 'score': round(raw_score, 1), 'health_index': round(health_index, 1), 'reason': f'Общий коэффициент состояния здания: {health_index:.1f}%'}


if __name__ == "__main__":
    PDF_PATH = "reports/oychetlihack.pdf"
    try:
        result = analyze_pdf(PDF_PATH, use_ml=True)
        priority = calculate_priority(result)
        print(f"Найдено конструкций: {result['total_constructs']}")
        print(f"Приоритет: {priority['priority']}")
        print(f"Общий коэффициент состояния здания: {priority['health_index']}%")
    except Exception as e:
        print(f"Ошибка: {e}")