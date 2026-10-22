import re
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from main import analyze_pdf, calculate_priority, extract_reinforcement
from rag_analyzer import RAGSystem
from database import Database


class DivisionType(Enum):
    AXIS = "Ось"
    PODJEZD = "Подъезд"
    LITER = "Литер"
    SECTION = "Секция"
    BUILDING = "Корпус"
    FLOOR = "Этаж"
    WING = "Крыло"
    UNKNOWN = "Не указана"


@dataclass
class ConstructInfo:
    name: str
    category: str
    status: str
    citation: str  
    all_citations: List[str] 
    section: str
    division_type: DivisionType
    division_value: str
    division_display: str
    description: str = ""
    reinforcement: str = ""
    reinforcement_year: str = ""
    has_collapse_risk: bool = False
    generation_method: str = "template"
    confidence: float = 0.0  
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'Конструкция': self.name,
            'Категория': self.category,
            'Статус': self.status,
            'Цитата': self.citation,
            'Все_цитаты': self.all_citations,
            'Секция': self.section,
            'Тип_разделения': self.division_type.value,
            'Разделение': self.division_display,
            'llm_description': self.description,
            'Усиление': self.reinforcement,
            'Год_усиления': self.reinforcement_year,
            'Угроза_обрушения': self.has_collapse_risk,
            'generation_method': self.generation_method,
            'confidence': self.confidence
        }


class CitationSelector:
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
    def select_best_citation(cls, citations: List[str]) -> Tuple[str, List[str]]:
     
        if not citations:
            return "", []
        
        clean_citations = []
        for c in citations:
            c = c.strip()
            if len(c) > 10: 
                clean_citations.append(c)
        
        if not clean_citations:
            return "", []
        
        sorted_citations = sorted(
            clean_citations,
            key=lambda x: (
                cls.has_defect(x),  
                cls.count_defects(x), 
                len(x)              
            ),
            reverse=True
        )
        best = sorted_citations[0]
        
        all_citations = sorted_citations[:5]
        
        return best, all_citations
    
    @classmethod
    def extract_defect_description(cls, citation: str) -> str:
        if not citation:
            return ""
        
        sentences = re.split(r'[.!?]', citation)
        defect_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            if cls.has_defect(sent):
                defect_sentences.append(sent)
        
        if defect_sentences:
            return defect_sentences[0][:200]
        
        if sentences:
            return sentences[0].strip()[:200]
        
        return ""

@dataclass
class DivisionGroup:
    division_type: DivisionType
    division_value: str
    division_display: str
    constructs: List[ConstructInfo] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.constructs)
    
    @property
    def danger_count(self) -> int:
        return sum(1 for c in self.constructs if c.status == 'АВАРИЙНОЕ')
    
    @property
    def limited_count(self) -> int:
        return sum(1 for c in self.constructs if c.status == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ')
    
    @property
    def working_count(self) -> int:
        return sum(1 for c in self.constructs if c.status == 'РАБОТОСПОСОБНОЕ')
    
    @property
    def normative_count(self) -> int:
        return sum(1 for c in self.constructs if c.status == 'НОРМАТИВНОЕ')
    
    @property
    def has_collapse_risk(self) -> bool:
        return any(c.has_collapse_risk for c in self.constructs)
    
    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
       
        summary = defaultdict(lambda: {
            'total': 0,
            'danger': 0,
            'limited': 0,
            'working': 0,
            'normative': 0
        })
        
        for c in self.constructs:
            cat = c.category
            summary[cat]['total'] += 1
            if c.status == 'АВАРИЙНОЕ':
                summary[cat]['danger'] += 1
            elif c.status == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ':
                summary[cat]['limited'] += 1
            elif c.status == 'РАБОТОСПОСОБНОЕ':
                summary[cat]['working'] += 1
            else:
                summary[cat]['normative'] += 1
        
        return dict(summary)


class DivisionExtractor:
    """
    КЛАСС ДЛЯ ИЗВЛЕЧЕНИЯ ВСЕХ ТИПОВ РАЗДЕЛЕНИЙ
    Поддерживает: оси, подъезды, литеры, секции, корпуса, этажи, крылья
    """
    
    PATTERNS = {
        DivisionType.AXIS: [
            r'ос[иь]\s*([А-Яа-яA-Za-z0-9]+)',
            r'в\s*осях\s*([0-9А-Яа-яA-Za-z\-/]+)',
            r'по\s*оси\s*([А-Яа-яA-Za-z0-9]+)',
            r'между\s*осями\s*([0-9А-Яа-яA-Za-z\-/]+)',
        ],
        DivisionType.PODJEZD: [
            r'подъезд\s*[№#]?\s*(\d+[а-я]?)',
            r'(\d+[а-я]?)\s*подъезд',
            r'п\.\s*(\d+[а-я]?)',
        ],
        DivisionType.LITER: [
            r'литер\s*([А-Я])',
            r'лит\.\s*([А-Я])',
            r'([А-Я])\s*литер',
        ],
        DivisionType.SECTION: [
            r'секци[яи]\s*[№#]?\s*(\d+[а-я]?)',
            r'(\d+[а-я]?)\s*секци[яи]',
        ],
        DivisionType.BUILDING: [
            r'корпус\s*[№#]?\s*(\d+[а-я]?)',
            r'корп\.\s*(\d+[а-я]?)',
        ],
        DivisionType.FLOOR: [
            r'(\d+[а-я]?)\s*этаж',
            r'этаж\s*(\d+[а-я]?)',
        ],
        DivisionType.WING: [
            r'(лево[её]|право[её])\s*крыло',
            r'крыло\s*([А-Я])',
        ],
    }
    
    PRIORITY = [
        DivisionType.AXIS,
        DivisionType.PODJEZD,
        DivisionType.SECTION,
        DivisionType.BUILDING,
        DivisionType.LITER,
        DivisionType.WING,
        DivisionType.FLOOR,
    ]
    
    @classmethod
    def extract_all(cls, text: str) -> List[Tuple[DivisionType, str]]:
        """Извлечение всех типов разделений"""
        if not text:
            return []
        
        results = []
        text_lower = text.lower()
        
        for div_type, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    value = match.group(1).strip() if match.groups() else match.group(0).strip()
                    value = re.sub(r'[^\w\-/]', '', value)
                    if value and len(value) < 20:
                        results.append((div_type, value))
                        break
        
        return results
    
    @classmethod
    def extract_primary(cls, text: str) -> Tuple[DivisionType, str, str]:
        """Извлечение основного разделения"""
        all_divisions = cls.extract_all(text)
        
        for div_type in cls.PRIORITY:
            for d_type, value in all_divisions:
                if d_type == div_type:
                    display = cls.format_display(d_type, value)
                    return d_type, value, display
        
        return DivisionType.UNKNOWN, "не указана", "не указана"
    
    @classmethod
    def format_display(cls, div_type: DivisionType, value: str) -> str:
        """Форматирование отображаемого имени"""
        if div_type == DivisionType.UNKNOWN:
            return "не указана"
        
        names = {
            DivisionType.AXIS: "Ось",
            DivisionType.PODJEZD: "Подъезд",
            DivisionType.LITER: "Литер",
            DivisionType.SECTION: "Секция",
            DivisionType.BUILDING: "Корпус",
            DivisionType.FLOOR: "Этаж",
            DivisionType.WING: "Крыло",
        }
        
        return f"{names.get(div_type, '')} {value}"
    

class LLMEngine:
   
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.device = None
        
        self.generation_config = {
            'max_new_tokens': 150,
            'temperature': 0.3,
            'do_sample': True,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'pad_token_id': None,
            'eos_token_id': None
        }
        
        logger.info(f"Инициализация LLM Engine. GPU доступен: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def load(self):
    
        if self.loaded:
            return
        
        logger.info(f"Загрузка LLM: {self.model_name}")
        t_start = time.time()
        
        try:
            self.device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"Используется устройство: {self.device}")
            
           
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.use_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_cache=True,
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model.to("cpu")
            
            self.generation_config['pad_token_id'] = self.tokenizer.pad_token_id
            self.generation_config['eos_token_id'] = self.tokenizer.eos_token_id
            
            t_end = time.time()
            self.loaded = True
            
            if self.use_gpu:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                cached = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"GPU память: выделено {allocated:.1f} GB, закэшировано {cached:.1f} GB")
            
            logger.info(f"LLM загружена за {t_end - t_start:.1f} сек")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки LLM: {e}")
            raise
    
    def generate(self, prompt: str) -> Optional[str]:
   
        if not self.loaded:
            self.load()
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_response(response)
            response = self._normalize_response(response)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response if response and len(response) >= 10 else None
            
        except torch.cuda.OutOfMemoryError:
            logger.error("Ошибка: недостаточно GPU памяти")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(f"Ошибка генерации LLM: {e}")
            return None
    
    def _extract_response(self, response: str) -> str:
        markers = ["Ответ:", "ОТВЕТ", "Ответ"]
        for marker in markers:
            if marker in response:
                response = response.split(marker)[-1].strip()
                break
        
        # Удаляем повторяющиеся части
        if len(response) > 200:
            sentences = re.split(r'[.!?]', response)
            if sentences:
                response = sentences[0].strip()
        
        # Удаляем эмодзи
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"
                                u"\U0001F300-\U0001F5FF"
                                u"\U0001F680-\U0001F6FF"
                                u"\U0001F1E0-\U0001F1FF"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        response = emoji_pattern.sub('', response)
        
        return response
    
    def _normalize_response(self, response: str) -> str:
        if not response:
            return "Дефектов не выявлено."
        
        # Удаляем всё, что похоже на диалог
        response = re.sub(r'(?:^|\n)\s*(?:Заказчик|Вопрос|Ответ|Обновление ответа|Объяснение)\s*[:.]\s*', '', response, flags=re.IGNORECASE)
        
        # Удаляем маркеры ---, **, : :
        response = re.sub(r'---', '', response)
        response = re.sub(r'\*\*', '', response)
        response = re.sub(r':\s*:', ':', response)
        
        # Удаляем цифры и единицы измерения
        response = re.sub(r'\d+[,.]?\d*', '', response)
        response = re.sub(r'\s*[мс][м]?\s*', ' ', response)
        response = re.sub(r'\s*%\s*', ' ', response)
        
        # Удаляем мусорные фразы
        garbage = [
            r'правильный ответ', r'это правильный ответ',
            r'в данном случае', r'можно заключить',
            r'следует отметить', r'необходимо отметить',
            r'обращает на себя внимание',
        ]
        for word in garbage:
            response = re.sub(word, '', response, flags=re.IGNORECASE)
        
        # Убираем лишние пробелы
        response = ' '.join(response.split())
        
        # Если пусто - дефолт
        if len(response) < 5:
            return "Дефектов не выявлено."
        
        # Добавляем точку
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response[:300]

    def clear_cache(self):
      
        if self.use_gpu:
            torch.cuda.empty_cache()
            logger.info("GPU кэш очищен")



class DescriptionGenerator:
    """Генератор описаний для всех статусов с использованием LLM"""
    
    def __init__(self, llm_engine: Optional[LLMEngine] = None):
        self.llm = llm_engine
        self.template_count = 0
        self.llm_count = 0
        
        self.defect_keywords = [
            'трещин', 'разруш', 'коррози', 'деформ', 'скол', 'износ',
            'протечк', 'увлажн', 'высол', 'отслоени', 'выветриван',
            'сквозн', 'обруш', 'прогиб', 'гниль', 'ржавч', 'поврежд',
            'дефект', 'осадк', 'выкрашиван', 'отклонен', 'осыпани',
            'выветриван', 'замачиван', 'каверн'
        ]
    
    def generate(self, construct: Dict[str, Any], use_llm: bool = False, context: str = "") -> Tuple[str, str]:
        """Генерация описания - ПЫТАЕМСЯ ИСПОЛЬЗОВАТЬ LLM ДЛЯ ВСЕХ СТАТУСОВ"""
        
        construct_name = construct.get('Конструкция', 'неизвестно')
        status = construct.get('Статус', 'НЕ ОПРЕДЕЛЕНО')
        citation = construct.get('Цитата', '')
        division = construct.get('Разделение', 'не указана')
        
        # ПЫТАЕМСЯ ИСПОЛЬЗОВАТЬ LLM ДЛЯ ВСЕХ СТАТУСОВ, ЕСЛИ ОН ДОСТУПЕН
        if use_llm and self.llm and self.llm.loaded:
            # Сначала проверяем, есть ли дефекты в цитате
            defects = self._extract_defects_from_citation(citation)
            if defects:
                # Если есть дефекты - используем их
                emoji = self._get_emoji(status)
                return f"{emoji} {construct_name} ({division}): {defects[:150]}", 'template'
            
            # Пробуем LLM для любого статуса
            description = self._generate_with_llm(construct_name, status, citation, division, context)
            if description:
                self.llm_count += 1
                return description, 'llm'
        
        # Если LLM не сработал - используем шаблон
        self.template_count += 1
        description = self._generate_template(construct_name, status, citation, division)
        return description, 'template'

    def _get_emoji(self, status: str) -> str:
        """Возвращает эмодзи для статуса"""
        emojis = {
            'АВАРИЙНОЕ': '🔴',
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': '🟡',
            'РАБОТОСПОСОБНОЕ': '🔵',
            'НОРМАТИВНОЕ': '🟢'
        }
        return emojis.get(status, '⚪')

    def _generate_template(self, construct: str, status: str, citation: str, division: str) -> str:
        location = f" ({division})" if division and division != 'не указана' else ""
        emoji = self._get_emoji(status)
        
        if not citation or len(citation) < 10:
            return self._get_status_description(construct, status, location, emoji)
        
        # Чистим цитату
        clean_citation = self._extract_defects_from_citation(citation)
        
        # Если после чистки остался мусор (капс, много цифр) — используем шаблон
        if clean_citation:
            # Проверяем на мусор
            is_garbage = False
            
            # Капс (3+ заглавных подряд)
            if re.search(r'[А-Я]{3,}', clean_citation):
                is_garbage = True
            
            # Много цифр подряд
            if re.search(r'\d{4,}', clean_citation):
                is_garbage = True
            
            # Строка начинается с цифры
            if re.match(r'^\d', clean_citation):
                is_garbage = True
            
            # Всё это - "Ведомость" и "Паспорт"
            garbage_phrases = ['ведомость', 'паспорт', 'приложение', 'таблица']
            for phrase in garbage_phrases:
                if phrase in clean_citation.lower():
                    is_garbage = True
                    break
            
            if not is_garbage and len(clean_citation) > 15:
                return f"{emoji} {construct}{location}: {clean_citation[:200]}"
        
        # Если мусор — используем шаблон по статусу
        return self._get_status_description(construct, status, location, emoji)

    def _get_status_description(self, construct: str, status: str, location: str, emoji: str) -> str:
        """Шаблонные описания по статусу"""
        descriptions = {
            'АВАРИЙНОЕ': f"{emoji} {construct}{location} в АВАРИЙНОМ состоянии! Требуется НЕМЕДЛЕННОЕ усиление.",
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': f"{emoji} {construct}{location} требует ремонта или усиления.",
            'РАБОТОСПОСОБНОЕ': f"{emoji} {construct}{location} в работоспособном состоянии.",
            'НОРМАТИВНОЕ': f"{emoji} {construct}{location} в нормативном состоянии. Дефектов не выявлено."
        }
        return descriptions.get(status, f"{emoji} {construct}{location} состояние не определено.")


    def _extract_defects_from_citation(self, citation: str) -> str:
        if not citation or len(citation) < 10:
            return ""
        
        # ===== 1. ОБРЕЗАЕМ ПО ВСЕМ МУСОРНЫМ ПАТТЕРНАМ =====
        cut_patterns = [
            # Спецсимволы
            r'---', r'\*\*', r':\s*:', r'=\s*=', r'\_\_', r'\|\|', r'##',
            # Фото/рисунки/таблицы
            r'Фото\s*[А-Я]?\s*\d+', r'Рис\.?\s*\d+', r'Табл\.?\s*\d+',
            r'Вид\s+поперечных\s+балок', r'Вид\s+[А-Я]',
            # Мусорные слова
            r'ГОСТ\s*[\d\.-]+', r'СНиП\s*[\d\.-]+', r'СП\s*[\d\.-]+',
            r'Общество с ограниченной ответственностью', r'ООО\s*«[^»]+»',
            r'Подпись\s*Дата', r'Изм\s*Лист', r'п/п', r'№ п/п',
            r'Ведомость дефектов и повреждений', r'Паспорт здания',
            r'Приложение', r'Таблица',
            # ТАБЛИЧНЫЕ ДАННЫЕ (главное!)
            r'\d+\s+[А-Я]\s+[А-Я]',  # "1 А В" - таблица
            r'\d+[,.]\d+\s+\d+[,.]\d+\s+\d+[,.]\d+',  # "9,97 163,0 198,0"
            r'\d{4,}',  # 4+ цифр подряд (ГОСТ, 8239-56)
            r'\(\d+;\s*\d+\)',  # "(72; 89)"
            r'\d+\s+\d+\s+\d+\s+\d+',  # "240 115 3460 289"
        ]
        
        for pattern in cut_patterns:
            match = re.search(pattern, citation, re.IGNORECASE)
            if match:
                citation = citation[:match.start()]
        
        # ===== 2. УБИРАЕМ ОСТАВШИЕСЯ ЦИФРЫ В НАЧАЛЕ/КОНЦЕ =====
        citation = re.sub(r'^\d+\s*', '', citation)
        citation = re.sub(r'\s+\d+$', '', citation)
        
        # ===== 3. ИСПРАВЛЯЕМ СЛИПШИЕСЯ СЛОВА =====
        replacements = {
            'сводовспереборкойотдельных': 'сводов с переборкой отдельных',
            'участковкладки': 'участков кладки',
            'усилениебалок': 'усиление балок',
            'наличиевременных': 'наличие временных',
            'Перекрытиенад': 'Перекрытие над',
            'мэтажом': 'м этажом',
            'этажейото': 'этажей ',
            'по осиекомендации': 'по оси рекомендации',
            '2004екомендации': 'рекомендации',
        }
        for old, new in replacements.items():
            citation = citation.replace(old, new)
        
        # ===== 4. ЧИСТИМ ПРОБЕЛЫ =====
        citation = re.sub(r'\s+', ' ', citation).strip()
        
        # ===== 5. ЕСЛИ ОСТАЛОСЬ МАЛО - ПУСТО =====
        if len(citation) < 15:
            return ""
        
        # ===== 6. БЕРЕМ ПЕРВОЕ ПРЕДЛОЖЕНИЕ =====
        sentences = re.split(r'[.!?]', citation)
        for sent in sentences[:2]:
            sent = sent.strip()
            if len(sent) > 15:
                if re.match(r'^[А-Я][а-я]+ состояние', sent, re.IGNORECASE):
                    continue
                if re.match(r'^категория', sent, re.IGNORECASE):
                    continue
                if re.match(r'^на основании', sent, re.IGNORECASE):
                    continue
                if re.match(r'^согласно', sent, re.IGNORECASE):
                    continue
                return sent[:200]
        
        return ""

    def _generate_with_llm(self, construct: str, status: str, citation: str, division: str, context: str) -> Optional[str]:
        if not self.llm or not self.llm.loaded:
            return None
        
        clean_citation = self._extract_defects_from_citation(citation)
        if not clean_citation or len(clean_citation) < 10:
            return None
        
        prompt = self._build_prompt(construct, status, clean_citation, division, context)
        if not prompt:
            return None
        
        try:
            response = self.llm.generate(prompt)
            if response:
                # Проверяем, что ответ не содержит мусора
                if re.search(r'(?:Заказчик|Вопрос|Ответ|Обновление|Объяснение)', response, re.IGNORECASE):
                    logger.warning(f"LLM вернул мусор: {response[:50]}...")
                    return None
                emoji = self._get_emoji(status)
                return f"{emoji} {response}"
            return None
        except Exception as e:
            logger.error(f"Ошибка LLM: {e}")
            return None


    def _build_prompt(self, construct: str, status: str, citation: str, division: str, context: str) -> str:
        if not citation or len(citation.strip()) < 10:
            return ""
        
        # Чистим цитату от мусора
        clean_citation = re.sub(r'\d+[,.]?\d*\s*[мс][м]?\s*', '', citation)
        clean_citation = re.sub(r'\d+[,.]?\d*\s*%', '', clean_citation)
        clean_citation = re.sub(r'\d+', '', clean_citation)
        clean_citation = re.sub(r'[А-Я]{4,}', '', clean_citation)
        clean_citation = clean_citation[:150]
        
        status_text = {
            'АВАРИЙНОЕ': 'аварийном (критическое, угроза обрушения)',
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': 'ограниченно-работоспособном (требует ремонта)',
            'РАБОТОСПОСОБНОЕ': 'работоспособном (нормальное)',
            'НОРМАТИВНОЕ': 'нормативном (без дефектов)',
        }.get(status, status)
        
        location = f"Локализация: {division}" if division and division != 'не указана' else ""
        
        # ПРОМПТ БЕЗ ДИАЛОГА!
        prompt = f"""Опиши состояние конструкции "{construct}" одним коротким предложением (10-20 слов).

    Статус: {status_text}
    {location}

    Конкретная информация из отчета: "{clean_citation}"

    Правила:
    - Пиши ТОЛЬКО описание состояния
    - НЕ используй слова "Ответ:", "Заказчик:", "Вопрос:"
    - НЕ используй эмодзи и маркеры ---, **
    - Если есть дефекты - опиши их (трещины, коррозия, разрушение)
    - Если дефектов нет - напиши "Дефектов не выявлено"
    - Одно предложение, 10-20 слов

    Описание:"""
        return prompt


    def get_stats(self) -> Dict[str, int]:
        return {
            'template_count': self.template_count,
            'llm_count': self.llm_count,
            'total_count': self.template_count + self.llm_count
        }

class ReportAnalyzer:
    """Основной класс анализатора"""
    
    def __init__(
        self,
        use_llm: bool = True,
        use_gpu: bool = True,
        use_llm_for_limited: bool = False,
        db_path: str = "data/reports.db"
    ):
        self.use_llm = use_llm
        self.use_gpu = use_gpu
        self.use_llm_for_limited = use_llm_for_limited
        
        self.db = Database(db_path)
        self.rag = RAGSystem()
        self.citation_selector = CitationSelector()
        self.division_extractor = DivisionExtractor()
        
        self.llm_engine = None
        if self.use_llm:
            self.llm_engine = LLMEngine(use_gpu=self.use_gpu)
        
        # Генератор описаний
        self.generator = DescriptionGenerator(self.llm_engine)
        
        logger.info(f"ReportAnalyzer инициализирован")
        logger.info(f"  - Использовать LLM: {self.use_llm}")
        logger.info(f"  - RAG загружен: {len(self.rag.records)} конструкций")
    
    def _create_generator(self):
        """Создание генератора описаний"""
        return DescriptionGenerator(self.llm_engine)
    
    def analyze_report(self, pdf_path: str, collect_all_citations: bool = True) -> Dict[str, Any]:
        """Анализ отчета с улучшенным выбором цитат"""
        logger.info(f"Начало анализа: {pdf_path}")
        t_start_overall = time.time()
        
        logger.info("Шаг 1: Извлечение текста и ML-классификация")
        t_start = time.time()
        
        analysis = analyze_pdf(pdf_path, use_ml=True, collect_all_citations=collect_all_citations)
        priority = calculate_priority(analysis)
        
        t_end = time.time()
        logger.info(f"  Найдено: {analysis['total_constructs']} конструкций")
        logger.info(f"  Приоритет: {priority['priority']}")
        logger.info(f"  Время: {t_end - t_start:.1f} сек")
        
        
        logger.info("Шаг 2: Улучшение цитат и извлечение разделений")
        constructs = analysis['all_constructs']
        enhanced_constructs = []
        
        for construct in constructs:
            # Получаем все цитаты
            all_citations = construct.get('all_citations', [construct.get('Цитата', '')])
            
            # Выбираем лучшую цитату
            best_citation, top_citations = self.citation_selector.select_best_citation(all_citations)
            
            # Если лучшая цитата пустая, используем текущую
            if not best_citation:
                best_citation = construct.get('Цитата', '')
            
            # Обновляем конструкцию
            construct['Цитата'] = best_citation
            construct['all_citations'] = top_citations
            
            # Извлекаем разделение из лучшей цитаты
            div_type, div_value, div_display = self.division_extractor.extract_primary(best_citation)
            construct['Тип_разделения'] = div_type.value
            construct['Разделение'] = div_display
            
            enhanced_constructs.append(construct)
        
        logger.info("Шаг 3: Генерация описаний для конструкций")
        
        # Загружаем LLM если нужно
        if self.use_llm and self.llm_engine and not self.llm_engine.loaded:
            logger.info("Загрузка LLM...")
            t_start = time.time()
            self.llm_engine.load()
            t_end = time.time()
            logger.info(f"LLM загружена за {t_end - t_start:.1f} сек")
        
        rag_total_time = 0
        generation_total_time = 0
        llm_count = 0
        template_count = 0
        
        for i, construct in enumerate(enhanced_constructs):
            if (i + 1) % 10 == 0:
                logger.info(f"  Прогресс: {i+1}/{len(enhanced_constructs)}")
            
            # RAG поиск
            t_start_rag = time.time()
            similar = self._search_rag(construct)
            t_end_rag = time.time()
            rag_total_time += t_end_rag - t_start_rag
            
            rag_context = self._format_rag_context(similar)
            
            # Генерация описания
            status = construct.get('Статус', '')
            use_llm_for_this = (
                self.use_llm and 
                self.llm_engine is not None and 
                self.llm_engine.loaded #and
                #(status == 'АВАРИЙНОЕ' or 
                # (status == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ' and self.use_llm_for_limited))
            )
            
            t_start_gen = time.time()
            description, method = self.generator.generate(construct, use_llm_for_this, rag_context)
            t_end_gen = time.time()
            generation_total_time += t_end_gen - t_start_gen
            
            if method == 'llm':
                llm_count += 1
            else:
                template_count += 1
            
            construct['llm_description'] = description
            if description and method == 'llm':
                logger.info(f"LLM для {construct.get('Конструкция')}: {description[:50]}...")
            else:
                logger.info(f"Шаблон для {construct.get('Конструкция')}")
            construct['generation_method'] = method
        
        analysis['all_constructs'] = enhanced_constructs

        for c in analysis['all_constructs']:
            citation = c.get('Цитата', '')
            reinforcement = extract_reinforcement(citation)
            c['Усиление'] = reinforcement.get('description', '')
            c['Год_усиления'] = reinforcement.get('year', '')

        division_summary = self._build_division_summary(enhanced_constructs)
        
        t_end_overall = time.time()
        
        stats = {
            'total_time': t_end_overall - t_start_overall,
            'rag_time': rag_total_time,
            'generation_time': generation_total_time,
            'ml_time': t_end - t_start,
            'llm_count': llm_count,
            'template_count': template_count
        }
        
        logger.info(f"\n=== СТАТИСТИКА ===")
        logger.info(f"Общее время: {stats['total_time']:.1f} сек")
        logger.info(f"  - ML классификация: {stats['ml_time']:.1f} сек")
        logger.info(f"  - RAG поиск: {stats['rag_time']:.1f} сек")
        logger.info(f"  - Генерация: {stats['generation_time']:.1f} сек")
        logger.info(f"  - LLM вызовов: {llm_count}")
        logger.info(f"  - Шаблонов: {template_count}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'analysis': analysis,
            'priority': priority,
            'stats': stats,
            'division_summary': division_summary
        }
    

    
    def _search_rag(self, construct: Dict[str, Any]) -> List[Any]:
        try:
            name = construct.get('Конструкция', '')
            status = construct.get('Статус', '')
            citation = construct.get('Цитата', '')
            query = f"{name} {status} {citation[:200]}"
            return self.rag.search(query, top_k=3)
        except Exception as e:
            logger.warning(f"Ошибка RAG: {e}")
            return []
    
    def _format_rag_context(self, similar_results: List[Any]) -> str:
        if not similar_results:
            return ""
        
        context = "Похожие конструкции из базы:\n"
        for j, s in enumerate(similar_results[:3]):
            context += f"{j+1}. {s.construct} — {s.status}\n"
            if hasattr(s, 'defect') and s.defect:
                context += f"   Дефект: {s.defect[:150]}\n"
        
        return context
    
    def _build_division_summary(self, constructs: List[Dict]) -> Dict[str, Dict[str, int]]:
        summary = defaultdict(lambda: {
            'total': 0, 'danger': 0, 'limited': 0, 'working': 0, 'normative': 0, 'has_collapse': False
        })
        
        for c in constructs:
            division = c.get('Разделение', 'не указана')
            status = c.get('Статус', '')
            
            summary[division]['total'] += 1
            if status == 'АВАРИЙНОЕ':
                summary[division]['danger'] += 1
            elif status == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ':
                summary[division]['limited'] += 1
            elif status == 'РАБОТОСПОСОБНОЕ':
                summary[division]['working'] += 1
            else:
                summary[division]['normative'] += 1
            
            if c.get('Угроза_обрушения', False):
                summary[division]['has_collapse'] = True
        
        return dict(summary)
    
    def save_report_to_db(self, report_data: Dict[str, Any]) -> str:
        """Сохранение отчета в базу"""
        constructs = report_data.get('analysis', {}).get('all_constructs', [])
        priority = report_data.get('priority', {})
        
        status_counts = defaultdict(int)
        for c in constructs:
            status_counts[c.get('Статус', 'НЕ ОПРЕДЕЛЕНО')] += 1
        
        db_data = {
            'hash': report_data.get('report_hash', ''),
            'building_id': report_data.get('building_id', 'unknown'),
            'filename': report_data.get('filename', 'unknown'),
            'date': report_data.get('date', ''),
            'priority': priority.get('priority', ''),
            'health_index': priority.get('health_index', 0),
            'reason': priority.get('reason', ''),
            'has_collapse': report_data.get('analysis', {}).get('has_collapse', False),
            'has_court_order': report_data.get('has_court_order', False),
            'total_constructs': len(constructs),
            'status_counts': dict(status_counts),
            'constructs': constructs
        }
        
        return self.db.save_report(db_data)

def print_detailed_report(result: Dict[str, Any]):
    analysis = result['analysis']
    priority = result['priority']
    constructs = analysis['all_constructs']
    division_summary = result.get('division_summary', {})
   
    print(f"\nОБЩИЙ ВЕРДИКТ:")
    print(f"  Приоритет: {priority['priority']}")
    print(f"  Коэффициент состояния: {priority.get('health_index', 0):.1f}%")
    print(f"  Причина: {priority['reason']}")
    
    print(f"\nСТАТИСТИКА:")
    print(f"  Всего конструкций: {len(constructs)}")
    
    status_counts = defaultdict(int)
    for c in constructs:
        status_counts[c.get('Статус', '')] += 1
    
    print(f"  🟢 Нормативных: {status_counts.get('НОРМАТИВНОЕ', 0)}")
    print(f"  🔵 Работоспособных: {status_counts.get('РАБОТОСПОСОБНОЕ', 0)}")
    print(f"  🟡 Ограниченных: {status_counts.get('ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 0)}")
    print(f"  🔴 Аварийных: {status_counts.get('АВАРИЙНОЕ', 0)}")
    
    print(f"\n{'='*70}")
    print("ВСЕ КОНСТРУКЦИИ:")
    print(f"{'='*70}")
    
    # Сортируем по статусу (аварийные сверху)
    status_order = {'АВАРИЙНОЕ': 0, 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': 1, 'РАБОТОСПОСОБНОЕ': 2, 'НОРМАТИВНОЕ': 3}
    sorted_constructs = sorted(constructs, key=lambda x: status_order.get(x.get('Статус', ''), 4))
    
    for c in sorted_constructs:
        name = c.get('Конструкция', '')
        status = c.get('Статус', '')
        division = c.get('Разделение', 'не указана')
        description = c.get('llm_description', '')
        
        emoji = {
            'АВАРИЙНОЕ': '🔴',
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': '🟡',
            'РАБОТОСПОСОБНОЕ': '🔵',
            'НОРМАТИВНОЕ': '🟢'
        }.get(status, '⚪')
        
        print(f"\n{emoji} {name}")
        print(f"    Статус: {status}")
        print(f"    Локализация: {division}")
        if description:
            print(f"    Описание: {description}")
        else:
            print(f"    Описание: (нет описания)")
        if c.get('Угроза_обрушения', False):
            print(f"    ⚠️ ЕСТЬ УГРОЗА ОБРУШЕНИЯ!")
    
    print(f"\n{'='*70}")
    print(f"\nРАЗДЕЛЕНИЯ:")
    for div, stats in division_summary.items():
        emoji = "🚨" if stats['has_collapse'] else "🔴" if stats['danger'] > 0 else "🟡" if stats['limited'] > 0 else "🟢"
        print(f"\n  {emoji} {div}:")
        print(f"    Всего: {stats['total']}")
        print(f"    🔴 Аварийных: {stats['danger']}")
        print(f"    🟡 Ограниченных: {stats['limited']}")
        print(f"    🔵 Работоспособных: {stats['working']}")
        print(f"    🟢 Нормативных: {stats['normative']}")
        if stats['has_collapse']:
            print(f"   Есть угроза обрушения!")
    
    danger_constructs = [c for c in constructs if c.get('Статус') == 'АВАРИЙНОЕ']
    if danger_constructs:
        print(f"\n🔴 АВАРИЙНЫЕ КОНСТРУКЦИИ ({len(danger_constructs)}):")
        for c in danger_constructs:
            print(f"\n  - {c.get('Конструкция')}")
            print(f"    Локализация: {c.get('Разделение', 'не указана')}")
            if c.get('llm_description'):
                print(f"    Описание: {c.get('llm_description')}")
            if c.get('Угроза_обрушения'):
                print(f"   Есть угроза обрушения!")
    
    print(f"\n{'='*70}")

def main():
    use_gpu = torch.cuda.is_available()
    logger.info(f"CUDA доступен: {use_gpu}")
    
    analyzer = ReportAnalyzer(
        use_llm=True,
        use_gpu=use_gpu,
        use_llm_for_limited=False
    )
    
    # ===== СПИСОК ФАЙЛОВ ДЛЯ АНАЛИЗА =====
    files_to_analyze = [
        "1. МАГДАГАЧИ, ул. М. Горького, д. 7.pdf",
        "1.ВЕРХНЕЗЕЙСК, д. 65 (скан) (1).pdf",
        "13. 14_92.ТО.2020_ул. Южная, д.14 (2).pdf",
        "163.24.СТЭ.2019 Михнево, 9 Мая, 1.pdf",
        "1796_5_ЗАКЛЮЧЕНИЕ 25 дом испр.pdf",
        "1796_2_ЗАКЛЮЧЕНИЕ 6 дом испр.pdf",
        "1796_3_ЗАКЛЮЧЕНИЕ 7 дом испр.pdf",
        "otchetvenchyr.pdf",
        "oychetlihack.pdf"
    ]
    
    results = []
    
    for filename in files_to_analyze:
        pdf_path = Path("reports") / filename
        
        if not pdf_path.exists():
            logger.error(f"Файл не найден: {pdf_path}")
            continue
        
        
        try:
            result = analyzer.analyze_report(str(pdf_path))
            results.append({
                'filename': filename,
                'result': result
            })
            
            # Выводим краткий результат
            priority = result['priority']
            constructs = result['analysis']['all_constructs']
            logger.info(f"{filename}: {priority['priority']} ({len(constructs)} конструкций)")
            
            # Сохраняем в БД
            report_data = {
                'report_hash': pdf_path.stem,
                'building_id': result.get('building_id', 'unknown'),
                'filename': filename,
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': result['analysis'],
                'priority': result['priority'],
                'has_court_order': False
            }
            report_id = analyzer.save_report_to_db(report_data)
            logger.info(f"Сохранен с ID: {report_id}")
            
        except Exception as e:
            logger.error(f"Ошибка при анализе {filename}: {e}")
            continue
    
    logger.info(f"Успешно обработано: {len(results)} из {len(files_to_analyze)} файлов")
    
    for r in results:
        priority = r['result']['priority']
        constructs = r['result']['analysis']['all_constructs']
        logger.info(f"  - {r['filename']}: {priority['priority']} ({len(constructs)} конструкций)")


if __name__ == "__main__":
    main()