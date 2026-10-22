import os
import json
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import pypdf
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ConstructRecord:
    
    address: str
    year: int
    construct: str
    category: str
    status: str
    defect: str
    citation: str
    source_file: str
    score: float = 0.0


class RAGSystem:
    
    CONSTRUCT_KEYWORDS = {
        'Фундамент': ['фундамент', 'свая', 'ростверк', 'плита', 'лента', 'подколонник'],
        'Стены': ['стена', 'колонна', 'пилон', 'пилястра', 'простенок', 'перемычка', 'стойка', 'опора'],
        'Перекрытия': ['плита', 'перекрытие', 'балка', 'ригель', 'прогон', 'консоль'],
        'Крыша': ['крыша', 'кровля', 'стропила', 'ферма'],
        'Лестницы': ['лестница', 'марш', 'площадка', 'косоур'],
    }
    
    STATUS_KEYWORDS = {
        'АВАРИЙНОЕ': ['аварийн', 'сквозн', 'обруш', 'разруш', 'критическ', 'угроза'],
        'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': ['ограничен', 'ремонт', 'усилени', 'коррози', 'дефект'],
        'РАБОТОСПОСОБНОЕ': ['работоспособн', 'удовлетворительн', 'обеспечен'],
        'НОРМАТИВНОЕ': ['нормативн', 'без замечаний', 'дефектов не']
    }
    
    def __init__(self, index_path: str = "rag_index"):
        self.index_path = Path(index_path)
        self.index_path.mkdir(exist_ok=True)
        
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.index = None
        self.records: List[ConstructRecord] = []
        self.documents: List[str] = []
        self.dim = 384
        
        self._load_index()
    
    def index_reports(self, pdf_dir: str) -> Dict[str, Any]:
       
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        print(f"Найдено {len(pdf_files)} отчетов")
        
        total_constructs = 0
        failed_files = []
        
        for i, pdf_path in enumerate(pdf_files):
            print(f"{i+1}/{len(pdf_files)}: {pdf_path.name}")
            
            constructs = self._parse_pdf(pdf_path)
            
            if not constructs:
                failed_files.append(pdf_path.name)
                continue
            
            for c in constructs:
                self.records.append(c)
                self.documents.append(self._format_document(c))
                total_constructs += 1
        
        print(f"Извлечено конструкций: {total_constructs}")
        if failed_files:
            print(f"Не удалось обработать: {failed_files}")
        
        if self.records:
            self._build_index()
            self._save_index()
        
        return {
            'total_files': len(pdf_files),
            'total_constructs': total_constructs,
            'failed_files': failed_files
        }
    
    def _parse_pdf(self, pdf_path: Path) -> List[ConstructRecord]:
    
        try:
        
            text = self._extract_text(pdf_path)
            if not text or len(text.strip()) < 100:
                return []
            
            
            address = self._extract_address(text)
            year = self._extract_year(text)
            
            
            records = []
            
            
            sections = self._split_sections(text)
            
            for section_name, section_text in sections.items():
                if len(section_text) < 30:
                    continue
                
                found = self._find_constructs_in_text(
                    section_text, 
                    section_name, 
                    address, 
                    year, 
                    pdf_path.name
                )
                records.extend(found)
            
            
            if not records:
                found = self._find_constructs_in_text(
                    text, 
                    "Общее", 
                    address, 
                    year, 
                    pdf_path.name
                )
                records.extend(found)
            
            return records
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return []
    
    def _extract_text(self, pdf_path: Path) -> str:
        
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"  Ошибка чтения: {e}")
        return text
    
    def _extract_address(self, text: str) -> str:
    
        patterns = [
            r'по адресу[:]?\s*([^,.!?]{10,120})',
            r'адрес[:]?\s*([^,.!?]{10,120})',
            r'расположен(?:ного)?\s*по адресу[:]?\s*([^,.!?]{10,120})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                addr = match.group(1).strip()
                addr = re.sub(r'\s+', ' ', addr)
                if len(addr) > 10:
                    return addr
        return "Неизвестно"
    
    def _extract_year(self, text: str) -> int:
        
        patterns = [
            r'Год\s+ввода.*?(\d{4})',
            r'год\s+постройки\s*[:]?\s*(\d{4})',
            r'год\s+возведения\s*[:]?\s*(\d{4})',
            r'(\d{4})\s*г(?:од)?\.',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0
    
    def _split_sections(self, text: str) -> Dict[str, str]:
        
        sections = {}
        current_section = "Общее"
        current_text = []
        
        section_patterns = {
            'Фундамент': r'(?i)(?:состояние\s+)?фундамент(?:а|ов)?',
            'Стены': r'(?i)(?:состояние\s+)?стен(?:ы)?',
            'Перекрытия': r'(?i)(?:состояние\s+)?перекрыт(?:ие|ий|ия)',
            'Крыша': r'(?i)(?:состояние\s+)?(?:кровл(?:я|и)|крыш(?:а|и))',
            'Лестницы': r'(?i)(?:состояние\s+)?лестниц(?:а|ы|и)',
        }
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            is_header = False
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    if current_text:
                        sections[current_section] = ' '.join(current_text)
                    current_section = section_name
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        if current_text:
            sections[current_section] = ' '.join(current_text)
        
        return sections
    
    def _find_constructs_in_text(self, text: str, category: str, 
                                 address: str, year: int, 
                                 source_file: str) -> List[ConstructRecord]:
        
        records = []
        sentences = re.split(r'[.!?;]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
        
            found_construct = None
            found_category = category
            
            for cat, keywords in self.CONSTRUCT_KEYWORDS.items():
                for kw in keywords:
                    if kw in sentence.lower():
                        found_construct = kw
                        found_category = cat
                        break
                if found_construct:
                    break
            
            if not found_construct:
                continue
            
            
            status = self._determine_status(sentence)
            
            
            defect = self._extract_defect(sentence)
            
            records.append(ConstructRecord(
                address=address,
                year=year,
                construct=found_construct,
                category=found_category,
                status=status,
                defect=defect,
                citation=sentence[:300],
                source_file=source_file
            ))
        
        return records
    
    def _determine_status(self, text: str) -> str:
    
        text_lower = text.lower()
        
        for status, keywords in self.STATUS_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return status
        
        return 'НОРМАТИВНОЕ'
    
    def _extract_defect(self, text: str) -> str:
        
        patterns = [
            r'(?:дефект(?:ы)?|повреждени[яе]|разрушени[яе]|трещин[ыа])\s*[^.!?]{10,100}',
            r'(?:выявлен[ыо]|обнаружен[ыо]|отмечен[ыо])\s+[^.!?]{10,100}',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        
        for sent in re.split(r'[.!?]', text):
            sent = sent.strip()
            if len(sent) > 20 and any(w in sent.lower() for w in ['трещин', 'разруш', 'дефект']):
                return sent[:200]
        
        return ""
    
    def _format_document(self, record: ConstructRecord) -> str:
        
        return f"""
Адрес: {record.address}
Год: {record.year}
Конструкция: {record.construct} ({record.category})
Статус: {record.status}
Дефект: {record.defect}
Цитата: {record.citation}
"""
    

    def _build_index(self):
    
        print(f"Генерация эмбеддингов для {len(self.documents)} документов...")
        embeddings = self.embedder.encode(self.documents, show_progress_bar=True)
        
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Индекс создан")
    
    def _save_index(self):
        
        faiss.write_index(self.index, str(self.index_path / "faiss_index.bin"))
        with open(self.index_path / "records.pkl", 'wb') as f:
            pickle.dump(self.records, f)
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"Индекс сохранен: {len(self.records)} записей")
    
    def _load_index(self):
        
        index_file = self.index_path / "faiss_index.bin"
        records_file = self.index_path / "records.pkl"
        docs_file = self.index_path / "documents.pkl"
        
        if index_file.exists() and records_file.exists() and docs_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(records_file, 'rb') as f:
                    self.records = pickle.load(f)
                with open(docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Индекс загружен: {len(self.records)} записей")
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
    

    def search(self, query: str, top_k: int = 10, 
               status_filter: Optional[str] = None,
               category_filter: Optional[str] = None) -> List[ConstructRecord]:
        """
        Поиск по запросу с фильтрацией.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            status_filter: Фильтр по статусу (АВАРИЙНОЕ, НОРМАТИВНОЕ...)
            category_filter: Фильтр по категории (Фундамент, Стены...)
        """
        if self.index is None or not self.records:
            return []
        
        
        query_emb = self.embedder.encode([query])
        
        
        k = min(top_k * 2, len(self.records))
        distances, indices = self.index.search(np.array(query_emb).astype('float32'), k)
        
        
        results = []
        for idx in indices[0]:
            if idx < len(self.records):
                record = self.records[idx]
        
                if status_filter and record.status != status_filter:
                    continue
                if category_filter and record.category != category_filter:
                    continue
                results.append(record)
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_by_status(self, status: str) -> List[ConstructRecord]:
        
        if not self.records:
            return []
        return [r for r in self.records if r.status == status]
    
    def search_by_category(self, category: str) -> List[ConstructRecord]:
        
        if not self.records:
            return []
        return [r for r in self.records if r.category == category]
    
    def search_by_address(self, address_query: str) -> List[ConstructRecord]:
      
        if not self.records:
            return []
        address_query = address_query.lower()
        return [r for r in self.records if address_query in r.address.lower()]
   
    def get_statistics(self) -> Dict[str, Any]:
    
        if not self.records:
            return {'total': 0}
        
        status_counts = defaultdict(int)
        category_counts = defaultdict(int)
        addresses = set()
        
        for r in self.records:
            status_counts[r.status] += 1
            category_counts[r.category] += 1
            addresses.add(r.address)
        
        return {
            'total': len(self.records),
            'buildings': len(addresses),
            'statuses': dict(status_counts),
            'categories': dict(category_counts)
        }
  
    def export_to_json(self, results: List[ConstructRecord]) -> str:

        return json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)
    
    def export_to_csv(self, results: List[ConstructRecord], filename: str = None) -> str:
        
        if not results:
            return ""
        
        fieldnames = ['address', 'year', 'construct', 'category', 'status', 'defect', 'source_file']
        
        output = ""
        import io
        with io.StringIO() as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'address': r.address,
                    'year': r.year,
                    'construct': r.construct,
                    'category': r.category,
                    'status': r.status,
                    'defect': r.defect,
                    'source_file': r.source_file
                })
            output = f.getvalue()
        
        if filename:
            with open(filename, 'w', encoding='utf-8-sig') as f:
                f.write(output)
        
        return output
    
    def export_summary(self, results: List[ConstructRecord]) -> str:
        
        if not results:
            return "Ничего не найдено"
        
        
        by_address = defaultdict(list)
        for r in results:
            by_address[r.address].append(r)
        
        lines = []
        for addr, items in by_address.items():
            lines.append(f"\n{addr}")
            
            
            statuses = defaultdict(int)
            for item in items:
                statuses[item.status] += 1
            
            status_str = ", ".join([f"{s}: {c}" for s, c in statuses.items()])
            lines.append(f"   Конструкций: {len(items)} ({status_str})")
            
        
            critical = [item for item in items if item.status == 'АВАРИЙНОЕ']
            if critical:
                lines.append(" Аварийные конструкции:")
                for item in critical:
                    defect = item.defect[:80] if item.defect else "нет описания"
                    lines.append(f"      - {item.construct}: {defect}...")
        
        return "\n".join(lines)



if __name__ == "__main__":
    import time
    
    REPORTS_DIR = r"C:\Users\sakura666\OneDrive\Desktop\practica\reports"
    
    
    rag = RAGSystem()
    
    
    if len(rag.records) == 0:
        print("\nИндекс пуст. Начинаем индексацию...")
        stats = rag.index_reports(REPORTS_DIR)
        print(f"\nРезультат: {stats}")
    
    
    stats = rag.get_statistics()
    
    print(f"Всего конструкций: {stats['total']}")
    print(f"Всего зданий: {stats['buildings']}")
    
    print("\nПо статусам:")
    for status, count in sorted(stats['statuses'].items(), key=lambda x: -x[1]):
        emoji = {'АВАРИЙНОЕ': '🔴', 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': '🟡',
                'РАБОТОСПОСОБНОЕ': '🔵', 'НОРМАТИВНОЕ': '🟢'}.get(status, '⚪')
        print(f"   {emoji} {status}: {count}")
    
    print("\nПо категориям:")
    for category, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"   {category}: {count}")
    
    print("Команды:")
    print("   поиск <текст>     - поиск по запросу")
    print("   аварийные         - все аварийные конструкции")
    print("   фундамент         - все конструкции фундамента")
    print("   статистика        - показать статистику")
    print("   экспорт <файл>    - экспорт в CSV")
    print("   выход             - выход")

    
    while True:
        try:
            cmd = input("\n> ").strip()
            if not cmd:
                continue
            
            if cmd.lower() in ['выход', 'exit', 'q']:
                break
            
            if cmd.lower().startswith('поиск '):
                query = cmd[6:].strip()
                start = time.time()
                results = rag.search(query, top_k=10)
                elapsed = time.time() - start
                print(f"\nПоиск занял: {elapsed*1000:.1f} мс")
                print(rag.export_summary(results))
            
            elif cmd.lower() == 'аварийные':
                results = rag.search_by_status('АВАРИЙНОЕ')
                print(rag.export_summary(results))
            
            elif cmd.lower() == 'фундамент':
                results = rag.search_by_category('Фундамент')
                print(rag.export_summary(results))
            
            elif cmd.lower() == 'статистика':
                stats = rag.get_statistics()
                print(f"\nВсего: {stats['total']}")
                print(f"Зданий: {stats['buildings']}")
                print(f"Статусы: {stats['statuses']}")
                print(f"Категории: {stats['categories']}")
            
            elif cmd.lower().startswith('экспорт '):
                filename = cmd[8:].strip()
                if not filename.endswith('.csv'):
                    filename += '.csv'
                all_results = rag.records
                rag.export_to_csv(all_results, filename)
                print(f"Экспортировано в {filename}")
            
            else:
            
                results = rag.search(cmd, top_k=5)
                print(rag.export_summary(results))
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")
    
    print("\n До свидания!")