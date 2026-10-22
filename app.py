import streamlit as st
import pandas as pd
import os
import tempfile
import io
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from main import analyze_pdf, calculate_priority
import re
from collections import defaultdict
from database import Database
from unified_analyzer_re import ReportAnalyzer
from main import calculate_priority
import torch


st.set_page_config(
    page_title="Анализ технического состояния зданий",
    layout="wide"
)

db = Database()

DATA_DIR = Path("data")
REPORTS_DIR = DATA_DIR / "reports"
SUMMARIES_DIR = DATA_DIR / "summaries"

for dir_path in [DATA_DIR, REPORTS_DIR, SUMMARIES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_analyzer():
    """Создание и кэширование анализатора"""
    use_gpu = torch.cuda.is_available()
    print("1. Создаем analyzer...")
    analyzer = ReportAnalyzer(
        use_llm=True,
        use_gpu=use_gpu,
        use_llm_for_limited=True
    )
    print("2. Analyzer создан, загружаем LLM...")
    # Принудительно загружаем LLM
    if analyzer.llm_engine:
        print("3. Загружаем LLM engine...")
        analyzer.llm_engine.load()
        print("4. LLM engine загружен!")
    print("5. Все готово!")
    return analyzer

analyzer = get_analyzer()

def clean_folder_name(name):
    if not name:
        return "unknown"
    invalid_chars = r'[<>:"/\\|?*;]'
    name = re.sub(invalid_chars, '_', name)
    name = re.sub(r'\s+', ' ', name)
    if len(name) > 100:
        name = name[:100]
    return name.strip()

def get_building_id(text):
    if not text:
        return "unknown"
    
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    address_patterns = [
        # "г. Ангарск, ул. Ленина, д. 17"
        r'(?:г\.\s*[А-Яа-я]+\s*[,.]?\s*(?:ул\.|улица|пр\.|проспект|пер\.|переулок|пл\.|площадь|наб\.|набережная|ш\.|шоссе|б-р|бульвар)\s*[А-Яа-я]+(?:[.,]\s*[А-Яа-я]+)?\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # "Ангарск, ул. Карла Маркса, д. 5"
        r'([А-Яа-я]+\s*[,.]?\s*(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+(?:[.,]\s*[А-Яа-я]+)?\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # "Иркутская область, г. Ангарск, ул. ..."
        r'(?:[А-Яа-я]+\s+область.*?г\.\s*[А-Яа-я]+\s*[,.]?\s*(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+(?:[.,]\s*[А-Яа-я]+)?\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # Просто "ул. Ленина, д. 5"
        r'(?:(?:ул\.|улица|пер\.|переулок|пр\.|проспект)\s*[А-Яа-я]+(?:[.,]\s*[А-Яа-я]+)?\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            addr = match.group(1).strip() if match.groups() else match.group(0).strip()
            addr = re.sub(r'\s+', ' ', addr)
            addr = re.sub(r',\s*$', '', addr)
            addr = re.sub(r',\s*,', ',', addr)
            if len(addr) > 5 and len(addr) < 150:
                if 'ул' in addr.lower() or 'д.' in addr.lower() or 'дом' in addr.lower() or 'пер' in addr.lower():
                    return clean_folder_name(addr[:100])
    
    city_patterns = [
        r'(?:г\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
        r'(?:город\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
        r'(?:пгт\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
        r'(?:р\.п\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
        r'(?:пос\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
    ]
    
    for pattern in city_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            if len(city) > 2 and city not in ['Россия', 'Российской', 'Федерации']:
                return clean_folder_name(city)
    
    first_lines = text[:300].strip()
    # Убираем названия организаций
    first_lines = re.sub(r'ООО\s+«[^»]+»', '', first_lines)
    first_lines = re.sub(r'[А-Я]{2,}\s*«[^»]+»', '', first_lines)
    first_lines = re.sub(r'Общество с ограниченной ответственностью', '', first_lines)
    first_lines = re.sub(r'\s+', ' ', first_lines).strip()
    
    if len(first_lines) > 10:
        return clean_folder_name(first_lines[:50])
    
    return "unknown"


def get_building_full_info(text):
    info = {
        'address': '',
        'cadastre': '',
        'name': '',
        'organization': '',
        'full': ''
    }
    
    if not text:
        return info
    
    # Ищем полный адрес
    address_patterns = [
        # "Амурская область, Магдагачинский округ, пгт. Магдагачи, ул. М. Горького, д. 7"
        r'(?:[А-Яа-я]+\s+область.*?(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # "г. Ангарск, ул. Ленина, д. 17"
        r'(?:г\.\s*[А-Яа-я]+\s*[,.]?\s*(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # "Ангарск, ул. Карла Маркса, д. 5"
        r'([А-Яа-я]+\s*[,.]?\s*(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
        
        # Просто "ул. Ленина, д. 5"
        r'(?:(?:ул\.|улица|пер\.|переулок)\s*[А-Яа-я]+\s*[,.]?\s*д\.?\s*[0-9]+[а-я]?)',
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            addr = match.group(1).strip() if match.groups() else match.group(0).strip()
            addr = re.sub(r'\s+', ' ', addr)
            addr = re.sub(r',\s*$', '', addr)
            if len(addr) > 5:
                info['address'] = addr
                break
    
    # Если адрес не найден - ищем город/поселок
    if not info['address']:
        city_patterns = [
            r'(?:г\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
            r'(?:пгт\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
            r'(?:р\.п\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
            r'(?:пос\.\s*)([А-Яа-я]+(?:-\s*[А-Яа-я]+)?)',
        ]
        for pattern in city_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                if len(city) > 2:
                    info['address'] = city
                    break
    
    # Формируем full ТОЛЬКО из адреса (БЕЗ ОРГАНИЗАЦИИ!)
    info['full'] = info['address'] if info['address'] else "Неизвестное здание"
    
    return info   

def get_report_hash(content):
    return hashlib.md5(content).hexdigest()[:12]


def save_pdf(pdf_content, building_id, report_hash):

    building_reports_dir = REPORTS_DIR / building_id
    building_reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = building_reports_dir / f"{report_hash}.pdf"
    with open(pdf_path, 'wb') as f:
        f.write(pdf_content)


def get_all_reports_for_building(building_id):
    
    return db.get_all_reports(building_id)


def format_summary(summary):
    lines = []
    lines.append(f"**Отчет:** {summary.get('filename', 'Неизвестно')}")
    lines.append(f"**Дата анализа:** {summary.get('date', 'Неизвестно')}")
    lines.append(f"**Приоритет:** {summary.get('priority', 'Не определен')}")
    lines.append(f"**Конструкций:** {summary.get('total_constructs', 0)}")
    statuses = summary.get('status_counts', {})
    if statuses:
        lines.append(f"   🟢 Нормативных: {statuses.get('НОРМАТИВНОЕ', 0)}")
        lines.append(f"   🔵 Работоспособных: {statuses.get('РАБОТОСПОСОБНОЕ', 0)}")
        lines.append(f"   🟡 Ограниченных: {statuses.get('ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 0)}")
        lines.append(f"   🔴 Аварийных: {statuses.get('АВАРИЙНОЕ', 0)}")
    if summary.get('has_collapse'):
        lines.append("**Обнаружена угроза обрушения!**")
    if summary.get('has_court_order'):
        lines.append("**Обнаружены судебные решения о расселении!**")
    if summary.get('reason'):
        lines.append(f"**Причина:** {summary['reason']}")
    return "\n".join(lines)

def get_report_safe(report_id):
    try:
        if not report_id:
            return None
        return db.load_report(report_id)
    except Exception as e:
        st.error(f"Ошибка загрузки отчета: {e}")
        return None

def show_comparison_two():
    st.subheader("Сравнение двух отчетов")
    
    building_id = st.session_state.get('comparison_building_id')
    if not building_id:
        st.warning("Не указано здание")
        return
    
    reports = db.get_all_reports(building_id)
    if len(reports) < 2:
        st.info("Нужно минимум 2 отчета")
        return
    
    options = []
    for r in reports:
        label = f"{r.get('filename', '')[:30]} ({r.get('date', '')[:10]})"
        options.append({'label': label, 'id': r.get('hash') or r.get('id', '')})
    
    col1, col2 = st.columns(2)
    with col1:
        idx1 = st.selectbox("Первый отчет", range(len(options)), format_func=lambda i: options[i]['label'], key="comp1")
    with col2:
        idx2 = st.selectbox("Второй отчет", range(len(options)), format_func=lambda i: options[i]['label'], key="comp2")
    
    if st.button("Сравнить", use_container_width=True):
        if idx1 == idx2:
            st.warning("Выберите разные отчеты")
            return
        
        report1 = get_report_safe(options[idx1]['id'])
        report2 = get_report_safe(options[idx2]['id'])
        
        if not report1 or not report2:
            st.error("Не удалось загрузить отчеты")
            return
        
        cons1 = {c.get('Конструкция', ''): c for c in report1.get('constructs', []) if c.get('Конструкция')}
        cons2 = {c.get('Конструкция', ''): c for c in report2.get('constructs', []) if c.get('Конструкция')}
        
        all_names = set(cons1.keys()) | set(cons2.keys())
        
        data = []
        improved = 0
        worsened = 0
        same = 0
        new_c = 0
        removed = 0
        
        priority = {'НОРМАТИВНОЕ': 0, 'РАБОТОСПОСОБНОЕ': 1, 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': 2, 'АВАРИЙНОЕ': 3}
        
        for name in all_names:
            c1 = cons1.get(name, {})
            c2 = cons2.get(name, {})
            
            s1 = c1.get('Статус', 'НЕТ')
            s2 = c2.get('Статус', 'НЕТ')
            
            if not c1 and c2:
                new_c += 1
                change = '🟢 Новая'
            elif c1 and not c2:
                removed += 1
                change = '🔴 Исчезла'
            elif s1 == s2:
                same += 1
                change = '➖ Без изменений'
            elif priority.get(s2, 0) > priority.get(s1, 0):
                worsened += 1
                change = '🔴 УХУДШИЛОСЬ'
            else:
                improved += 1
                change = '🟢 УЛУЧШИЛОСЬ'
            
            data.append({'Конструкция': name, 'Было': s1, 'Стало': s2, 'Изменение': change})
        
        df = pd.DataFrame(data)
        df = df.sort_values('Изменение', ascending=False)
        
        st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🟢 Улучшилось", improved)
        c2.metric("🔴 Ухудшилось", worsened)
        c3.metric("➖ Без изменений", same)
        c4.metric("🟢 Новые", new_c)
        c5.metric("🔴 Исчезли", removed)
        
        if worsened > 0:
            st.error(f"⚠️ {worsened} конструкций УХУДШИЛИСЬ!")
        elif improved > 0:
            st.success(f"✅ {improved} конструкций УЛУЧШИЛИСЬ!")
        else:
            st.info("➖ Изменений нет")

def show_comparison_all():
    st.subheader("Сравнение со всеми отчетами")
    
    building_id = st.session_state.get('all_comparison_building')
    if not building_id:
        st.warning("Не указано здание")
        return
    
    reports = db.get_all_reports(building_id)
    if len(reports) < 2:
        st.info("Нужно минимум 2 отчета")
        return
    
    all_cons = {}
    dates = []
    
    for r in reports:
        date = r.get('date', '')
        dates.append(date)
        report_id = r.get('hash') or r.get('id', '')
        full = get_report_safe(report_id)
        if full:
            for c in full.get('constructs', []):
                name = c.get('Конструкция', '')
                if name:
                    if name not in all_cons:
                        all_cons[name] = {}
                    all_cons[name][date] = c.get('Статус', '')
    
    if not all_cons:
        st.info("Нет данных для сравнения")
        return
    
    # Таблица
    rows = []
    for name, statuses in all_cons.items():
        row = {'Конструкция': name}
        for i, date in enumerate(sorted(dates)):
            row[f'Отчет {i+1}'] = statuses.get(date, '-')
        rows.append(row)
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=400)
    
    # Статистика
    st.markdown("---")
    st.markdown("### Статистика по отчетам")
    
    for i, r in enumerate(reports):
        sc = r.get('status_counts', {})
        if not sc:
            report_id = r.get('hash') or r.get('id', '')
            full = get_report_safe(report_id)
            if full:
                sc = full.get('status_counts', {})
        
        cols = st.columns(5)
        cols[0].metric(f"Отчет {i+1}", r.get('filename', '')[:20])
        cols[1].metric("🟢 Нормативные", sc.get('НОРМАТИВНОЕ', 0))
        cols[2].metric("🔵 Работоспособные", sc.get('РАБОТОСПОСОБНОЕ', 0))
        cols[3].metric("🟡 Ограниченно-работоспособные", sc.get('ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 0))
        cols[4].metric("🔴 Аварийные", sc.get('АВАРИЙНОЕ', 0))

def compare_reports(summary1, summary2):
    result = {
        'has_changes': False,
        'priority_changed': False,
        'collapse_changed': False,
        'status_changes': {},
        'total_constructs_change': 0,
        'summary': [],
        'recommendation': ''
    }
    
    p1 = summary1.get('priority', 'НЕ ИЗВЕСТНО')
    p2 = summary2.get('priority', 'НЕ ИЗВЕСТНО')
    priority_order = {'НИЗКИЙ': 0, 'СРЕДНИЙ': 1, 'ВЫСОКИЙ': 2, 'КРИТИЧЕСКОЕ': 3}
    p1_score = priority_order.get(p1, -1)
    p2_score = priority_order.get(p2, -1)
    
    if p1 != p2:
        result['priority_changed'] = True
        result['has_changes'] = True
        if p2_score > p1_score:
            result['summary'].append(f"Приоритет УХУДШИЛСЯ: {p1} → {p2}")
        else:
            result['summary'].append(f"Приоритет УЛУЧШИЛСЯ: {p1} → {p2}")
    else:
        result['summary'].append(f"➖ Приоритет не изменился: {p1}")
    
    c1 = summary1.get('has_collapse', False)
    c2 = summary2.get('has_collapse', False)
    if c1 != c2:
        result['collapse_changed'] = True
        result['has_changes'] = True
        if c2 and not c1:
            result['summary'].append("ПОЯВИЛАСЬ угроза обрушения!")
        else:
            result['summary'].append("Угроза обрушения ИСЧЕЗЛА")
    else:
        result['summary'].append(f"➖ Угроза обрушения: {'Есть' if c1 else 'Нет'} (без изменений)")
    
    s1 = summary1.get('status_counts', {})
    s2 = summary2.get('status_counts', {})
    all_statuses = set(s1.keys()) | set(s2.keys())
    status_changes = {}
    for status in all_statuses:
        count1 = s1.get(status, 0)
        count2 = s2.get(status, 0)
        diff = count2 - count1
        if diff != 0:
            status_changes[status] = diff
            result['has_changes'] = True
    result['status_changes'] = status_changes
    
    if status_changes:
        for status, diff in status_changes.items():
            emoji = {'НОРМАТИВНОЕ': '🟢', 'РАБОТОСПОСОБНОЕ': '🔵', 
                     'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': '🟡', 'АВАРИЙНОЕ': '🔴'}.get(status, '⚪')
            if status == 'АВАРИЙНОЕ' and diff > 0:
                result['summary'].append(f"{emoji} {status}: увеличилось на {diff} (УХУДШЕНИЕ!)")
            elif status == 'НОРМАТИВНОЕ' and diff < 0:
                result['summary'].append(f"{emoji} {status}: уменьшилось на {abs(diff)} (УХУДШЕНИЕ!)")
            elif status == 'АВАРИЙНОЕ' and diff < 0:
                result['summary'].append(f"{emoji} {status}: уменьшилось на {abs(diff)} (УЛУЧШЕНИЕ!)")
            elif status == 'НОРМАТИВНОЕ' and diff > 0:
                result['summary'].append(f"{emoji} {status}: увеличилось на {diff} (УЛУЧШЕНИЕ!)")
            else:
                result['summary'].append(f"   {emoji} {status}: {'↑' if diff > 0 else '↓'} ({abs(diff)})")
    
    total1 = summary1.get('total_constructs', 0)
    total2 = summary2.get('total_constructs', 0)
    result['total_constructs_change'] = total2 - total1
    if total2 != total1:
        result['has_changes'] = True
        direction = "увеличилось" if total2 > total1 else "уменьшилось"
        result['summary'].append(f"Всего конструкций: {direction} с {total1} до {total2}")
    
    if not result['has_changes']:
        result['recommendation'] = "Состояние здания не изменилось между отчетами"
    else:
        worsening = False
        if result['priority_changed'] and p2_score > p1_score:
            worsening = True
        if result['collapse_changed'] and c2 and not c1:
            worsening = True
        if status_changes.get('АВАРИЙНОЕ', 0) > 0:
            worsening = True
        if status_changes.get('НОРМАТИВНОЕ', 0) < 0:
            worsening = True
        
        if worsening:
            result['recommendation'] = "СОСТОЯНИЕ УХУДШИЛОСЬ! Требуется внимание!"
        else:
            result['recommendation'] = "СОСТОЯНИЕ УЛУЧШИЛОСЬ или стабильно"
    
    return result



def extract_collapse_description(text):
    if not text:
        return None
    keywords = ['обрушение', 'угроза', 'аварийное', 'сквозная', 'критическое']
    for kw in keywords:
        if kw in text.lower():
            sentences = re.split(r'[.!?]', text)
            for s in sentences:
                if kw in s.lower():
                    return s.strip()
    return None


def get_construct_comment(row):
    status = row.get('Статус', '')
    name = row.get('Конструкция', '')
    if status == 'АВАРИЙНОЕ':
        return f"Конструкция «{name}» в АВАРИЙНОМ состоянии! Требуется НЕМЕДЛЕННОЕ усиление."
    elif status == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ':
        return f"Конструкция «{name}» требует ремонта или усиления."
    elif status == 'РАБОТОСПОСОБНОЕ':
        return f"Конструкция «{name}» в работоспособном состоянии."
    else:
        return f"Конструкция «{name}» в нормативном состоянии."


def check_court_order(text):
    if not text:
        return False
    
    court_patterns = [
        r'судебн[ое]?\s+решени[ея]',
        r'решение\s+суда',
        r'постановлени[ея]\s+суда',
        r'вступивш[ее]?\s+в\s+силу',
        r'незамедлительн[ое]?\s+расселени[ея]',
        r'суд\s+обязал',
        r'исков[ое]?\s+заявлени[ея]',
        r'апелляционн[ое]?\s+определени[ея]',
        r'судебный\s+акт',
        r'судебное\s+предписание',
    ]
    
    text_lower = text.lower()
    for pattern in court_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False



if 'stats' not in st.session_state:
    st.session_state.stats = {"processed": 0, "total_constructs": 0, "danger_constructs": 0}
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'compare_reports' not in st.session_state:
    st.session_state.compare_reports = []
if 'force_reanalyze' not in st.session_state:
    st.session_state.force_reanalyze = False
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'text_preview' not in st.session_state:
    st.session_state.text_preview = ""
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'show_detail' not in st.session_state:
    st.session_state.show_detail = False
if 'detail_idx' not in st.session_state:
    st.session_state.detail_idx = 0
if 'show_all_comparison' not in st.session_state:
    st.session_state.show_all_comparison = False
if 'all_comparison_building' not in st.session_state:
    st.session_state.all_comparison_building = None
if 'show_two_comparison' not in st.session_state:
    st.session_state.show_two_comparison = False
if 'show_two' not in st.session_state:
    st.session_state.show_two = False
if 'show_all' not in st.session_state:
    st.session_state.show_all = False
if 'comparison_building_id' not in st.session_state:
    st.session_state.comparison_building_id = None


st.title("Анализ технического состояния зданий")

col_left, col_right = st.columns([1, 2])

with col_left:
    uploaded_files = st.file_uploader("Выберите PDF-файл", type=['pdf'], accept_multiple_files=True)
    analyze_btn = st.button("Анализировать", type="primary", use_container_width=True, disabled=not uploaded_files)
    
    if st.button("Очистить результат", use_container_width=True):
        st.session_state.analysis_result = None
        st.session_state.comparison_mode = False
        st.session_state.compare_reports = []
        st.session_state.force_reanalyze = False
        st.rerun()


if analyze_btn and uploaded_files:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results_summary = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Обработка {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        
        try:
            pdf_content = uploaded_file.getvalue()
            report_hash = get_report_hash(pdf_content)
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
            
            import pypdf
            full_text = ""
            text_preview = ""
            
            
            try:
                with open(tmp_path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text_preview += page_text[:500]
                        full_text += page_text
            except Exception as e:
                print(f"pypdf ошибка: {e}")
            
            
            if len(full_text.strip()) < 500:
                try:
                    import pytesseract
                    from PIL import Image
                    import fitz
                    
                    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                    
                    doc = fitz.open(tmp_path)
                    for page_num in range(min(len(doc), 30)):
                        page = doc[page_num]
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text = pytesseract.image_to_string(img, lang='rus')
                        full_text += page_text + " "
                        text_preview += page_text[:500]
                    doc.close()
                    print("  OCR отработал для определения building_id")
                except Exception as e:
                    print(f" OCR ошибка: {e}")
            
            building_id = get_building_id(text_preview)
            st.session_state.text_preview = full_text[:2000]
            
            existing = db.get_report_by_hash(report_hash, building_id)
            
            if existing:
            
                st.session_state.all_results.append({
                    'type': 'existing',
                    'summary': existing,
                    'building_id': building_id,
                    'report_hash': report_hash,
                    'filename': uploaded_file.name
                })
                st.session_state.analysis_result = {
                    'type': 'existing',
                    'summary': existing,
                    'building_id': building_id,
                    'report_hash': report_hash
                }
                results_summary.append(f"{uploaded_file.name} — уже был в системе")
            else:
            
                has_court = check_court_order(full_text)
                
                result = analyzer.analyze_report(tmp_path, collect_all_citations=True)

                analysis = result['analysis']
                priority = result['priority']
                df = pd.DataFrame(analysis['all_constructs'])
                status_counts = defaultdict(int)
                for c in analysis['all_constructs']:
                    status_counts[c.get('Статус', 'НЕ ОПРЕДЕЛЕНО')] += 1

                # Добавляем информацию о судебных решениях
                has_court = check_court_order(full_text)

                # Обновляем приоритет если есть судебное решение
                if has_court and priority.get('priority') != 'КРИТИЧЕСКОЕ':
                    priority = {
                        'priority': 'КРИТИЧЕСКОЕ',
                        'color': '🔴',
                        'health_index': 0,
                        'reason': 'Судебное решение о расселении'
                    }
                
                df = pd.DataFrame(analysis['all_constructs'])
                status_counts = defaultdict(int)
                for c in analysis['all_constructs']:
                    status_counts[c['Статус']] += 1
                
        
                summary = {
                    'hash': report_hash,
                    'building_id': building_id,
                    'filename': uploaded_file.name,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_constructs': analysis['total_constructs'],
                    'status_counts': dict(status_counts),
                    'has_collapse': analysis['has_collapse'],
                    'has_court_order': has_court,
                    'priority': priority['priority'],
                    'reason': priority['reason'],
                    'health_index': priority.get('health_index', 0)
                }
                
            
                save_pdf(pdf_content, building_id, report_hash)
                
                
                report_data = {
                    'hash': report_hash,
                    'building_id': building_id,
                    'filename': uploaded_file.name,
                    'date': summary['date'],
                    'priority': priority['priority'],
                    'health_index': priority.get('health_index', 0),
                    'reason': priority.get('reason', ''),
                    'has_collapse': analysis.get('has_collapse', False),
                    'has_court_order': has_court,
                    'total_constructs': analysis['total_constructs'],
                    'status_counts': dict(status_counts),
                    'constructs': analysis['all_constructs']
                }
                db.save_report(report_data)
                
                st.session_state.all_results.append({
                    'type': 'new',
                    'analysis': analysis,
                    'priority': priority,
                    'building_id': building_id,
                    'report_hash': report_hash,
                    'summary': summary,
                    'df': df,
                    'filename': uploaded_file.name
                })

                st.session_state.analysis_result = {
                    'type': 'new',
                    'analysis': analysis,
                    'priority': priority,
                    'building_id': building_id,
                    'report_hash': report_hash,
                    'summary': summary,
                    'df': df
                }
                
                st.session_state.stats["processed"] += 1
                st.session_state.stats["total_constructs"] += analysis['total_constructs']
                danger_count = len([c for c in analysis['all_constructs'] if c['Статус'] == 'АВАРИЙНОЕ'])
                st.session_state.stats["danger_constructs"] += danger_count
                
                results_summary.append(f"{uploaded_file.name} — проанализирован")
            
            os.unlink(tmp_path)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            results_summary.append(f"{uploaded_file.name} — ошибка: {str(e)[:50]}")
            continue
    
    status_text.text("Обработка завершена!")
    for line in results_summary:
        if line.startswith("✅"):
            st.success(line)
        else:
            st.error(line)
    
    st.info(f"Обработано {len(uploaded_files)} файлов")

    st.subheader("Все обработанные отчеты")
    for res in st.session_state.all_results:
        if res.get('type') == 'new':
            st.success(f"{res.get('filename', 'Неизвестно')} — {res.get('priority', {}).get('priority', 'Н/Д')}")
        else:
            st.info(f"{res.get('filename', 'Неизвестно')} — уже был в системе")
    st.rerun()





if st.session_state.get('show_detail', False):
    if 'comparison_building_id' not in st.session_state:
        st.session_state.comparison_building_id = None

    detail_idx = st.session_state.get('detail_idx', 0)
    if detail_idx < len(st.session_state.all_results):
        res = st.session_state.all_results[detail_idx]
        
        st.subheader(f"Детальный отчет: {res.get('filename', 'Неизвестно')}")
        
        if st.button("Вернуться к списку отчетов", use_container_width=True):
            st.session_state.show_detail = False
            st.rerun()
        
        st.divider()
        
        analysis = res.get('analysis')
        priority = res.get('priority')
        df = res.get('df')
        summary = res.get('summary', {})
        
        if df is None or len(df) == 0:
            st.warning("В отчете не найдено конструкций для анализа.")
            st.info("Возможные причины:\n- Отчет не содержит текста о конструкциях\n- Формат PDF не распознан\n- Файл поврежден")
            st.stop()
        
        if 'Статус' not in df.columns:
            st.warning("В отчете не найдены конструкции с присвоенным статусом.")
            st.stop()
        
        

        st.caption(f"Анализ: {summary.get('date', 'Неизвестно')}")
        
        st.subheader("Общий вердикт по зданию")
        
        total = len(df)
        danger = len(df[df['Статус'] == 'АВАРИЙНОЕ'])
        limited = len(df[df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
        working = len(df[df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
        normative = len(df[df['Статус'] == 'НОРМАТИВНОЕ'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = priority.get('priority', 'Н/Д') if priority else 'Н/Д'
            emoji = {'КРИТИЧЕСКОЕ': '🔴', 'ВЫСОКИЙ': '🟠', 'СРЕДНИЙ': '🟡', 'НИЗКИЙ': '🟢'}.get(p, '⚪')
            st.markdown(f"## {emoji} **{p}**")
            if p == 'КРИТИЧЕСКОЕ':
                st.error("ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ РАССЕЛЕНИЕ!")
        
        with col2:
            st.markdown(f"**Всего конструкций:** {total}")
            st.markdown(f"**Аварийных:** {danger} 🔴")
            st.markdown(f"**Ограниченных:** {limited} 🟡")
            st.markdown(f"**Причина:** {priority.get('reason', 'Н/Д') if priority else 'Н/Д'}")
        
        with col3:
            health_index = priority.get('health_index', 0) if priority else 0
            st.markdown("### Общий коэффициент состояния")
            if health_index >= 80:
                st.success(f"### {health_index:.1f}%")
            elif health_index >= 50:
                st.warning(f"### {health_index:.1f}%")
            else:
                st.error(f"### {health_index:.1f}%")
        
        st.divider()
        
        st.subheader("Состояние конструкций по осям")
        
        axes = [ax for ax in df['Разделение'].unique() if ax and ax != 'не указана' and len(ax) > 1]
        no_axis_df = df[df['Разделение'] == 'не указана']
        
        if not axes:
            axes = ['Все здание']
            no_axis_df = pd.DataFrame()
        
        for axis in axes:
            axis_df = df if axis == 'Все здание' else df[df['Разделение'] == axis]
            if len(axis_df) == 0:
                continue
            
            total_a = len(axis_df)
            danger_a = len(axis_df[axis_df['Статус'] == 'АВАРИЙНОЕ'])
            limited_a = len(axis_df[axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
            working_a = len(axis_df[axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
            normative_a = len(axis_df[axis_df['Статус'] == 'НОРМАТИВНОЕ'])
            
            has_collapse = any('обруш' in str(c).lower() or 'угроз' in str(c).lower() for c in axis_df['Цитата'])
            
            if has_collapse:
                axis_icon = "🚨 "
            elif danger_a > 0:
                axis_icon = "🔴 "
            elif limited_a > 0:
                axis_icon = "🟡 "
            elif working_a > 0:
                axis_icon = "🔵 "
            else:
                axis_icon = "🟢 "
            
            with st.expander(f"{axis_icon}Ось {axis} — {total_a} конструкций", expanded=has_collapse or danger_a > 0):
                cols = st.columns(5)
                cols[0].metric("Всего", total_a)
                cols[1].metric("🟢 Нормативное", normative_a)
                cols[2].metric("🔵 Работоспособное", working_a)
                cols[3].metric("🟡 Ограниченно-работоспособное", limited_a)
                cols[4].metric("🔴 Аварийное", danger_a)
                
                if has_collapse:
                    st.error("УГРОЗА ОБРУШЕНИЯ!")
                    for _, row in axis_df.iterrows():
                        if row['Статус'] == 'АВАРИЙНОЕ':
                            desc = extract_collapse_description(row['Цитата'])
                            if desc:
                                st.warning(f"Описание: {desc}")
                                break
                
                st.markdown("---")
                st.markdown("#### Конструкции по категориям")
                
                categories = ['Фундамент', 'Вертикальные', 'Перекрытия', 'Крыша', 'Лестница']
                
                for category in categories:
                    cat_df = axis_df[axis_df['Категория'] == category]
                    if len(cat_df) == 0:
                        continue
                    
                    cat_total = len(cat_df)
                    cat_danger = len(cat_df[cat_df['Статус'] == 'АВАРИЙНОЕ'])
                    cat_limited = len(cat_df[cat_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
                    cat_working = len(cat_df[cat_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
                    cat_normative = len(cat_df[cat_df['Статус'] == 'НОРМАТИВНОЕ'])
                    
                    cat_has_collapse = any('обруш' in str(c).lower() or 'угроз' in str(c).lower() for c in cat_df['Цитата'])
                    
                    if cat_has_collapse:
                        cat_icon = "🚨"
                    elif cat_danger > 0:
                        cat_icon = "🔴"
                    elif cat_limited > 0:
                        cat_icon = "🟡"
                    elif cat_working > 0:
                        cat_icon = "🔵"
                    else:
                        cat_icon = "🟢"
                    
                    with st.expander(f"{cat_icon} {category} — {cat_total} конструкций"):
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Всего", cat_total)
                        c2.metric("🟢 Нормативное", cat_normative)
                        c3.metric("🔵 Работоспособное", cat_working)
                        c4.metric("🟡 Ограниченно-работоспособное", cat_limited)
                        c5.metric("🔴 Аварийное", cat_danger)
                        
                        if cat_has_collapse:
                            st.warning("Угроза обрушения!")
                        
                        st.markdown("---")
                        
                        danger_rows = cat_df[cat_df['Статус'] == 'АВАРИЙНОЕ']
                        if len(danger_rows) > 0:
                            st.markdown("##### Аварийные конструкции")
                            for _, row in danger_rows.iterrows():
                                st.markdown(f"**{row['Конструкция']}**")
                                st.caption(f"Ось: {row['Разделение']}")
                                st.caption(f"{row['Цитата'][:250]}..." if len(row['Цитата']) > 250 else row['Цитата'])
                                # Получаем описание
                                llm_desc = row.get('llm_description', '')
                                if llm_desc and len(llm_desc) > 10:
                                    st.error(f"🔴 {llm_desc}")
                                else:
                                    st.error(get_construct_comment(row))
                                st.divider()
                        
                        limited_rows = cat_df[cat_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ']
                        if len(limited_rows) > 0:
                            with st.expander(f"Ограниченные ({len(limited_rows)})"):
                                for _, row in limited_rows.iterrows():
                                    st.markdown(f"**{row['Конструкция']}**")
                                    st.caption(f"Ось: {row['Разделение']}")
                                    st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                                    llm_desc = row.get('llm_description', '')
                                    if llm_desc and len(llm_desc) > 10:
                                        st.warning(f"🟡 {llm_desc}")
                                    else:
                                        st.info(get_construct_comment(row))
                                    st.divider()
                        
                        working_rows = cat_df[cat_df['Статус'] == 'РАБОТОСПОСОБНОЕ']
                        if len(working_rows) > 0:
                            with st.expander(f"Работоспособные ({len(working_rows)})"):
                                for _, row in working_rows.iterrows():
                                    st.markdown(f"**{row['Конструкция']}**")
                                    st.caption(f"Ось: {row['Разделение']}")
                                    st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                                    llm_desc = row.get('llm_description', '')
                                    if llm_desc and len(llm_desc) > 10:
                                        st.info(f"🔵 {llm_desc}")
                                    else:
                                        st.markdown(f"**{row['Конструкция']}** в работоспособном состоянии.")
                                    st.divider()
                        
                        norm_rows = cat_df[cat_df['Статус'] == 'НОРМАТИВНОЕ']
                        if len(norm_rows) > 0:
                            with st.expander(f"Нормативные ({len(norm_rows)})"):
                                for _, row in norm_rows.iterrows():
                                    st.markdown(f"**{row['Конструкция']}**")
                                    st.caption(f"Ось: {row['Разделение']}")
                                    st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                                    llm_desc = row.get('llm_description', '')
                                    if llm_desc and len(llm_desc) > 10:
                                        st.success(f"🟢 {llm_desc}")
                                    else:
                                        st.markdown(f"**{row['Конструкция']}** в нормативном состоянии.")
                                    st.divider()
        
        st.divider()
        
        if len(no_axis_df) > 0:
            st.subheader("Конструкции без привязки к осям")
            st.caption("Эти конструкции указаны в отчете, но их расположение по осям не определено")
            
            total_na = len(no_axis_df)
            danger_na = len(no_axis_df[no_axis_df['Статус'] == 'АВАРИЙНОЕ'])
            limited_na = len(no_axis_df[no_axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
            working_na = len(no_axis_df[no_axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
            normative_na = len(no_axis_df[no_axis_df['Статус'] == 'НОРМАТИВНОЕ'])
            
            cols = st.columns(5)
            cols[0].metric("Всего", total_na)
            cols[1].metric("🟢 Нормативное", normative_na)
            cols[2].metric("🔵 Работоспособное", working_na)
            cols[3].metric("🟡 Ограниченное-работоспособное ", limited_na)
            cols[4].metric("🔴 Аварийное", danger_na)
            
        
        
            danger_na_rows = no_axis_df[no_axis_df['Статус'] == 'АВАРИЙНОЕ']
            if len(danger_na_rows) > 0:
                st.warning(f"⚠️ {len(danger_na_rows)} аварийных конструкций без указания оси!")
                for _, row in danger_na_rows.iterrows():
                    with st.expander(f"🔴 {row['Конструкция']} — {row['Категория']}", expanded=True):
                        st.markdown(f"**Статус:** {row['Статус']}")
                        st.markdown(f"**Категория:** {row['Категория']}")
                        st.markdown(f"**Цитата:**")
                        st.code(row['Цитата'][:500] + "..." if len(row['Цитата']) > 500 else row['Цитата'], language="text")
                        
                        # ЗДЕСЬ МЕНЯЕМ:
                        llm_desc = row.get('llm_description', '')
                        if llm_desc and len(llm_desc) > 10:
                            st.error(f"🔴 {llm_desc}")
                        else:
                            st.error(get_construct_comment(row))
                        
                        if row.get('Усиление') and row['Усиление'] not in ['', 'nan', 'None']:
                            st.info(f"**Усиление:** {row['Усиление']}")
                        if 'обруш' in str(row['Цитата']).lower() or 'угроз' in str(row['Цитата']).lower():
                            st.error("🚨 ОБНАРУЖЕНА УГРОЗА ОБРУШЕНИЯ!")
                            desc = extract_collapse_description(row['Цитата'])
                            if desc:
                                st.markdown(f"**Описание угрозы:** {desc}")
            
            
            limited_na_rows = no_axis_df[no_axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ']
            if len(limited_na_rows) > 0:
                with st.expander(f"🟡 Ограниченные без оси ({len(limited_na_rows)})", expanded=False):
                    for _, row in limited_na_rows.iterrows():
                        st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                        st.caption(f"{row['Цитата'][:200]}..." if len(row['Цитата']) > 200 else row['Цитата'])
                        
                        # ЗДЕСЬ МЕНЯЕМ:
                        llm_desc = row.get('llm_description', '')
                        if llm_desc and len(llm_desc) > 10:
                            st.warning(f"🟡 {llm_desc}")
                        else:
                            st.info(get_construct_comment(row))
                        
                        if row.get('Усиление') and row['Усиление'] not in ['', 'nan', 'None']:
                            st.info(f"**Усиление:** {row['Усиление']}")
                        st.divider()
            
            
            working_na_rows = no_axis_df[no_axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ']
            if len(working_na_rows) > 0:
                with st.expander(f"🔵 Работоспособные без оси ({len(working_na_rows)})", expanded=False):
                    for _, row in working_na_rows.iterrows():
                        st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                        # Добавляем LLM описание
                        llm_desc = row.get('llm_description', '')
                        if llm_desc and len(llm_desc) > 10:
                            st.info(f"🔵 {llm_desc}")
                        else:
                            st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                        st.divider()

            norm_na_rows = no_axis_df[no_axis_df['Статус'] == 'НОРМАТИВНОЕ']
            if len(norm_na_rows) > 0:
                with st.expander(f"🟢 Нормативные без оси ({len(norm_na_rows)})", expanded=False):
                    for _, row in norm_na_rows.iterrows():
                        st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                        # Добавляем LLM описание
                        llm_desc = row.get('llm_description', '')
                        if llm_desc and len(llm_desc) > 10:
                            st.success(f"🟢 {llm_desc}")
                        else:
                            st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                        st.divider()

        st.divider()

        # Показываем усиление, если есть
        if row.get('Усиление') and str(row['Усиление']) not in ['', 'nan', 'None']:
            st.info(f"**Усиление:** {row['Усиление']}")
        if row.get('Год_усиления') and str(row['Год_усиления']) not in ['', 'nan', 'None']:
            st.caption(f"Год усиления: {row['Год_усиления']}")

        # Показываем судебные решения
        if summary.get('has_court_order', False):
            st.error("**ЕСТЬ СУДЕБНОЕ РЕШЕНИЕ О НЕЗАМЕДЛИТЕЛЬНОМ РАССЕЛЕНИИ!**")
        
        st.subheader("Рекомендации по действиям")
        
        p = priority.get('priority', 'Н/Д') if priority else 'Н/Д'
        if p == 'КРИТИЧЕСКОЕ':
            st.error("""
            ### НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:
            1. **Эвакуировать жильцов** из здания
            2. **Ограничить доступ** в здание
            3. **Провести внеочередное обследование**
            4. **Разработать проект** сноса/усиления
            """)
        elif p == 'ВЫСОКИЙ':
            st.warning("""
            ### ПРИОРИТЕТНЫЕ ДЕЙСТВИЯ:
            1. Включить в программу расселения **2025-2026**
            2. Провести детальное обследование аварийных конструкций
            3. Разработать проект усиления
            """)
        elif p == 'СРЕДНИЙ':
            st.info("""
            ### ПЛАНОВЫЕ ДЕЙСТВИЯ:
            1. Включить в программу расселения **2027-2028**
            2. Провести плановый ремонт
            3. Установить мониторинг состояния
            """)
        else:
            st.success("""
            ### ТЕКУЩЕЕ ОБСЛУЖИВАНИЕ:
            1. Плановое техническое обслуживание
            2. Периодический мониторинг состояния
            """)
    
    st.stop()

if st.session_state.get('show_two', False):
    show_comparison_two()
    if st.button("Закрыть", use_container_width=True):
        st.session_state.show_two = False
        st.rerun()
    st.stop()

if st.session_state.get('show_all', False):
    show_comparison_all()
    if st.button("Закрыть", use_container_width=True):
        st.session_state.show_all = False
        st.rerun()
    st.stop()
        

if st.session_state.all_results:
    st.subheader(f"Результаты ({len(st.session_state.all_results)} отчетов)")
    
    summary_data = []
    for idx, res in enumerate(st.session_state.all_results):
        summary = res.get('summary', {})
        summary_data.append({
            '№': idx + 1,
            'Файл': res.get('filename', 'Неизвестно'),
            'Приоритет': summary.get('priority', 'Н/Д'),
            'Конструкций': summary.get('total_constructs', 0),
            'Аварийных': summary.get('status_counts', {}).get('АВАРИЙНОЕ', 0),
            'Угроза': '⚠️' if summary.get('has_collapse') else '—'
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    for idx, res in enumerate(st.session_state.all_results):
        summary = res.get('summary', {})
        building_id = res.get('building_id', 'Неизвестно')
        building_info = get_building_full_info(st.session_state.get('text_preview', ''))
        full_name = building_info.get('full', building_id)
        
        priority_emoji = {'КРИТИЧЕСКОЕ': '🔴', 'ВЫСОКИЙ': '🟠', 
                          'СРЕДНИЙ': '🟡', 'НИЗКИЙ': '🟢'}.get(summary.get('priority'), '⚪')
        
        with st.expander(
            f"{priority_emoji} Отчет {idx+1}: {res.get('filename', 'Неизвестно')} — {summary.get('priority', 'Н/Д')}",
            expanded=(idx == len(st.session_state.all_results) - 1)  
        ):
            st.caption(f"Анализ: {summary.get('date', 'Неизвестно')}")
            
            df = res.get('df')
            
            if df is not None and len(df) > 0 and 'Статус' in df.columns:

                total = len(df)
                normative = len(df[df['Статус'] == 'НОРМАТИВНОЕ'])
                working = len(df[df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
                limited = len(df[df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
                danger = len(df[df['Статус'] == 'АВАРИЙНОЕ'])
                
                cols = st.columns(5)
                cols[0].metric("Всего", total)
                cols[1].metric("🟢 Нормативное", normative)
                cols[2].metric("🔵 Работоспособное", working)
                cols[3].metric("🟡 Ограниченно", limited)
                cols[4].metric("🔴 Аварийное", danger)
                
                if summary.get('has_collapse'):
                    st.error("Обнаружена угроза обрушения!")
            else:
                #st.warning("В отчете не найдено конструкций для анализа")
                #st.caption("Возможные причины: неподдерживаемый формат или пустой отчет")
        
                cols = st.columns(5)
                cols[0].metric("Всего", summary.get('total_constructs', 0))
                cols[1].metric("🟢 Нормативное", summary.get('status_counts', {}).get('НОРМАТИВНОЕ', 0))
                cols[2].metric("🔵 Работоспособное", summary.get('status_counts', {}).get('РАБОТОСПОСОБНОЕ', 0))
                cols[3].metric("🟡 Ограниченно-работоспособное", summary.get('status_counts', {}).get('ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 0))
                cols[4].metric("🔴 Аварийное", summary.get('status_counts', {}).get('АВАРИЙНОЕ', 0))
            
            if st.button(f"Показать детали отчета {idx+1}", key=f"detail_{idx}"):
                st.session_state.show_detail = True
                st.session_state.detail_idx = idx
                st.session_state.analysis_result = res
                st.rerun()
    
    st.divider()


        

if st.session_state.analysis_result is not None:
    result = st.session_state.analysis_result
    #building_info = get_building_full_info(st.session_state.get('text_preview', ''))
    #building_id = result.get('building_id', 'Неизвестно')
        

    #full_building_name = building_info.get('full', building_id)

    #display_name = building_id.replace('_', ', ') if building_id else "Неизвестное здание"
    #st.info(f"**Здание:** {display_name}")

    filename = result.get('filename', 'Неизвестное здание')
    # Убираем расширение .pdf
    building_display = filename.replace('.pdf', '').replace('.PDF', '')
    # Если имя файла слишком длинное, можно обрезать
    if len(building_display) > 80:
        building_display = building_display[:77] + '...'
    #st.info(f"**Здание:** {building_display}")

    if building_info.get('cadastre'):
        st.caption(f"Кадастровый номер: {building_info['cadastre']}")
        st.info(f"**Здание:** {building_id}")
    
    # Найти этот блок в коде (примерно строка 850-900) и заменить:

    if result.get('type') == 'existing':
        summary = result['summary']
        
        st.subheader("Сохраненное саммари отчета")
        st.markdown(format_summary(summary))
        
        st.divider()
    
        
        # Кнопки для сравнения
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Заново проанализировать", use_container_width=True):
                db.delete_report(result.get('report_hash'))
                st.session_state.all_results = [r for r in st.session_state.all_results if r.get('report_hash') != result.get('report_hash')]
                st.session_state.analysis_result = None
                st.session_state.force_reanalyze = True
                st.rerun()
        
        with col2:
            if st.button("Сравнить два отчета", use_container_width=True):
                st.session_state.show_two = True
                st.session_state.comparison_building_id = building_id
                st.rerun()
        
        with col3:
            if st.button("Сравнить со всеми", use_container_width=True):
                st.session_state.show_all = True
                st.session_state.all_comparison_building = building_id
                st.rerun()
        
        st.stop()

    analysis = result.get('analysis')
    priority = result.get('priority')
    df = result.get('df')
    
    if analysis is None or df is None:
        st.error("Ошибка: данные не загружены")
        st.stop()
    
    with st.expander("Сохраненное саммари отчета", expanded=False):
        summary = result.get('summary', {})
        st.markdown(format_summary(summary))
    
        all_reports = get_all_reports_for_building(building_id)
        if len(all_reports) > 1:
            st.info(f"По этому зданию найдено {len(all_reports)} отчетов")
            
            current_hash = result.get('report_hash')
            other_reports = [r for r in all_reports if r.get('hash') != current_hash]
            
            if other_reports:
                report_options = []
                for report in other_reports:
                    report_options.append({
                        'label': f"{report.get('filename', '')} ({report.get('date', '')})",
                        'hash': report.get('hash'),
                        'building_id': building_id
                    })
                
                selected = st.selectbox(
                    "Выберите отчет для сравнения",
                    options=[(i, opt['label']) for i, opt in enumerate(report_options)],
                    format_func=lambda x: x[1],
                    key="compare_with_current"
                )
                
                if st.button("Сравнить с текущим отчетом", use_container_width=True):
                    st.session_state.compare_reports = [
                        {'hash': current_hash, 'building_id': building_id},
                        report_options[selected[0]]
                    ]
                    st.session_state.comparison_mode = True
                    st.rerun()

            with st.expander("Предыдущие отчеты по этому зданию"):
                for report in all_reports:
                    if report.get('hash') != current_hash:
                        with st.container():
                            st.markdown(f"**{report.get('filename', 'Неизвестно')}**")
                            st.caption(f"{report.get('date', '')}")
                            st.caption(f"Приоритет: {report.get('priority', 'Не определен')}")
                            st.divider()
    st.divider()
    
    st.subheader("Общий вердикт по зданию")

    col1, col2, col3 = st.columns(3)  

    with col1:
        emoji = {'КРИТИЧЕСКОЕ': '🔴', 'ВЫСОКИЙ': '🟠', 'СРЕДНИЙ': '🟡', 'НИЗКИЙ': '🟢'}.get(priority['priority'], '⚪')
        st.markdown(f"## {emoji} **{priority['priority']}**")
        if priority['priority'] == 'КРИТИЧЕСКОЕ':
            st.error("ТРЕБУЕТСЯ НЕМЕДЛЕННОЕ РАССЕЛЕНИЕ!")

    with col2:
        total = len(df)
        danger = len(df[df['Статус'] == 'АВАРИЙНОЕ'])
        limited = len(df[df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
        st.markdown(f"**Всего конструкций:** {total}")
        st.markdown(f"**Аварийных:** {danger} 🔴")
        st.markdown(f"**Ограниченных:** {limited} 🟡")
        st.markdown(f"**Причина:** {priority['reason']}")

    with col3:
        health_index = priority.get('health_index', 0)
        st.markdown("### Общий коэффициент состояния здания")
        if health_index >= 80:
            st.success(f"### {health_index:.1f}%")
            st.caption("🟢 Хорошее состояние")
        elif health_index >= 50:
            st.warning(f"### {health_index:.1f}%")
            st.caption("🟡 Удовлетворительное состояние")
        else:
            st.error(f"### {health_index:.1f}%")
            st.caption("🔴 Требуется внимание")
        
        # Показываем судебные решения
        if summary.get('has_court_order', False):
            st.warning("**Есть судебные решения о расселении!**")
    st.divider()
    
    st.subheader("Состояние конструкций по осям (подъездам)")
    
    axes = [ax for ax in df['Разделение'].unique() if ax and ax != 'не указана' and len(ax) > 1]
    no_axis_df = df[df['Разделение'] == 'не указана']
    
    if not axes:
        axes = ['Все здание']
        no_axis_df = pd.DataFrame()
    
    for axis in axes:
        if axis == 'Все здание':
            axis_df = df
        else:
            axis_df = df[df['Разделение'] == axis]
        
        if len(axis_df) == 0:
            continue
        
        total = len(axis_df)
        danger = len(axis_df[axis_df['Статус'] == 'АВАРИЙНОЕ'])
        limited = len(axis_df[axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
        working = len(axis_df[axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
        normative = len(axis_df[axis_df['Статус'] == 'НОРМАТИВНОЕ'])
        
        has_collapse = any('обруш' in str(cit).lower() or 'угроз' in str(cit).lower() for cit in axis_df['Цитата'])
        
        if has_collapse:
            axis_icon = "🚨 "
        elif danger > 0:
            axis_icon = "🔴 "
        elif limited > 0:
            axis_icon = "🟡 "
        elif working > 0:
            axis_icon = "🔵 "
        else:
            axis_icon = "🟢 "
        
        with st.expander(f"{axis_icon}Ось {axis} — {total} конструкций", expanded=has_collapse or danger > 0):
            cols = st.columns(5)
            cols[0].metric("Всего", total)
            cols[1].metric("🟢 Нормативное", normative)
            cols[2].metric("🔵 Работоспособное", working)
            cols[3].metric("🟡 Ограниченно-работоспособное", limited)
            cols[4].metric("🔴 Аварийное", danger)
            
            if has_collapse:
                st.error("🚨 В данной оси обнаружена УГРОЗА ОБРУШЕНИЯ!")
                for _, row in axis_df.iterrows():
                    if row['Статус'] == 'АВАРИЙНОЕ':
                        desc = extract_collapse_description(row['Цитата'])
                        if desc:
                            st.warning(f"**Описание угрозы:** {desc}")
                            break
            
            st.markdown("---")
            st.markdown("#### Конструкции по категориям")
            
            categories = ['Фундамент', 'Вертикальные', 'Перекрытия', 'Крыша', 'Лестница']
            
            for category in categories:
                cat_df = axis_df[axis_df['Категория'] == category]
                if len(cat_df) == 0:
                    continue
                
                cat_total = len(cat_df)
                cat_danger = len(cat_df[cat_df['Статус'] == 'АВАРИЙНОЕ'])
                cat_limited = len(cat_df[cat_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
                cat_working = len(cat_df[cat_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
                cat_normative = len(cat_df[cat_df['Статус'] == 'НОРМАТИВНОЕ'])
                
                cat_has_collapse = any('обруш' in str(cit).lower() or 'угроз' in str(cit).lower() for cit in cat_df['Цитата'])
                
                if cat_has_collapse:
                    cat_icon = "🚨"
                elif cat_danger > 0:
                    cat_icon = "🔴"
                elif cat_limited > 0:
                    cat_icon = "🟡"
                elif cat_working > 0:
                    cat_icon = "🔵"
                else:
                    cat_icon = "🟢"
                
                with st.expander(f"{cat_icon} {category} — {cat_total} конструкций"):
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Всего", cat_total)
                    c2.metric("🟢 Нормативное", cat_normative)
                    c3.metric("🔵 Работоспособное", cat_working)
                    c4.metric("🟡 Ограниченно-работоспособное", cat_limited)
                    c5.metric("🔴 Аварийное", cat_danger)
                    
                    if cat_has_collapse:
                        st.warning("В данной категории есть угроза обрушения!")
                    
                    st.markdown("---")
                    
                    
                    danger_rows = cat_df[cat_df['Статус'] == 'АВАРИЙНОЕ']
                    if len(danger_rows) > 0:
                        st.markdown("##### 🔴 Аварийные конструкции")
                        for _, row in danger_rows.iterrows():
                            with st.container():
                                st.markdown(f"**{row['Конструкция']}**")
                                st.caption(f"Ось: {row['Разделение']}")
                                st.caption(f"{row['Цитата'][:300]}..." if len(row['Цитата']) > 300 else f"{row['Цитата']}")
                                llm_desc = row.get('llm_description', '')
                                if llm_desc and len(llm_desc) > 10:
                                    st.error(f"🔴 {llm_desc}")
                                else:
                                    st.error(get_construct_comment(row))
                                if row['Статус'] == 'АВАРИЙНОЕ':
                                    st.markdown("**Требуется:** Немедленное усиление или замена")
                                st.divider()
                    
                    
                    limited_rows = cat_df[cat_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ']
                    if len(limited_rows) > 0:
                        with st.expander(f"🟡 Ограниченные ({len(limited_rows)})"):
                            for _, row in limited_rows.iterrows():
                                st.markdown(f"**{row['Конструкция']}**")
                                st.caption(f"Ось: {row['Разделение']}")
                                st.caption(f"{row['Цитата'][:200]}..." if len(row['Цитата']) > 200 else f"{row['Цитата']}")
                                st.info(get_construct_comment(row))
                                st.divider()
                    
                    
                    working_rows = cat_df[cat_df['Статус'] == 'РАБОТОСПОСОБНОЕ']
                    if len(working_rows) > 0:
                        with st.expander(f"🔵 Работоспособные ({len(working_rows)})"):
                            for _, row in working_rows.iterrows():
                                st.markdown(f"**{row['Конструкция']}**")
                                st.caption(f"Ось: {row['Разделение']}")
                                st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else f"{row['Цитата']}")
                                st.divider()
                    
                
                    norm_rows = cat_df[cat_df['Статус'] == 'НОРМАТИВНОЕ']
                    if len(norm_rows) > 0:
                        with st.expander(f"🟢 Нормативные ({len(norm_rows)})"):
                            for _, row in norm_rows.iterrows():
                                st.markdown(f"**{row['Конструкция']}**")
                                st.caption(f"Ось: {row['Разделение']}")
                                st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else f"{row['Цитата']}")
                                st.divider()
    
    st.divider()
    
    if len(no_axis_df) > 0:
        st.subheader("Конструкции без привязки к осям")
        st.caption("Эти конструкции указаны в отчете, но их расположение по осям не определено")
        
        total_na = len(no_axis_df)
        danger_na = len(no_axis_df[no_axis_df['Статус'] == 'АВАРИЙНОЕ'])
        limited_na = len(no_axis_df[no_axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ'])
        working_na = len(no_axis_df[no_axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ'])
        normative_na = len(no_axis_df[no_axis_df['Статус'] == 'НОРМАТИВНОЕ'])
        
        cols = st.columns(5)
        cols[0].metric("Всего", total_na)
        cols[1].metric("🟢 Нормативное", normative_na)
        cols[2].metric("🔵 Работоспособное", working_na)
        cols[3].metric("🟡 Ограниченное-работоспособное ", limited_na)
        cols[4].metric("🔴 Аварийное", danger_na)
        
        danger_na_rows = no_axis_df[no_axis_df['Статус'] == 'АВАРИЙНОЕ']
        if len(danger_na_rows) > 0:
            st.warning(f"Найдено {len(danger_na_rows)} аварийных конструкций без указания оси!")
            for _, row in danger_na_rows.iterrows():
                with st.expander(f"🔴 {row['Конструкция']} — {row['Категория']}", expanded=True):
                    st.markdown(f"**Статус:** {row['Статус']}")
                    st.markdown(f"**Категория:** {row['Категория']}")
                    st.markdown(f"**Цитата:**")
                    st.code(row['Цитата'], language="text")
                    st.error(get_construct_comment(row))
                    if 'обруш' in str(row['Цитата']).lower() or 'угроз' in str(row['Цитата']).lower():
                        st.error("ОБНАРУЖЕНА УГРОЗА ОБРУШЕНИЯ!")
                        desc = extract_collapse_description(row['Цитата'])
                        if desc:
                            st.markdown(f"**Описание угрозы:** {desc}")
        
        limited_na_rows = no_axis_df[no_axis_df['Статус'] == 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ']
        if len(limited_na_rows) > 0:
            with st.expander(f"🟡 Ограниченные без оси ({len(limited_na_rows)})"):
                for _, row in limited_na_rows.iterrows():
                    st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                    st.caption(f"{row['Цитата'][:200]}..." if len(row['Цитата']) > 200 else row['Цитата'])
                    st.info(get_construct_comment(row))
                    st.divider()
        
        working_na_rows = no_axis_df[no_axis_df['Статус'] == 'РАБОТОСПОСОБНОЕ']
        if len(working_na_rows) > 0:
            with st.expander(f"🔵 Работоспособные без оси ({len(working_na_rows)})"):
                for _, row in working_na_rows.iterrows():
                    st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                    st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                    st.divider()
        
        norm_na_rows = no_axis_df[no_axis_df['Статус'] == 'НОРМАТИВНОЕ']
        if len(norm_na_rows) > 0:
            with st.expander(f"🟢 Нормативные без оси ({len(norm_na_rows)})"):
                for _, row in norm_na_rows.iterrows():
                    st.markdown(f"**{row['Конструкция']}** — {row['Категория']}")
                    st.caption(f"{row['Цитата'][:150]}..." if len(row['Цитата']) > 150 else row['Цитата'])
                    st.divider()
        
        st.divider()
    
    st.subheader("Рекомендации по действиям")
    
    if priority['priority'] == 'КРИТИЧЕСКОЕ':
        st.error("""
        ### ⚠️ НЕМЕДЛЕННЫЕ ДЕЙСТВИЯ:
        1. **Эвакуировать жильцов** из здания
        2. **Ограничить доступ** в здание
        3. **Провести внеочередное обследование**
        4. **Разработать проект** сноса/усиления
        """)
    elif priority['priority'] == 'ВЫСОКИЙ':
        st.warning("""
        ### 🟠 ПРИОРИТЕТНЫЕ ДЕЙСТВИЯ:
        1. Включить в программу расселения **2025-2026**
        2. Провести детальное обследование аварийных конструкций
        3. Разработать проект усиления
        """)
    elif priority['priority'] == 'СРЕДНИЙ':
        st.info("""
        ### 🟡 ПЛАНОВЫЕ ДЕЙСТВИЯ:
        1. Включить в программу расселения **2027-2028**
        2. Провести плановый ремонт
        3. Установить мониторинг состояния
        """)
    else:
        st.success("""
        ### 🟢 ТЕКУЩЕЕ ОБСЛУЖИВАНИЕ:
        1. Плановое техническое обслуживание
        2. Периодический мониторинг состояния
        """)