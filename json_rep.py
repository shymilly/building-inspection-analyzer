import json
import os
from pathlib import Path
from database import Database
from datetime import datetime
import re

def find_reports_by_patterns():
    """Находит отчеты по названиям файлов"""
    
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print(f" Папка {reports_dir} не найдена")
        return []
    
    # Список файлов для поиска
    patterns = [
        "МАГДАГАЧИ, ул. М. Горького, д. 7",
        "ВЕРХНЕЗЕЙСК, д. 65",
        "13. 14_92.ТО.2020_ул. Южная, д.14",
        "Михнево, 9 Мая, 1",
        "1796_5_ЗАКЛЮЧЕНИЕ 25 дом испр",
        "1796_2_ЗАКЛЮЧЕНИЕ 6 дом испр",
        "1796_3_ЗАКЛЮЧЕНИЕ 7 дом испр",
        "otchetvenchyr",
        "oychetlihack"
    ]
    
    found_files = []
    for file_path in reports_dir.glob("*.pdf"):
        file_name = file_path.stem  # без .pdf
        for pattern in patterns:
            if pattern in file_name or file_name in pattern:
                found_files.append(file_path)
                print(f"Найден: {file_path.name}")
                break
    
    return found_files

def export_selected_reports():
    """Экспорт выбранных отчетов"""
    
    db = Database()
    
    # Находим файлы
    pdf_files = find_reports_by_patterns()
    
    if not pdf_files:
        print("Ни один файл не найден")
        return
    
    print(f"\nНайдено {len(pdf_files)} файлов")
    
    export_data = {
        "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_reports": len(pdf_files),
        "reports": []
    }
    
    for pdf_path in pdf_files:
        report_hash = pdf_path.stem 
        
        # Ищем в БД по имени файла
        all_reports = db.get_all_reports()
        found_report = None
        
        for r in all_reports:
            if r.get('filename') == pdf_path.name:
                found_report = r
                break
        
        if not found_report:
            print(f"Отчет {pdf_path.name} не найден в БД, пропускаем")
            continue
        
        report_id = found_report.get('id') or found_report.get('hash')
        full_report = db.load_report(report_id) if report_id else None
        
        constructs = []
        if full_report:
            for c in full_report.get('constructs', []):
                constructs.append({
                    "name": c.get('Конструкция', ''),
                    "category": c.get('Категория', ''),
                    "status": c.get('Статус', ''),
                    "location": c.get('Разделение', 'не указана'),
                    "description": c.get('llm_description', ''),
                    "citation": c.get('Цитата', '')[:150],
                    "has_collapse_risk": c.get('Угроза_обрушения', False)
                })
        
        report_data = {
            "filename": pdf_path.name,
            "building_id": found_report.get('building_id', 'Неизвестно'),
            "date": found_report.get('date', ''),
            "priority": found_report.get('priority', 'Не определен'),
            "health_index": found_report.get('health_index', 0),
            "reason": found_report.get('reason', ''),
            "has_collapse": bool(found_report.get('has_collapse', False)),
            "total_constructs": found_report.get('total_constructs', 0),
            "status_counts": found_report.get('status_counts', {}),
            "constructs": constructs
        }
        
        export_data["reports"].append(report_data)
        print(f"Добавлен: {pdf_path.name}")
    
    output_path = Path("selected_reports_export.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nЭкспортировано {len(export_data['reports'])} отчетов в {output_path}")
    print(f"Файл: {output_path.absolute()}")

if __name__ == "__main__":
    
    export_selected_reports()