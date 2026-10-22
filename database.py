import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class Database:
  

    def __init__(self, db_path: str = "data/reports.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                building_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                date TEXT NOT NULL,
                priority TEXT,
                health_index REAL,
                reason TEXT,
                has_collapse INTEGER DEFAULT 0,
                has_court_order INTEGER DEFAULT 0,
                total_constructs INTEGER DEFAULT 0,
                status_counts TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS constructs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                section TEXT,
                citation TEXT,
                description TEXT,
                reinforcement TEXT,
                reinforcement_year TEXT,
                FOREIGN KEY (report_id) REFERENCES reports (id)
            )
        """)

        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_building ON reports(building_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_constructs_report ON constructs(report_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_constructs_status ON constructs(status)")

        conn.commit()
        conn.close()
        print("База данных инициализирована")

    def save_report(self, report_data: Dict[str, Any]) -> str:
       
        report_id = report_data.get('hash', '') or report_data.get('id', '')
        if not report_id:
            report_id = f"rep_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        conn = self._get_connection()
        cursor = conn.cursor()

        
        cursor.execute("""
            INSERT OR REPLACE INTO reports (
                id, building_id, filename, date, priority, health_index,
                reason, has_collapse, has_court_order, total_constructs, status_counts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id,
            report_data.get('building_id', 'unknown'),
            report_data.get('filename', 'unknown'),
            report_data.get('date', datetime.now().isoformat()),
            report_data.get('priority', ''),
            report_data.get('health_index', 0.0),
            report_data.get('reason', ''),
            1 if report_data.get('has_collapse', False) else 0,
            1 if report_data.get('has_court_order', False) else 0,
            report_data.get('total_constructs', 0),
            json.dumps(report_data.get('status_counts', {}), ensure_ascii=False)
        ))

    
        constructs = report_data.get('constructs', [])
        for c in constructs:
            cursor.execute("""
                INSERT OR REPLACE INTO constructs (
                    report_id, name, category, status, section, citation,
                    description, reinforcement, reinforcement_year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                c.get('Конструкция', ''),
                c.get('Категория', ''),
                c.get('Статус', ''),
                c.get('Секция', ''),
                c.get('Цитата', ''),
                c.get('llm_description', ''),
                c.get('Усиление', ''),
                c.get('Год_усиления', '')
            ))

        conn.commit()
        conn.close()
        return report_id

    def load_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        report = cursor.fetchone()

        if not report:
            conn.close()
            return None

        result = dict(report)
        result['status_counts'] = json.loads(result['status_counts'])

        
        cursor.execute("SELECT * FROM constructs WHERE report_id = ?", (report_id,))
        constructs = [dict(row) for row in cursor.fetchall()]

        result['constructs'] = constructs
        conn.close()

        return result

    def get_all_reports(self, building_id: Optional[str] = None) -> List[Dict[str, Any]]:
        
        conn = self._get_connection()
        cursor = conn.cursor()

        if building_id:
            cursor.execute(
                "SELECT * FROM reports WHERE building_id = ? ORDER BY date DESC",
                (building_id,)
            )
        else:
            cursor.execute("SELECT * FROM reports ORDER BY date DESC")

        reports = [dict(row) for row in cursor.fetchall()]
        conn.close()

    
        for r in reports:
            if isinstance(r.get('status_counts'), str):
                r['status_counts'] = json.loads(r['status_counts'])

        return reports

    def get_report_by_hash(self, report_hash: str, building_id: str) -> Optional[Dict[str, Any]]:
    
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM reports WHERE id = ? AND building_id = ?",
            (report_hash, building_id)
        )
        report = cursor.fetchone()
        conn.close()

        if report:
            result = dict(report)
            result['status_counts'] = json.loads(result['status_counts'])
            return result

        return None

    def get_buildings(self) -> List[Dict[str, Any]]:
    
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                building_id,
                COUNT(*) as reports_count,
                MAX(date) as last_report,
                MAX(priority) as last_priority
            FROM reports 
            GROUP BY building_id
            ORDER BY building_id
        """)

        buildings = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return buildings

    def get_constructs_by_status(self, status: str) -> List[Dict[str, Any]]:
    
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.*, r.building_id, r.filename, r.date
            FROM constructs c
            JOIN reports r ON c.report_id = r.id
            WHERE c.status = ?
            ORDER BY r.date DESC
        """, (status,))

        constructs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return constructs

    def delete_report(self, report_id: str) -> bool:
    
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM constructs WHERE report_id = ?", (report_id,))
            cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка удаления: {e}")
            conn.close()
            return False

    def get_statistics(self) -> Dict[str, Any]:
    
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total_reports FROM reports")
        total_reports = cursor.fetchone()['total_reports']

        cursor.execute("SELECT COUNT(DISTINCT building_id) as total_buildings FROM reports")
        total_buildings = cursor.fetchone()['total_buildings']

        cursor.execute("SELECT COUNT(*) as total_constructs FROM constructs")
        total_constructs = cursor.fetchone()['total_constructs']

        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM constructs 
            GROUP BY status
        """)
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

        conn.close()

        return {
            'total_reports': total_reports,
            'total_buildings': total_buildings,
            'total_constructs': total_constructs,
            'status_counts': status_counts
        }


if __name__ == "__main__":

    db = Database()

    test_report = {
        'id': 'test_001',
        'building_id': 'г. Москва, ул. Тестовая, д. 1',
        'filename': 'test_report.pdf',
        'date': '2025-01-15 12:00:00',
        'priority': 'НИЗКИЙ',
        'health_index': 74.4,
        'reason': 'Тестовый отчет',
        'has_collapse': False,
        'has_court_order': False,
        'total_constructs': 5,
        'status_counts': {'НОРМАТИВНОЕ': 3, 'РАБОТОСПОСОБНОЕ': 2},
        'constructs': [
            {
                'Конструкция': 'фундамент',
                'Категория': 'Фундамент',
                'Статус': 'НОРМАТИВНОЕ',
                'Секция': 'основная',
                'Цитата': 'Фундамент в нормативном состоянии',
                'llm_description': 'Фундамент в нормативном состоянии. Дефектов не выявлено.',
                'Усиление': '',
                'Год_усиления': ''
            },
            {
                'Конструкция': 'стена',
                'Категория': 'Вертикальные',
                'Статус': 'АВАРИЙНОЕ',
                'Секция': 'основная',
                'Цитата': 'Сквозные трещины в стене',
                'llm_description': 'стена в АВАРИЙНОМ состоянии! Сквозные трещины в стене',
                'Усиление': 'усиление выполнено в 2020 году',
                'Год_усиления': '2020'
            }
        ]
    }

    report_id = db.save_report(test_report)
    print(f"Отчет сохранен: {report_id}")

    loaded = db.load_report(report_id)
    if loaded:
        print(f"Загружен отчет: {loaded['filename']}")
        print(f"   Приоритет: {loaded['priority']}")
        print(f"   Конструкций: {len(loaded['constructs'])}")

    
    stats = db.get_statistics()
    print(f"\nСтатистика:")
    print(f"   Отчетов: {stats['total_reports']}")
    print(f"   Зданий: {stats['total_buildings']}")
    print(f"   Конструкций: {stats['total_constructs']}")
    print(f"   Статусы: {stats['status_counts']}")