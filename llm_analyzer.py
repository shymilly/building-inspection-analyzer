import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class LocalLLMAnalyzer:
    """
    Улучшенный локальный LLM-анализатор на TinyLlama.
    С явными инструкциями и примерами.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"🔄 Загрузка модели {model_name}...")    
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Устанавливаем pad_token, если его нет
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Модель загружена!")

    def generate_description(self, construct_type, status, defects, citations, style="expert"):
        """
        Генерирует описание состояния конструкции.
        """
        
        # Выбираем промпт в зависимости от стиля
        prompts = self._get_improved_prompts(construct_type, status, defects, citations)
        prompt = prompts.get(style, prompts['expert'])
        
        # Токенизируем с паддингом
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.3,  # <-- УМЕНЬШИЛИ ДЛЯ БОЛЬШЕЙ ТОЧНОСТИ
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Чистим ответ
        response = self._clean_response(response, prompt)
        
        # Если ответ пустой или слишком короткий — fallback
        if len(response) < 15 or response.lower().startswith("конструкция"):
            response = self._get_fallback_description(construct_type, status, defects)
        
        return response
    
    def _get_improved_prompts(self, construct_type, status, defects, citations):
        """Улучшенные промпты с примерами"""
        
        # Статус с эмодзи для контекста
        status_emoji = {
            'НОРМАТИВНОЕ': '✅ в хорошем состоянии',
            'РАБОТОСПОСОБНОЕ': '🔵 работоспособен, но требует внимания',
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': '🟡 требует усиления и ремонта',
            'АВАРИЙНОЕ': '🔴 в АВАРИЙНОМ состоянии! Требуется срочное вмешательство'
        }.get(status, '⚪ состояние не определено')
        
        return {
            'expert': f"""Ты — эксперт по строительным конструкциям. Дай КРАТКИЙ и ЧЕТКИЙ ответ на русском языке.

Конструкция: {construct_type}
Состояние: {status_emoji}
Дефекты: {defects}
Цитата из отчета: {citations}

Напиши 2-3 предложения о состоянии конструкции. Укажи:
1. В чем проблема
2. Что нужно сделать
3. Насколько срочно

Ответ должен быть понятным и конкретным, без воды.

Ответ:""",

            'simple': f"""Объясни простыми словами, что происходит с конструкцией.

Конструкция: {construct_type}
Состояние: {status_emoji}
Проблемы: {defects}

Напиши 1-2 предложения для жильцов дома. Без сложных терминов.

Ответ:""",

            'urgent': f"""СРОЧНО! Опиши угрозу для конструкции.

Конструкция: {construct_type}
Состояние: АВАРИЙНОЕ! 🔴
Дефекты: {defects}

Напиши 2 предложения:
1. В чем опасность
2. Что делать немедленно

Ответ:""",

            'brief': f"""Кратко опиши состояние конструкции в одном предложении.

Конструкция: {construct_type}
Состояние: {status_emoji}

Ответ:""",

            'recommendation': f"""Дай конкретные рекомендации по устранению дефектов.

Конструкция: {construct_type}
Состояние: {status_emoji}
Дефекты: {defects}

Напиши 3 пункта плана действий для управляющей компании.

Ответ:"""
        }

    def _clean_response(self, response, prompt):
        """Очищает ответ от мусора"""
        # Убираем промпт из ответа
        if prompt in response:
            response = response.replace(prompt, "")
        
        # Убираем маркеры
        markers = ["Ответ:", "Ответ", "assistant", "<|assistant|>"]
        for marker in markers:
            if marker in response:
                response = response.split(marker)[-1]
        
        # Убираем лишние пробелы
        response = response.strip()
        
        # Если ответ начинается с "Конструкция" — обрезаем
        if response.startswith("Конструкция:"):
            response = response.split("Конструкция:")[0].strip()
        
        return response
    
    def _get_fallback_description(self, construct_type, status, defects=""):
        """Запасное описание"""
        fallbacks = {
            'НОРМАТИВНОЕ': f'✅ Конструкция «{construct_type}» в нормативном состоянии. Дефектов не выявлено.',
            'РАБОТОСПОСОБНОЕ': f'🔵 Конструкция «{construct_type}» работоспособна. Рекомендуется плановый осмотр.',
            'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ': f'🟡 Конструкция «{construct_type}» требует усиления. Дефекты: {defects[:100]}.',
            'АВАРИЙНОЕ': f'🔴 КОНСТРУКЦИЯ «{construct_type}» В АВАРИЙНОМ СОСТОЯНИИ! Требуется НЕМЕДЛЕННОЕ усиление. Дефекты: {defects[:100]}.'
        }
        return fallbacks.get(status, f'⚪ Состояние «{construct_type}» не определено.')


# ---- ИНИЦИАЛИЗАЦИЯ ----
_llm_instance = None

def get_llm_analyzer():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLMAnalyzer()
    return _llm_instance


# ---- ТЕСТ ----
if __name__ == "__main__":
    print("🧪 Тестирование УЛУЧШЕННОГО LLM-анализатора...")
    
    llm = get_llm_analyzer()
    
    test_data = [
        ('фундамент', 'АВАРИЙНОЕ', 'сквозные трещины, просадка грунта, разрушение бетона', 'фундамент в аварийном состоянии'),
        ('стена', 'ОГРАНИЧЕННО-РАБОТОСПОСОБНОЕ', 'трещины в кладке, коррозия арматуры', 'требуется усиление стен'),
        ('перекрытие', 'НОРМАТИВНОЕ', 'без замечаний', 'состояние удовлетворительное'),
    ]
    
    for construct, status, defects, citations in test_data:
        print(f"\n{'='*60}")
        print(f"🏗️ {construct} — {status}")
        print(f"   Дефекты: {defects}")
        print("-" * 60)
        
        result = llm.generate_description(construct, status, defects, citations, style="expert")
        print(f"📝 {result}")