#!/usr/bin/env python3
"""
Тестовый файл для проверки рассуждений Claude Sonnet 4.5
Отправляет запрос и выводит полный JSON ответ от API
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Загружаем переменные окружения
load_dotenv()

# Константы
ANTHROPIC_BASE_URL = "https://api.proxyapi.ru/anthropic"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
SYSTEM_PROMPT = "Ты полезный ассистент. Веди диалог на русском языке. Отвечай подробно и по делу."

def get_api_key() -> str:
    """Получает API ключ из переменных окружения"""
    api_key = os.getenv("PROXYAPI_KEY")
    if not api_key:
        print("Ошибка: API ключ не найден в переменных окружения.")
        print("Пожалуйста, создайте файл .env и добавьте туда PROXYAPI_KEY=ваш_ключ")
        exit(1)
    return api_key

def test_claude_reasoning():
    """Тестирует запрос к Claude с выводом полного JSON ответа"""
    
    api_key = get_api_key()
    
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=ANTHROPIC_BASE_URL
    )
    
    print("="*70)
    print("Тест запроса к Claude Sonnet 4.5 с рассуждениями")
    print("="*70)
    print(f"Модель: {ANTHROPIC_MODEL}")
    print(f"Запрос: Объясни мне теорему Пифагора")
    print("="*70)
    print()
    
    try:
        # Отправляем запрос с включенным режимом рассуждений
        # Для Claude Sonnet 4.5 нужно добавить параметр thinking для включения режима рассуждений
        # Согласно документации Anthropic, используется параметр thinking с типом "enabled" и budget_tokens
        # budget_tokens - количество токенов, которые модель может потратить на рассуждения
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            thinking={
                "type": "enabled",
                "budget_tokens": 1024  # Минимальный бюджет для рассуждений
            },
            messages=[
                {
                    "role": "user",
                    "content": "Объясни мне теорему Пифагора"
                }
            ]
        )
        
        # Выводим полный JSON ответ в красивом формате
        print("="*70)
        print("ПОЛНЫЙ JSON ОТВЕТ ОТ API:")
        print("="*70)
        
        # Функция для безопасной сериализации значений
        def safe_serialize(value):
            """Преобразует значение в JSON-сериализуемый формат"""
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                return [safe_serialize(item) for item in value]
            if isinstance(value, dict):
                return {k: safe_serialize(v) for k, v in value.items()}
            # Для объектов Pydantic и других объектов
            if hasattr(value, 'model_dump'):
                try:
                    return value.model_dump()
                except:
                    pass
            if hasattr(value, 'dict'):
                try:
                    return value.dict()
                except:
                    pass
            # Пробуем преобразовать в строку
            try:
                return str(value)
            except:
                return None
        
        # Преобразуем ответ в словарь для красивого вывода
        response_dict = {
            "id": str(response.id) if hasattr(response, 'id') else None,
            "type": str(response.type) if hasattr(response, 'type') else None,
            "role": str(response.role) if hasattr(response, 'role') else None,
            "content": [],
            "model": str(response.model) if hasattr(response, 'model') else None,
            "stop_reason": str(response.stop_reason) if hasattr(response, 'stop_reason') else None,
            "stop_sequence": str(response.stop_sequence) if hasattr(response, 'stop_sequence') else None,
        }
        
        # Добавляем usage, если есть
        if hasattr(response, 'usage') and response.usage:
            response_dict["usage"] = {
                "input_tokens": response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else None,
                "output_tokens": response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else None
            }
        
        # Обрабатываем content блоки
        for i, content_block in enumerate(response.content):
            block_dict = {
                "index": i,
                "type": str(content_block.type) if hasattr(content_block, 'type') else "unknown",
            }
            
            # Добавляем специфичные поля в зависимости от типа блока
            if hasattr(content_block, 'text'):
                block_dict["text"] = content_block.text
            
            if hasattr(content_block, 'name'):
                block_dict["name"] = content_block.name
            
            if hasattr(content_block, 'input'):
                block_dict["input"] = safe_serialize(content_block.input)
            
            # Проверяем только основные атрибуты, которые точно есть
            important_attrs = ['id', 'text', 'name', 'input', 'type']
            for attr in important_attrs:
                if hasattr(content_block, attr) and attr not in block_dict:
                    try:
                        value = getattr(content_block, attr)
                        if value is not None and not callable(value):
                            # Пропускаем Pydantic-специфичные атрибуты
                            if not isinstance(value, type) and not attr.startswith('model_'):
                                block_dict[attr] = safe_serialize(value)
                    except:
                        pass
            
            response_dict["content"].append(block_dict)
        
        # Выводим в красивом JSON формате
        print(json.dumps(response_dict, indent=2, ensure_ascii=False))
        print()
        print("="*70)
        print("АНАЛИЗ СТРУКТУРЫ ОТВЕТА:")
        print("="*70)
        
        # Анализируем структуру ответа
        print(f"Количество блоков в content: {len(response.content)}")
        print()
        
        for i, content_block in enumerate(response.content):
            print(f"Блок #{i+1}:")
            print(f"  Тип: {content_block.type}")
            
            if hasattr(content_block, 'text'):
                text = content_block.text
                print(f"  Текст (первые 200 символов): {text[:200]}...")
            
            # Выводим все атрибуты блока
            print(f"  Все атрибуты блока:")
            for attr in dir(content_block):
                if not attr.startswith('_') and not callable(getattr(content_block, attr, None)):
                    try:
                        value = getattr(content_block, attr)
                        if not callable(value):
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            print(f"    {attr}: {value_str}")
                    except:
                        pass
            print()
        
        print("="*70)
        print("ОБРАБОТКА РАССУЖДЕНИЙ:")
        print("="*70)
        
        # Ищем рассуждения
        reasoning_found = False
        for content_block in response.content:
            block_type = str(content_block.type).lower()
            if 'reasoning' in block_type:
                reasoning_found = True
                print("✓ Найден блок рассуждений!")
                if hasattr(content_block, 'text'):
                    print(f"Рассуждения: {content_block.text[:500]}...")
                break
        
        if not reasoning_found:
            print("✗ Блок рассуждений не найден в ответе")
            print("Проверьте структуру ответа выше")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_claude_reasoning()
