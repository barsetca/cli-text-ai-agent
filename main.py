#!/usr/bin/env python3
"""
CLI приложение для работы с нейросетями через ProxyAPI
Поддерживает два режима: с рассуждениями и без
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai
import anthropic

# Загружаем переменные окружения
load_dotenv()

# Константы
OPENAI_BASE_URL = "https://openai.api.proxyapi.ru/v1"
ANTHROPIC_BASE_URL = "https://api.proxyapi.ru/anthropic"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"  # Модель Claude Sonnet 4.5 (поддерживается в ProxyAPI)

# Системный промпт для поддержания диалога на русском языке
SYSTEM_PROMPT = "Ты полезный ассистент. Веди диалог на русском языке. Отвечай подробно и по делу."


class ChatSession:
    """Класс для управления сессией чата с сохранением контекста"""
    
    def __init__(self, use_web_search: bool = False):
        self.messages: List[Dict[str, Any]] = []
        self.use_web_search = use_web_search
        # Добавляем системный промпт
        self.messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })
    
    def add_user_message(self, content: str):
        """Добавляет сообщение пользователя в историю"""
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, content: str):
        """Добавляет сообщение ассистента в историю"""
        self.messages.append({
            "role": "assistant",
            "content": content
        })
    
    def get_messages_for_openai(self) -> List[Dict[str, str]]:
        """Возвращает сообщения в формате для OpenAI API"""
        return [msg for msg in self.messages]
    
    def get_messages_for_anthropic(self) -> List[Dict[str, Any]]:
        """Возвращает сообщения в формате для Anthropic API (без system)"""
        return [msg for msg in self.messages if msg["role"] != "system"]
    
    def get_system_prompt(self) -> str:
        """Возвращает системный промпт"""
        if self.messages and self.messages[0]["role"] == "system":
            return self.messages[0]["content"]
        return SYSTEM_PROMPT


def get_api_key() -> str:
    """Получает API ключ из переменных окружения"""
    api_key = os.getenv("PROXYAPI_KEY")
    if not api_key:
        print("Ошибка: API ключ не найден в переменных окружения.")
        print("Пожалуйста, создайте файл .env и добавьте туда PROXYAPI_KEY=ваш_ключ")
        sys.exit(1)
    return api_key


def chat_without_reasoning(api_key: str, use_web_search: bool):
    """
    Режим без вывода рассуждений (OpenAI GPT-4 mini)
    """
    print("\n=== Режим без вывода рассуждений (GPT-4 mini) ===")
    print("Введите 'exit' для выхода из чата\n")
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL
    )
    
    session = ChatSession(use_web_search=use_web_search)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if user_input.lower() == "exit":
                print("\nВыход из чата...\n")
                break
            
            if not user_input:
                continue
            
            # Добавляем сообщение пользователя
            session.add_user_message(user_input)
            
            # Подготавливаем параметры запроса
            # Если веб-поиск включен, используем модель с поддержкой веб-поиска
            model_to_use = OPENAI_MODEL
            if use_web_search:
                # Используем модель с поддержкой веб-поиска
                # Для gpt-4o-mini используем gpt-4o-mini-search-preview
                model_to_use = "gpt-4o-mini-search-preview"
            
            request_params = {
                "model": model_to_use,
                "messages": session.get_messages_for_openai(),
            }
            
            # Модели с веб-поиском не поддерживают параметр temperature
            # Добавляем temperature только для обычных моделей
            if not use_web_search:
                request_params["temperature"] = 0.7
            
            # Если веб-поиск включен, добавляем параметр web_search_options
            # Согласно документации ProxyAPI, для Chat Completions API используется web_search_options
            # с моделями gpt-4o-search-preview или gpt-4o-mini-search-preview
            if use_web_search:
                request_params["web_search_options"] = {
                    "search_context_size": "medium",  # low, medium (по умолчанию), high
                    "user_location": {
                        "type": "approximate",
                        "approximate": {
                            "country": "RU",
                            "city": "Moscow",
                            "region": "Moscow"
                        }
                    }
                }
                print("\n[Веб-поиск включен. Используется модель с поддержкой поиска]\n")
            
            # Отправляем запрос
            response = client.chat.completions.create(**request_params)
            
            # Обрабатываем ответ
            message = response.choices[0].message
            
            # Если использовался веб-поиск, выводим информацию
            # Для OpenAI формат tool_calls может быть разным
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = None
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        tool_name = tool_call.function.name
                    elif hasattr(tool_call, 'name'):
                        tool_name = tool_call.name
                    elif isinstance(tool_call, dict):
                        tool_name = tool_call.get('function', {}).get('name') or tool_call.get('name')
                    
                    if tool_name == "web_search":
                        print(f"\n[Модель выполняет веб-поиск...]\n")
            
            # Получаем ответ
            assistant_message = message.content
            
            # Если ответ None, возможно модель использует tool_calls
            if assistant_message is None and hasattr(message, 'tool_calls') and message.tool_calls:
                # В этом случае нужно сделать дополнительный запрос с результатами tool_calls
                # Но для упрощения просто выводим информацию
                print(f"\n[Модель использует инструменты для получения информации...]\n")
                assistant_message = "[Ответ будет получен после выполнения инструментов]"
            
            # Выводим ответ
            if assistant_message:
                print(f"\nАссистент: {assistant_message}\n")
            
            # Сохраняем ответ в историю
            if assistant_message:
                session.add_assistant_message(assistant_message)
            
        except KeyboardInterrupt:
            print("\n\nВыход из чата...\n")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            continue


def chat_with_reasoning(api_key: str, use_web_search: bool):
    """
    Режим с выводом рассуждений (Anthropic Claude Sonnet 4.5)
    """
    print("\n=== Режим с выводом рассуждений (Claude Sonnet 4.5) ===")
    print("Введите 'exit' для выхода из чата\n")
    
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=ANTHROPIC_BASE_URL
    )
    
    session = ChatSession(use_web_search=use_web_search)
    
    while True:
        try:
            user_input = input("Вы: ").strip()
            
            if user_input.lower() == "exit":
                print("\nВыход из чата...\n")
                break
            
            if not user_input:
                continue
            
            # Добавляем сообщение пользователя
            session.add_user_message(user_input)
            
            # Подготавливаем параметры запроса
            # Для Anthropic нужно отдельно передавать системный промпт и сообщения
            # Для режима с рассуждениями включаем параметр thinking
            # Согласно документации Anthropic: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
            request_params = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 4096,
                "system": session.get_system_prompt(),
                "messages": session.get_messages_for_anthropic(),
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 1024  # Количество токенов для рассуждений (минимум 1024)
                }
            }
            
            # Если веб-поиск включен, добавляем инструмент
            if use_web_search:
                request_params["tools"] = [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5
                }]
            
            # Отправляем запрос
            try:
                response = client.messages.create(**request_params)
            except anthropic.BadRequestError as e:
                # Если модель не поддерживается, пробуем альтернативные варианты
                if "Model not supported" in str(e) or "model" in str(e).lower():
                    print(f"\n[Предупреждение: Модель {ANTHROPIC_MODEL} не поддерживается, пробуем альтернативные варианты...]\n")
                    # Пробуем другие варианты названий моделей
                    alternative_models = [
                        "claude-sonnet-4-5-20250929",
                        "claude-sonnet-4-5-20250514",
                        "claude-sonnet-4-5",
                        "claude-3-7-sonnet-20250219",
                        "claude-3-5-sonnet-20241022"
                    ]
                    response = None
                    for alt_model in alternative_models:
                        try:
                            request_params["model"] = alt_model
                            response = client.messages.create(**request_params)
                            print(f"[Используется модель: {alt_model}]\n")
                            break
                        except:
                            continue
                    if response is None:
                        raise Exception("Не удалось найти поддерживаемую модель Claude. Проверьте доступные модели в документации ProxyAPI.")
                else:
                    raise
            
            # Обрабатываем ответ с рассуждениями
            # В Claude Sonnet 4.5 рассуждения могут быть в отдельных блоках
            reasoning_blocks = []
            text_blocks = []
            tool_use_blocks = []
            
            # Проходим по всем блокам контента
            for content_block in response.content:
                block_type = getattr(content_block, 'type', None)
                
                # Проверяем тип блока
                if block_type == "text":
                    # Это текстовый ответ
                    text_content = getattr(content_block, 'text', '')
                    text_blocks.append(text_content)
                elif block_type == "thinking":
                    # Это блок рассуждений (thinking mode)
                    reasoning_text = ""
                    if hasattr(content_block, 'text'):
                        reasoning_text = content_block.text
                    elif hasattr(content_block, 'content'):
                        reasoning_text = str(content_block.content)
                    else:
                        # Пробуем получить текст из других атрибутов
                        for attr in ['thinking', 'reasoning', 'thought', 'content']:
                            if hasattr(content_block, attr):
                                value = getattr(content_block, attr)
                                if isinstance(value, str):
                                    reasoning_text = value
                                elif hasattr(value, 'text'):
                                    reasoning_text = value.text
                                break
                    
                    if reasoning_text:
                        reasoning_blocks.append(reasoning_text)
                elif block_type == "tool_use":
                    # Это использование инструмента (например, веб-поиск)
                    tool_name = getattr(content_block, 'name', '')
                    if tool_name == "web_search":
                        print(f"\n[Модель выполняет веб-поиск...]\n")
                    tool_use_blocks.append(content_block)
                else:
                    # Проверяем другие возможные типы блоков
                    # Рассуждения могут быть в блоке с другими названиями
                    block_type_str = str(block_type).lower()
                    
                    # Проверяем различные варианты названий блоков рассуждений
                    if any(keyword in block_type_str for keyword in ['reasoning', 'thought']):
                        # Это блок рассуждений
                        reasoning_text = ""
                        if hasattr(content_block, 'text'):
                            reasoning_text = content_block.text
                        elif hasattr(content_block, 'content'):
                            reasoning_text = str(content_block.content)
                        else:
                            # Пробуем получить текст из других атрибутов
                            for attr in ['thinking', 'reasoning', 'thought', 'content']:
                                if hasattr(content_block, attr):
                                    value = getattr(content_block, attr)
                                    if isinstance(value, str):
                                        reasoning_text = value
                                    elif hasattr(value, 'text'):
                                        reasoning_text = value.text
                                    break
                        
                        if reasoning_text:
                            reasoning_blocks.append(reasoning_text)
            
            # Выводим рассуждения, если они есть
            if reasoning_blocks:
                print(f"\n[Рассуждения модели]:\n")
                for reasoning in reasoning_blocks:
                    print(reasoning)
                print()
            else:
                # Если рассуждений нет, выводим отладочную информацию
                print(f"\n[Отладка: Рассуждения не найдены. Типы блоков в ответе: {[getattr(b, 'type', 'unknown') for b in response.content]}]\n")
            
            # Выводим финальный ответ
            final_answer = ""
            if text_blocks:
                final_answer = "\n".join(text_blocks)
                print(f"\n[Окончательный ответ]:\n{final_answer}\n")
            else:
                # Если структура ответа отличается, выводим весь текст
                full_text = ""
                for content_block in response.content:
                    if hasattr(content_block, 'text'):
                        full_text += content_block.text + "\n"
                if full_text:
                    final_answer = full_text.strip()
                    print(f"\n[Окончательный ответ]:\n{final_answer}\n")
            
            # Сохраняем ответ в историю
            # Для Anthropic нужно сохранить весь ответ
            if final_answer:
                session.add_assistant_message(final_answer)
            
        except KeyboardInterrupt:
            print("\n\nВыход из чата...\n")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            import traceback
            traceback.print_exc()
            continue


def ask_web_search() -> bool:
    """Спрашивает пользователя, использовать ли веб-поиск"""
    while True:
        choice = input("Использовать интернет поиск? (да/нет): ").strip().lower()
        if choice in ["да", "д", "yes", "y"]:
            return True
        elif choice in ["нет", "н", "no", "n"]:
            return False
        else:
            print("Пожалуйста, введите 'да' или 'нет'")


def show_menu():
    """Отображает главное меню"""
    print("\n" + "="*50)
    print("CLI приложение для работы с нейросетями")
    print("="*50)
    print("1. Режим без вывода рассуждений (GPT-4 mini)")
    print("2. Режим с выводом рассуждений (Claude Sonnet 4.5)")
    print("0. Завершить работу")
    print("="*50)


def main():
    """Главная функция приложения"""
    api_key = get_api_key()
    
    while True:
        try:
            show_menu()
            choice = input("\nВыберите режим: ").strip()
            
            if choice == "0":
                print("\nДо свидания!")
                break
            elif choice == "1":
                use_web_search = ask_web_search()
                chat_without_reasoning(api_key, use_web_search)
            elif choice == "2":
                use_web_search = ask_web_search()
                chat_with_reasoning(api_key, use_web_search)
            else:
                print("\nНеверный выбор. Пожалуйста, выберите 0, 1 или 2.\n")
                
        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
