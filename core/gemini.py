import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types


# chat.py / tools.py は Anthropic の Message と同じ形（.content / .stop_reason、
# ブロックの .type / .id / .name / .input）を前提に書かれているので、
# それに合わせた軽量オブジェクトを返す。
@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    raw: Any = None  # Gemini の元の Part（thought_signature を保持するため）
    type: str = "tool_use"


@dataclass
class Message:
    content: list
    stop_reason: str


class Gemini:
    def __init__(self, model: str):
        # GEMINI_API_KEY を環境変数から自動で読む
        self.client = genai.Client()
        self.model = model
        # tool_result には名前がなく id だけなので、id -> 名前 の対応を覚えておく
        self._tool_names: dict[str, str] = {}

    def add_user_message(self, messages: list, message):
        messages.append(
            {
                "role": "user",
                "content": message.content
                if isinstance(message, Message)
                else message,
            }
        )

    def add_assistant_message(self, messages: list, message):
        messages.append(
            {
                "role": "assistant",
                "content": message.content
                if isinstance(message, Message)
                else message,
            }
        )

    def text_from_message(self, message: Message):
        return "\n".join(
            [b.text for b in message.content if b.type == "text"]
        )

    # ---- Anthropic形式のメッセージ -> Gemini の contents ----
    def _to_part(self, block) -> types.Part | None:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                return types.Part(text=text) if text else None
            if btype == "tool_result":
                name = self._tool_names.get(block["tool_use_id"], "unknown")
                key = "error" if block.get("is_error") else "result"
                return types.Part.from_function_response(
                    name=name, response={key: block.get("content")}
                )
            return None
        if isinstance(block, ToolUseBlock):
            if block.raw is not None:
                return block.raw
            return types.Part(
                function_call=types.FunctionCall(
                    name=block.name, args=block.input
                )
            )
        if isinstance(block, TextBlock):
            return types.Part(text=block.text) if block.text else None
        return None

    def _to_contents(self, messages: list) -> list[types.Content]:
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            if isinstance(content, str):
                parts = [types.Part(text=content)] if content else []
            else:
                parts = [
                    p for p in (self._to_part(b) for b in content) if p
                ]
            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    # ---- ToolManager が作る Anthropic形式のツール定義 -> Gemini ----
    def _to_tools(self, tools: list[dict]) -> list[types.Tool]:
        declarations = [
            types.FunctionDeclaration(
                name=t["name"],
                description=t.get("description") or "",
                parameters_json_schema=t["input_schema"],
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> Message:
        config = types.GenerateContentConfig(
            max_output_tokens=8000,
            temperature=temperature,
            system_instruction=system,
            stop_sequences=stop_sequences or None,
            tools=self._to_tools(tools) if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        if thinking:
            config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=self._to_contents(messages),
            config=config,
        )

        candidate = response.candidates[0] if response.candidates else None
        parts = (
            candidate.content.parts
            if candidate and candidate.content and candidate.content.parts
            else []
        )

        blocks: list = []
        for part in parts:
            if part.function_call:
                fc = part.function_call
                call_id = fc.id or f"call_{uuid.uuid4().hex[:8]}"
                self._tool_names[call_id] = fc.name
                blocks.append(
                    ToolUseBlock(
                        id=call_id,
                        name=fc.name,
                        input=dict(fc.args or {}),
                        raw=part,
                    )
                )
            elif part.text and not part.thought:
                blocks.append(TextBlock(text=part.text))

        stop_reason = (
            "tool_use"
            if any(b.type == "tool_use" for b in blocks)
            else "end_turn"
        )
        return Message(content=blocks, stop_reason=stop_reason)
