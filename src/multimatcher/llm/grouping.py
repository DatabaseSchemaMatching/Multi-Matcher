from __future__ import annotations

from typing import List, Any

from langchain_core.messages import HumanMessage, SystemMessage

def run_grouping(
    chat_model: Any,
    llm_reasoning_inputs: List[str],
    system_prompt: str,
) -> List[str]:
    """
    공통 루프:
    - SystemMessage + HumanMessage
    - chat_model.invoke()
    - result.content 수집
    """
    outputs: List[str] = []

    for llm_input in llm_reasoning_inputs:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=llm_input),
        ]
        try:
            res = chat_model.invoke(messages)
            outputs.append(getattr(res, "content", str(res)))
        except Exception as e:
            # 실패한 경우에도 길이를 맞추고 싶으면 None 넣기
            outputs.append("None")
            print("Failed to group:", e)

    return outputs