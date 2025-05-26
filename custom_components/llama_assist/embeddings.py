from typing import List

from homeassistant.helpers.llm import Tool


def get_matching_tools(user_input: str, tools: List[Tool]) -> List[Tool]:
    return tools


def get_matching_entities(user_input: str, entities: dict[str, dict]) -> dict[str, dict[str, dict]]:
    return entities
