import logging

from homeassistant.const import Platform

DOMAIN = "llama_assist"
LLAMA_LLM_API = DOMAIN + "_api"
LOGGER = logging.getLogger(__name__)
PLATFORMS = (Platform.CONVERSATION,)

CONF_COMPLETION_SERVER_URL = "completion-server-url"
CONF_SERVER_EMBEDDINGS_URL = "embeddings-server-url"
CONF_PROMPT = "prompt"

SERVER_API_TIMEOUT = 5

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

CONF_MAX_HISTORY = "max_history"
DEFAULT_MAX_HISTORY = 20

CONF_DISABLE_REASONING = "disable_reasoning"
DISABLE_REASONING = False

CONF_USE_EMBEDDINGS_TOOLS = "use_embeddings_tools"
USE_EMBEDDINGS_TOOLS = False

CONF_USE_EMBEDDINGS_ENTITIES = "use_embeddings_entities"
USE_EMBEDDINGS_ENTITIES = False

CONF_OVERWRITE_EMBEDDINGS = "overwrite_embeddings"
OVERWRITE_EMBEDDINGS = False

EMBEDDINGS_MIN_SCORE = 0.65

CONF_BLACKLIST_TOOLS = "blacklist_tools"
EXISTING_TOOLS = [
    'HassTurnOn',
    'HassTurnOff',
    'HassCancelAllTimers',
    'HassMediaUnpause',
    'HassMediaPause',
    'HassMediaNext',
    'HassMediaPrevious',
    'HassSetVolume',
    'HassMediaSearchAndPlay',
    'HassListAddItem',
    'HassListCompleteItem',
    'todo_get_items',
    'GetLiveContext',
    'HassSetPosition',
    'HassShoppingListAddItem',
    'HassShoppingListLastItems',
]

EMBEDDINGS_SQLITE = "llama_assist_embeddings.db"
