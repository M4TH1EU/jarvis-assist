DOMAIN = "llama_assist"
LLAMA_LLM_API = DOMAIN + "_api"

CONF_COMPLETION_SERVER_URL = "completion-server-url"
CONF_SERVER_EMBEDDINGS_URL = "embeddings-server-url"
CONF_PROMPT = "prompt"

DEFAULT_TIMEOUT = 5

CONF_MAX_HISTORY = "max_history"
DEFAULT_MAX_HISTORY = 20

CONF_DISABLE_REASONING = "disable_reasoning"
DISABLE_REASONING = False

CONF_USE_EMBEDDINGS_TOOLS = "use_embeddings_tools"
CONF_USE_EMBEDDINGS_ENTITIES = "use_embeddings_entities"
USE_EMBEDDINGS_TOOLS = False
USE_EMBEDDINGS_ENTITIES = False

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
    'GetLiveContext'
]

CHROMADB_PATH = "./chroma_llama_assist_db"
