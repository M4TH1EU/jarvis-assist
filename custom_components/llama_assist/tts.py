"""Support for Wyoming text-to-speech services."""

import io
import logging
import wave
from collections import defaultdict

from homeassistant.components import tts
from homeassistant.components.tts import TextToSpeechEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Wyoming speech-to-text."""
    # item: DomainDataItem = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities([LlamaAssistTtsProvider(config_entry), ])


class LlamaAssistTtsProvider(TextToSpeechEntity):
    """Wyoming text-to-speech provider."""

    def __init__(
            self,
            config_entry: ConfigEntry,
    ) -> None:
        """Set up provider."""
        # self.service = service
        # self._tts_service = next(tts for tts in service.info.tts if tts.installed)

        voice_languages: set[str] = set()
        self._voices: dict[str, list[tts.Voice]] = defaultdict(list)

        voice_languages.add("en")
        self._voices["en"].append(tts.Voice(
            voice_id="default",
            name="Default Voice",
        ))

        # for voice in self._tts_service.voices:
        #     if not voice.installed:
        #         continue
        #
        #     voice_languages.update(voice.languages)
        #     for language in voice.languages:
        #         self._voices[language].append(
        #             tts.Voice(
        #                 voice_id=voice.name,
        #                 name=voice.description or voice.name,
        #             )
        #         )

        # Sort voices by name
        # for language in self._voices:
        #     self._voices[language] = sorted(
        #         self._voices[language], key=lambda v: v.name
        #     )

        self._supported_languages: list[str] = list(voice_languages)

        # self._attr_name = self._tts_service.name
        self._attr_name = "Llama Assist TTS"
        self._attr_unique_id = f"{config_entry.entry_id}-tts"

    @property
    def default_language(self):
        """Return default language."""
        if not self._supported_languages:
            return None

        return self._supported_languages[0]

    @property
    def supported_languages(self):
        """Return list of supported languages."""
        return self._supported_languages

    @property
    def supported_options(self):
        """Return list of supported options like voice, emotion."""
        return [
            tts.ATTR_AUDIO_OUTPUT,
            tts.ATTR_VOICE,
            # ATTR_SPEAKER,
        ]

    @property
    def default_options(self):
        """Return a dict include default options."""
        return {}

    @callback
    def async_get_supported_voices(self, language: str) -> list[tts.Voice] | None:
        """Return a list of supported voices for a language."""
        return self._voices.get(language)

    async def async_get_tts_audio(self, message, language, options):
        """Stream TTS audio from OpenAI backend and return as WAV."""
        voice_name: str | None = options.get("voice") if options else None
        selected_voice = voice_name or self.voice

        try:
            _LOGGER.debug("Sending TTS request to OpenAI backend (voice: %s, model: %s)", selected_voice, self.model)

            # OpenAI's streaming response gives us raw PCM data
            with self.client.audio.speech.with_streaming_response.create(
                    model=self.model,
                    voice=selected_voice,
                    input=message,
                    response_format="pcm"  # PCM 16-bit, 24kHz, mono
            ) as response:
                pcm_data = b""
                async for chunk in response.iter_bytes(chunk_size=1024):
                    pcm_data += chunk

            # Convert PCM to WAV in memory (24kHz, 16-bit, mono)
            with io.BytesIO() as wav_io:
                wav_writer = wave.open(wav_io, "wb")
                wav_writer.setnchannels(1)
                wav_writer.setsampwidth(2)  # 16-bit
                wav_writer.setframerate(24000)
                wav_writer.writeframes(pcm_data)
                wav_writer.close()
                data = wav_io.getvalue()

            _LOGGER.debug("TTS audio successfully received and converted to WAV")
            return ("wav", data)

        except Exception as e:
            _LOGGER.exception("Failed to generate TTS audio via OpenAI: %s", e)
            return (None, None)
