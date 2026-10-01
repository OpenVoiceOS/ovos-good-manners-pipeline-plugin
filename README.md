# <img src='goodmanner.png' width='50' height='50' style='vertical-align:bottom'/> Good Manners

An OVOS pipeline plugin that reinforces good manners and reprimands foul language, without getting in the way of what you actually asked for.

```
"hey mycroft, what the fuck is the weather"
→ the weather is answered as usual
→ "avoid using the word fuck, it makes you look bad"

"please tell me a joke" / "what time is it, please" / ...
→ answered as usual, and after a few polite requests in a row:
→ "you have really good manners, i like you"
```

> Originally a Mycroft skill by [@JarbasAl](https://github.com/JarbasAl) (2018). It is being rewritten as a pipeline plugin, see [#9](https://github.com/OpenVoiceOS/ovos-skill-good-manners/issues/9).

## How it works

The plugin is a stage in ovos-core's intent pipeline that **observes but never matches**:

1. It sits first in the pipeline, so it sees every utterance before any other stage can claim it.
2. It classifies the utterance (polite, foul language) and remembers the result for that session, then declines, so the utterance goes on to the stage that really handles it.
3. When that utterance has been handled (`ovos.utterance.handled`), it speaks its comeback, in the language of the utterance and to the session that said it.

Intent probes (`intent.service.intent.get`) are ignored, and every session keeps its own polite streak, so on a HiveMind hub one client's manners don't affect another's.

Classification is plain vocabulary matching (`locale/<lang>/foul_language.voc` and `polite_words.voc`). A trained politeness/insult classifier is planned.

## Install

```bash
pip install ovos-good-manners-pipeline-plugin
```

## Configuration

Put the plugin **first** in the pipeline in `mycroft.conf`, followed by your normal pipeline:

```json
{
  "intents": {
    "pipeline": [
      "ovos-good-manners-pipeline-plugin",
      "stop_high",
      "converse",
      "..."
    ],
    "ovos-good-manners-pipeline-plugin": {
      "polite_threshold": 4,
      "polite_timeout": 10,
      "max_delay": 20
    }
  }
}
```

| Setting | Default | Meaning |
|---|---|---|
| `polite_threshold` | `4` | polite requests in a row before you get a compliment |
| `polite_timeout` | `10` | minutes without a polite request before the streak starts over |
| `max_delay` | `20` | seconds after the utterance after which a comeback is dropped instead of said late |

Since the pipeline is chosen per session, a HiveMind client can enable or disable the plugin through its own `session.pipeline`.

## Languages

English, German and Danish. A language needs `foul_language.voc`, `polite_words.voc`, `and.voc` and the four `.dialog` files in `locale/<lang>/`.

## Credits

- [@JarbasAl](https://github.com/JarbasAl), original skill
- [@gras64](https://github.com/gras64), German translation
- [@andlo](https://github.com/andlo)
