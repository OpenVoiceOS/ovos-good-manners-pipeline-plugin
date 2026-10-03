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

> Originally a Mycroft skill by [@JarbasAl](https://github.com/JarbasAl) (2018). It is being rewritten as a pipeline plugin, see [#9](https://github.com/OpenVoiceOS/ovos-good-manners-pipeline-plugin/issues/9).

## Why manners matter

### Acknowledge and appreciate good manners in others

Manners are reciprocal. If someone holds a door, say thank you. If you need something from a grocery shelf, and someone else is in the way, say "excuse me, please." Or ask them "could you please hand me a box of that cereal?" and then thank them.

### Why do we need manners?

Manners make the world go 'round. They are to the smooth functioning of society as oil is to an engine. Without good manners, people get offended, hurt, and in extreme cases, very bad manners can lead to things such as the all-too-familiar public shootings, and even wars between countries when some official protocol is snubbed.

So this plugin listens to how you talk to your assistant: it shows appreciation for good manners, and reprimands insults and foul language.

## How it works

The plugin is a stage in ovos-core's intent pipeline that **observes but never matches**:

1. It sits first in the pipeline, so it sees every utterance before any other stage can claim it.
2. It classifies the utterance (polite, foul language) and remembers the result for that session, then declines, so the utterance goes on to the stage that really handles it.
3. When that utterance has been handled (`ovos.utterance.handled`), it speaks its comeback, in the language of the utterance and to the session that said it.

Intent probes (`intent.service.intent.get`) are ignored, and every session keeps its own polite streak, so on a HiveMind hub one client's manners don't affect another's.

Classification is plain vocabulary matching, with three lists per language in `ovos_good_manners_pipeline_plugin/locale/<lang>/`:

- `foul_language.voc` – ordinary swearing. The comeback names the word: *"shit is such an ugly word"*.
- `foul_prefixes.voc` – for compounding languages: any word that *starts* with one of these is foul language too, so Danish *lortebil* and *pissekoldt* or German *Scheißwetter* are caught without listing every compound.
- `slurs.voc` – slurs and derogatory terms about groups of people. The comeback never repeats the word: *"please don't use words like that"*.
- `polite_words.voc` – polite phrases, written by hand after the [Stanford politeness strategies](https://github.com/sudhof/politeness): please, thanks, apologies, *could you* / *would you*, deference and greetings. Plain *can you* is not counted, since that is how almost every command starts.

Words people use in ordinary questions ("sex", "nude", "jesus", "bloody nose", "summa cum laude") are deliberately left out, so the plugin never scolds someone for asking about something.

A trained politeness/insult classifier is planned.

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

English, German and Danish. A language needs `foul_language.voc`, `foul_prefixes.voc` (may be empty), `slurs.voc`, `polite_words.voc`, `and.voc` and the five `.dialog` files in `ovos_good_manners_pipeline_plugin/locale/<lang>/`.

### Word lists

`foul_language.voc` and `slurs.voc` are built by `scripts/import_wordlists.py` from open word lists, with hand corrections per language in `scripts/overrides/<lang>.json` (`foul`, `slurs`, `remove`). Re-running the script keeps everything already in the `.voc` files, so hand-written entries are never lost.

- **English** comes from [dsojevic/profanity-list](https://github.com/randolf/profanity-list) (MIT). It is tagged, so slurs (racial, LGBTQ, religious) are sorted out automatically.
- **Other languages** start from [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) (CC-BY-4.0), which is untagged, so each language is sorted by hand in its override file.

Untagged English lists such as [cuss](https://github.com/wooorm/cuss) and LDNOOBW's English list are not used: they mix in ethnic slurs (which would be said back), ordinary words and porn search terms. See [#12](https://github.com/OpenVoiceOS/ovos-good-manners-pipeline-plugin/issues/12).

## Credits

- [@JarbasAl](https://github.com/JarbasAl), original skill
- [@gras64](https://github.com/gras64), German translation
- Word lists: [dsojevic/profanity-list](https://github.com/randolf/profanity-list) (MIT) and [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words) (CC-BY-4.0, by Shutterstock)
- [@andlo](https://github.com/andlo)

## License

Apache 2.0, see [LICENSE.md](LICENSE.md).
