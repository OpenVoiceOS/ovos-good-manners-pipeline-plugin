# Overheard: can the assistant catch swearing it was not asked about?

An experiment for [emkay-voice/emkay#94](https://github.com/emkay-voice/emkay/issues/94): instead of only judging what is said *to* the assistant, listen on the device for swearing in the room ("av for helvede!") and let good manners comment. Everything runs locally. Nothing is recorded, sent or stored.

The listener is [WakePhoneHuBERT](https://huggingface.co/TigreGotico/wakephonehubert) (Apache-2.0), a 3.5 MB streaming phone model that spots a phrase zero-shot from its phones. The question is whether it does so without too many false comments, in more languages than English.

## Phrase lists

`phrases.json` holds a few **long** phrases per language. Short words ("lort", "shit") have too few phones and give many false hits. There are no slurs, because a false accusation there is too ugly. The English list is a **base listened for in every language**, both with English phones and with the language's own phones (the way a Dane says "what the fuck"). Each language also has innocent **look-alikes** ("jeg fandt den i skuffen" against "for fanden") that must not trigger.

## Test

- **Must not trigger:** FLEURS test speech (CC-BY-4.0), 1–3.8 h of ordinary read speech per language, 23 h in total.
- **Must trigger:** every phrase spoken by 3–20 Piper voices per language, in four conditions: clean, room echo (RT60 0.3–0.6 s, a fairly distant talker), pink noise at 10 dB SNR, and other people talking at 10 dB SNR. Danish has only one Piper voice, so Norwegian and Swedish voices are also made to speak Danish (Danish phonemes into their models).
- Scored like the model card and the plugin: a 1.5 s window, scored after every 80 ms block, with hits less than 2 s apart counting once.

```bash
pip install onnxruntime piper-tts soundfile numpy pyarrow phonemizer==3.4.0 espeakng-loader==0.2.4 faster-whisper
python3 download.py                      # model, voices and FLEURS into ./data (about 8 GB)
for l in en da de sv nb nl fr es it; do python3 run_neg.py $l; python3 run_pos.py $l; done
python3 report.py en da de sv nb nl fr es it
python3 stage2.py en; python3 stage2.py da; python3 stage2.py de
python3 run_enroll.py da                 # heard-phone keywords, see below
```

## Results (November 2026)

### One stage: WakePhoneHuBERT alone

Recall at the threshold that gives at most **1 false hit per hour per phrase** on ordinary speech:

| lang | own phrases | English base | clean | room | noise | babble | best own phrase |
|---|---|---|---|---|---|---|---|
| en | 0.60 | 0.60 | 0.81 | 0.19 | 0.75 | 0.65 | motherfucker 0.80 |
| it | 0.75 | 0.54 | 0.72 | 0.36 | 0.72 | 0.62 | porca miseria 1.00 |
| es | 0.51 | 0.28 | 0.48 | 0.20 | 0.39 | 0.37 | me cago en la leche 0.67 |
| de | 0.40 | 0.17 | 0.33 | 0.10 | 0.31 | 0.27 | verdammte Scheiße 0.46 |
| sv | 0.32 | 0.20 | 0.42 | 0.09 | 0.22 | 0.22 | fan också 0.64 |
| nl | 0.31 | 0.15 | 0.32 | 0.08 | 0.23 | 0.17 | wat een klootzak 0.43 |
| nb | 0.24 | 0.41 | 0.50 | 0.15 | 0.39 | 0.39 | faen i helvete 0.66 |
| da | 0.19 | 0.15 | 0.27 | 0.05 | 0.18 | 0.18 | hold kæft mand 0.59 |
| fr | 0.18 | 0.13 | 0.23 | 0.03 | 0.16 | 0.18 | va te faire foutre 0.29 |

On its own this is **not usable**. The rate is 1 false hit per hour *per phrase*, and with 9 English and 4–6 own phrases they add up to over 10 false comments an hour of conversation. Room echo is the hardest condition. Full per-phrase numbers are in `report.py`'s output.

**Why Danish is worst:** eSpeak's Danish phones are far from what the model hears. For "av for helvede" eSpeak gives `ɑ u f ʌ h ɛ l ʋ ə ð ə`, while the model hears `a l f o h i v ə`. `run_enroll.py` takes the keyword phones from what the model hears when a few enrolment voices say the phrase, and measures on the other voices. That raises Danish one-stage recall from **0.20 to 0.32** ("av for helvede" 0.23 → 0.55, "god damn it" 0.02 → 0.48). The enrolment and test voices partly come from the same Piper model (nvcc), so this is somewhat optimistic.

### Two stages: spotter as a sensitive first ear, local STT confirms

WakePhoneHuBERT is set sensitive (10 false hits per hour per phrase). Only when it fires, local STT (faster-whisper small, standing in for the device's own STT) transcribes the last 2.5 s, and good manners' own word lists decide on the text.

| lang | STT runs / hour of talk | **false comments / hour** | recall | clean | room | noise | babble |
|---|---|---|---|---|---|---|---|
| en | 60 | **0** (in 1.95 h) | 0.65 | 0.83 | 0.44 | 0.81 | 0.53 |
| de | 95 | **0** (in 3.4 h) | 0.15 | 0.18 | 0.11 | 0.14 | 0.16 |
| da | 101 | **0.31** (1 in 3.2 h) | 0.08 | 0.17 | 0.03 | 0.07 | 0.03 |

The second stage **removes almost all false comments**. The one Danish false comment was Whisper mishearing "frugtjuicer" as "Fuckdjusern". In English it is usable. In German and Danish it is limited by three things that can be fixed:

- **STT forced to the device language:** STT runs in the device's language, so English swearing in German is written as German ("Das war's für ein Teufel").
- **Word lists too strict:** "Verdammt nochmal!" and "Halt die Klappen" are missed because the lists only have "verdammt noch mal" and "Klappe". "Motherfuck" without -er is missing too.
- **Censored words:** Whisper censors words as `f**k`. `stage2.py` now counts that as foul, but the numbers above were measured before that change.
- **Danish STT:** whisper-small is weak on Danish. A device with a national Danish STT model will do better.

The positives are synthetic: Piper voices, mostly accented for Danish. Real recordings from real people are still needed before this decision is final.

## What this means

- **One stage alone** (phone spotting) is not good enough to comment on what is overheard.
- **Two stages** (spotter → local STT → good manners' word lists) give almost no false comments. That is the design to build: the spotter keeps STT from running all the time, and the final decision is made on text with the lists this plugin already has, short words included.

## Building it OVOS style

- **Stage 1 already exists:** [ovos-ww-plugin-wakeforge](https://github.com/OpenVoiceOS/ovos-ww-plugin-wakeforge) has the `ovos-ww-plugin-wakeforge-zeroshot` engine. It takes `ipa` as a list of pronunciations, so the English and local pronunciations and the heard phones from `run_enroll.py` can all be listed. Its `confirm_blocks` (two windows in a row) was not used in this test and should lower false hits further. The same plugin can also run trained heads, which would beat zero-shot for the short and hard phrases.
- **The listener already supports it:** in ovos-dinkum-listener a hotword with `"listen": false` and `"bus_event": "<message>"` sends a message without starting to listen.
- **The missing piece is the audio:** stage 2 needs the seconds around the hit, transcribed locally. The bus event does not carry them. Two options: the listener transcribes the hotword's own audio (for example a `"transcribe": true` hotword option), or the event carries the audio.
- **Good manners** answers that message with its existing cool-down, per-session state, word lists and dialogs. The overheard phrases become a locale file next to the others, converted to IPA in CI.
