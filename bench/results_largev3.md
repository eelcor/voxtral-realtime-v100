# Whisper vs. Voxtral-realtime — Nederlandse WER (FLEURS-nl)

- Clips: **80** (794.7s audio)
- Whisper-model: **large-v3** (cuda/float16)
- Voxtral: **realtime** (streaming, via vLLM)

| Model | WER ↓ | RTF ↓ |
|---|---|---|
| Whisper large-v3 |   6.2% | 0.065 |
| Voxtral-realtime |   9.0% | 0.695 |

_Lager is beter. WER = Word Error Rate; RTF = Real-Time Factor (verwerkingstijd ÷ audioduur)._

### Grootste verschillen (Voxtral t.o.v. Whisper)

**clip 36 (8.8s)** — W= 10.0% V= 40.0%
- ref:     met zijn m16-geweer schoot rolando mendoza op de toeristen
- whisper: Met zijn M16 geweer schoot Ronaldo Mendoza op de toeristen.
- voxtral: Metzen M16-geweers schoot Ronaldo Mendoza op de toeristen.

**clip 5 (13.0s)** — W=  5.3% V= 26.3%
- ref:     er wordt met name gesteld dat niemand een leugen kan herkennen door op enkel op micro-expressies te letten
- whisper: Er wordt met name gesteld dat niemand een leugen kan herkennen door enkel op micro-expressies te letten.
- voxtral: Ruft mijn naam gesteld dat niemand een leugen kan herkennen door enkel op micro-expressies te letten.

**clip 50 (4.3s)** — W=  0.0% V= 16.7%
- ref:     de vondst geeft ook inzicht in de evolutie van veren bij vogels
- whisper: De vondst geeft ook inzicht in de evolutie van veren bij vogels.
- voxtral: De voorschrift ook inzicht in de evolutie van veren bij vogels.

**clip 74 (15.2s)** — W=  2.9% V= 17.6%
- ref:     de onderzoekers trokken de conclusie dat de rachis waarschijnlijk een latere evolutionaire ontwikkeling was doordat dinosaurusveren geen goed ontwikkelde schacht oftewel een rachis hebben maar wel over andere kenmerken van veren beschikken zoals weerhaken
- whisper: De onderzoekers trokken de conclusie dat de rachis waarschijnlijk een later evolutionaire ontwikkeling was,  doordat dinosaurusveren geen goed ontwikkelde schacht, oftewel een rachis, hebben,  maar wel over andere kenmerken van veren beschikken, zoals weerhaken.
- voxtral: Der onderzoekerstrakkere conclusie dat de rachis waarschijnlijk een later evolutionaire ontwikkeling was, doordat dinosaursveren geen goed ontwikkelde schacht oftewel een rachis hebben, maar wel over andere kenmerken van veren beschikken zoals weerhaken.

**clip 60 (13.5s)** — W=  4.5% V= 18.2%
- ref:     wees duidelijk wanneer je een man afwijst en weest niet bang om je mannetje te staan culturele verschillen praten dit niet goed!
- whisper: Wees duidelijk wanneer je een man afwijst en wees niet bang om je mannetje te staan.  Culturele verschillen praten dit niet goed.
- voxtral: Wie is duidelijk wanneer je een man afwijst en wie is niet bang om je mannetje te staan? Culturele verschillen praten dit niet goed.
