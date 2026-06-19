# Aura AI Assistant — Kaynaklı (Grounded) RAG Destek Asistanı

[🇬🇧 English](README.md) · **🇹🇷 Türkçe**

Bir B2B SaaS senaryosu için Retrieval-Augmented Generation (RAG) destek asistanı.
Son kullanıcı sorularını yalnızca müşterinin bilgi tabanından yanıtlar ve **her cevap
korpusa dayanır ve kaynaklarını gösterir**. Sistem, getirilen bağlamdan güvenilir biçimde
cevap veremediğinde tahmin yürütmek yerine **çekimser kalır** (abstain).

Bu vaka çalışmasının korpusu **Aura Hub G2** akıllı ev cihazı etrafında kuruludur: kurulum
kılavuzları, abonelik/iade politikası, LED durum göstergeleri, hata kodları, teknik
özellikler, ayrıntılı bir kullanıcı kılavuzu, sorun giderme SSS'i ve bir gizlilik & veri
politikası. Korpus bilinçli olarak heterojendir — düz metin ve yapılı dokümanlar, çok
sayfalı PDF'ler, taranmış sayfalar, ekran görüntüleri, gömülü metin/tablo içeren görseller
ve elektronik tablo dosyaları (Excel/CSV). **Bazı bilgiler yalnızca görsellerin veya
tabloların içinde bulunur ve hiçbir düz-metin dokümanda tekrarlanmaz**; bu yüzden görsel ve
tablolu içerik, yapısı ve bağlamı korunarak çıkarılmalıdır.

---

## İçindekiler

- [Mimari Genel Bakış](#mimari-genel-bakış)
- [Korpus](#korpus)
- [Gereksinimler](#gereksinimler)
- [Ortam Değişkenleri](#ortam-değişkenleri)
- [Kurulum & Çalıştırma (Docker)](#kurulum--çalıştırma-docker)
- [Ingestion (Korpusun İndekslenmesi)](#ingestion-korpusun-i̇ndekslenmesi)
- [Değerlendirme](#değerlendirme)
- [Proje Yapısı](#proje-yapısı)
- [Temel Kararlar](#temel-kararlar)
- [Değerlendirme Sonuçları](#değerlendirme-sonuçları)

---

## Mimari Genel Bakış

```
Kullanıcı ──> Frontend ──> Backend API
                              │
                ┌─────────────┼──────────────┐
                │             │              │
            Retrieval    Generation      Abstention
           (hybrid +     (LLM, atıflı    (düşük güven
            reranking)    cevap)         → cevap yok)
                │
          Vektör Deposu  <── Ingestion ── Korpus (PDF / görsel / Excel / md)
```

> Bileşen seçimleri [`DECISIONS.tr.md`](DECISIONS.tr.md) içinde gerekçelendirilmiştir.

## Korpus

Bilgi tabanı (yerelde tutulur, depoya **commit edilmez** — bkz. [.gitignore](.gitignore)):

| Dosya                              | Tür             | Notlar                                 |
|------------------------------------|-----------------|----------------------------------------|
| `01_aura_kurulum_kilavuzu.md`      | Markdown        | Kurulum & başlangıç kılavuzu           |
| `02_abonelik_ve_iade_politikasi`   | PDF             | Abonelik & iade politikası             |
| `03_led_durum_gostergeleri`        | PNG (görsel)    | LED durum göstergeleri (yalnızca görsel)|
| `04_hata_kodlari`                  | XLSX            | Hata kodları (tablo)                   |
| `05_teknik_ozellikler_scan`        | PDF (taranmış)  | Teknik özellikler (taranmış sayfa)     |
| `06_detayli_kullanici_kilavuzu`    | PDF             | Ayrıntılı kullanıcı kılavuzu           |
| `07_sorun_giderme_sss`             | PDF             | Sorun giderme SSS                      |
| `08_gizlilik_ve_veri_politikasi`   | PDF             | Gizlilik & veri politikası             |


## Gereksinimler

- Docker ve Docker Compose
- _(Yerel geliştirme için)_ Python 3.11+ ve Node.js 20+

## Ortam Değişkenleri

`.env.example`'ı `.env`'e kopyalayıp doldurun:

```bash
cp .env.example .env
```

| Değişken            | Açıklama                                                     | Zorunlu  |
|---------------------|--------------------------------------------------------------|----------|
| `OPENAI_API_KEY`    | OpenAI anahtarı — embedding (`text-embedding-3-small`) **ve** generation (`gpt-4o-mini`) | Evet |
| `ANTHROPIC_API_KEY` | Claude anahtarı — taranmış/görsel dokümanlar için varsayılan vision sağlayıcısı (`claude-opus-4-8`) | Evet¹ |
| `GEMINI_API_KEY`    | Gemini anahtarı — alternatif vision sağlayıcısı (`VISION_PROVIDER=gemini`) | Evet¹ |
| `VISION_PROVIDER`   | `anthropic` (varsayılan) veya `gemini`; boşsa hangi anahtar varsa ondan otomatik seçer | Hayır |
| `GENERATION_MODEL`  | Generation modeli override (varsayılan `gpt-4o-mini`)        | Hayır    |
| `EMBEDDING_MODEL`   | Embedding modeli override (varsayılan `text-embedding-3-small`)| Hayır    |
| `ABSTENTION_THRESHOLD` | En iyi retrieval skoru bunun altındaysa sistem çekimser kalır (varsayılan `0.38`) | Hayır |
| `QDRANT_URL`        | Vektör deposu URL'i (`http://localhost:6333`, Compose'da `http://qdrant:6333`) | Hayır |
| `QDRANT_COLLECTION` | Qdrant koleksiyon adı (varsayılan `aura_corpus`)             | Hayır    |

> ¹ Bir vision anahtarı yalnızca ham korpus üzerinde **ingestion'ı yeniden çalıştırmak**
> için gerekir. Önceden hazırlanmış `data/chunks.jsonl` commit'lidir; dolayısıyla normal
> bir `docker compose up` açılışı ondan indeksler ve **yalnızca `OPENAI_API_KEY`'e**
> ihtiyaç duyar (embedding + generation).

## Kurulum & Çalıştırma (Docker)

Tüm sistem (backend + frontend + vektör deposu) tek komutla ayağa kalkar.
`docker-compose.yml` iki servis tanımlar: **Qdrant** (vektör deposu) ve **backend**
(statik frontend'i de servis eden RAG API'si — ayrı bir frontend container'ı yok).

### 1. İlk derleme

```bash
cp .env.example .env      # ardından OPENAI_API_KEY'i doldurun (bkz. Ortam Değişkenleri)
docker compose up --build
```

`--build`, backend imajını derler. Başlangıçta backend, **korumalı bir otomatik indeksleme**
çalıştırır (`backend/entrypoint.sh`): Qdrant koleksiyonu boşsa önceden hazırlanmış
`data/chunks.jsonl`'i embed edip upsert eder, sonra API'yi servis eder. Pahalı
ingestion/vision adımı **yeniden çalıştırılmaz** — chunk listesi commit'lidir ve imaja gömülüdür.

- **Uygulama (UI + API):** http://localhost:4242
- API uçları: `POST /query`, `GET /metrics`, `GET /decisions/{tr|en}`, `GET /health`, `GET /docs`
- Qdrant paneli: http://localhost:6333/dashboard

### 2. Sonraki çalıştırmalar

İmajlar mevcutsa yeniden derleme gerekmez. Qdrant verisi adlandırılmış bir Docker
volume'ünde (`qdrant_storage`) yaşar; böylece indeks yeniden başlatmalar arasında korunur ve
otomatik indeksleme adımı atlanır (koleksiyonu boş olmayan görür):

```bash
docker compose up            # başlat (derlenmiş imajları + indeksli volume'ü tekrar kullanır)
docker compose up -d         # ... veya arka planda (detached)
docker compose down          # durdur (volume korunur → veri kalıcı)
docker compose down -v       # durdur VE Qdrant volume'ünü sil (sonraki açılışta yeniden indeksler)
```

Yeniden derleme yalnızca kod değişikliğinden sonra: `docker compose up --build`.

### 3. Sıfırdan git clone ile çalıştırma (size verilen bir `.env` ile)

Bu en yaygın devir senaryosudur: biri size hazır bir `.env` verir, siz depoyu clone'lar ve
çalıştırmak istersiniz. Proje kendi kendine yeterlidir — korpus indeksi (`data/chunks.jsonl`)
commit'lidir ve imaja gömülür — bu yüzden temiz bir clone artı geçerli bir `.env` yeterlidir.
Hiçbir ingestion veya vision adımı devreye girmez.

```bash
# 1. Depoyu clone'layın
git clone <repo> && cd project

# 2. Size verilen .env'i proje köküne (docker-compose.yml'in yanına) koyun.
#    .env gitignore'dadır, yani clone ile GELMEZ — kendiniz kopyalamanız gerekir:
cp /yol/verilen/.env .env
#    ... ya da başka bir makineden kopyalayın:
#    scp kullanıcı@host:/yol/.env .env
#    (Alternatif olarak şablondan yeniden oluşturun: cp .env.example .env, sonra doldurun.)

# 3. Derle ve başlat
docker compose up --build
```

Uygulama ardından **http://localhost:4242** adresinde servis edilir.

Notlar:
- `.env` **gitignore'dadır** ve clone'un parçası asla değildir — her zaman depo dışından
  taşınmalıdır (paylaşılan dosya, `scp`, secrets manager) ya da `.env.example`'dan yeniden
  oluşturulmalıdır.
- Normal bir açılış için yalnızca `OPENAI_API_KEY` gereklidir (embedding + generation). Bir
  vision anahtarı (`ANTHROPIC_API_KEY` veya `GEMINI_API_KEY`) sadece ham `doc/` korpusu
  üzerinde ingestion'ı yeniden çalıştırmak için gerekir.
- Compose içinde backend Qdrant'a servis adıyla ulaşır; `QDRANT_URL` otomatik olarak
  `http://qdrant:6333` ile geçersiz kılınır, dolayısıyla `.env`'de ayarlamanıza gerek yok.

## Ingestion (Korpusun İndekslenmesi)

Bu adım korpusu (PDF, görseller, Excel/CSV, markdown) işler ve görsel/tablolu içeriğin
yapısını ve bağlamını koruyarak vektör deposuna yükler. Yalnızca ham korpus yeniden
indekslenirken gereklidir; standart bir açılış önceden hazırlanmış `data/chunks.jsonl`'i
kullanır.

```bash
# 1. Vektör deposunu (Qdrant) başlatın. Panel: http://localhost:6333/dashboard
docker compose up -d qdrant

# 2. Korpusu etiketli bir chunk listesine oku
python -m backend.ingestion.run --doc-dir doc --out data/chunks.jsonl

# 3. Her chunk'ı embed et ve Qdrant'a yükle
python -m backend.index

# Cevabı yalnızca bir görselde olan bir sorguyla doğrula (LED kartı, doc 03)
python -m backend.index --query "internet bağlantısı yok"
```

## Değerlendirme

45 soru/cevap çiftinden oluşan bir golden set (`eval/golden_set.jsonl`), retrieval kalitesini
(recall@k, MRR), generation kalitesini (faithfulness/dayalılık, answer relevance) ve
abstention doğruluğunu — **dense vs hybrid yan yana** — tek komutla ve harici bir eval
framework'ü olmadan ölçer:

```bash
python -m eval.run                 # tam: retrieval + generation (LLM judge) + abstention
python -m eval.run --retrieval-only  # ücretsiz, deterministik yalnızca retrieval metrikleri (LLM yok)
python -m eval.run --no-judge        # cevaplar + abstention, LLM judge'ı atla
```

Sonuçlar, koşum config'i ve soru-başına detay `eval/results.json`'a yazılır. LLM judge,
`gpt-4o`'da koşar (`gpt-4o-mini` üreticiden daha güçlü, self-bias'ı sınırlamak için).
Metodoloji için bkz. **[`DECISIONS.tr.md`](DECISIONS.tr.md#değerlendirme-adım-4)**.

## Proje Yapısı

```
.
├── backend/          # RAG API (ingestion, retrieval, generation) + frontend'i servis eder
├── frontend/         # Statik web UI (sohbet, metrikler, mimari, prompt-akışı, kararlar)
├── eval/             # Değerlendirme harness'ı + golden set
├── docker-compose.yml
├── README.md         # İngilizce README
├── README.tr.md      # Bu dosya (Türkçe)
├── DECISIONS.md      # Tasarım & gerekçe dokümanı (İngilizce)
├── DECISIONS.tr.md   # Tasarım & gerekçe dokümanı (Türkçe)
├── brief.pdf         # Vaka çalışması brief'i (git-ignore)
└── doc/              # Korpus dokümanları (git-ignore)
```

## Temel Kararlar

Chunking, embedding modeli, vektör deposu, retrieval stratejisi, görsel/tablo işleme ve
production'a taşıma planı için gerekçeler **[`DECISIONS.tr.md`](DECISIONS.tr.md)** içinde
belgelenmiştir.

## Değerlendirme Sonuçları

45 soruluk golden set üzerinde ölçüldü. Hybrid retrieval her retrieval metriğinde ve
generation kalitesinde kazanır — Adım 3b kararının ölçülmüş gerekçesi.

| Metrik             | Dense (baseline) | Hybrid   |
|--------------------|------------------|----------|
| Recall@3           | 0.93             | **0.97** |
| Recall@5           | 0.95             | **0.97** |
| MRR                | 0.81             | **0.90** |
| Faithfulness       | 0.96             | **0.99** |
| Answer Relevance   | 0.91             | **0.92** |
| Abstention recall  | 5/5              | 5/5      |
| False abstentions  | 3/40             | 3/40     |

> `python -m eval.run` ile yeniden üretin. Judge modeli: `gpt-4o`; generation: `gpt-4o-mini`.
