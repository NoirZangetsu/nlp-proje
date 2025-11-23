# Örnek Inputlar - XAI Rationale Generation Agent

Bu dosya, Streamlit uygulamasını test etmek için kullanabileceğiniz örnek film yorumlarını içerir.

## Pozitif Yorumlar

### 1. Kısa ve Net
```
This movie was absolutely fantastic! The acting was superb and the plot was engaging.
```

### 2. Detaylı
```
I loved every minute of this film. The cinematography was breathtaking, the soundtrack was perfect, and the performances were outstanding. Highly recommended!
```

### 3. Duygusal
```
This is one of the best movies I've ever seen. It made me laugh, cry, and think deeply about life. Truly inspiring!
```

### 4. Karşılaştırmalı
```
After watching many disappointing films this year, this one restored my faith in cinema. Brilliant storytelling and exceptional direction.
```

## Negatif Yorumlar

### 5. Kısa ve Net
```
This movie was terrible. The plot made no sense and the acting was awful.
```

### 6. Detaylı
```
I was extremely disappointed with this film. The dialogue was poorly written, the characters were one-dimensional, and the pacing was completely off. A waste of time and money.
```

### 7. Duygusal
```
This is the worst movie I've seen in years. Boring, confusing, and completely unengaging. I wanted to leave halfway through.
```

### 8. Karşılaştırmalı
```
I had high expectations but this film failed to deliver. The special effects were cheap, the story was predictable, and the ending was unsatisfying.
```

## İlginç Örnekler (Test İçin)

### 9. Karışık Duygular
```
The movie had some great moments but overall it was disappointing. The first half was excellent but it fell apart in the second act.
```

### 10. İroni/Sarkazm
```
Oh sure, this was a "masterpiece". If you consider terrible acting and a nonsensical plot a masterpiece.
```

### 11. Nötr/Objektif
```
The film presents an interesting perspective on the topic. While the technical aspects are solid, the narrative could have been more compelling.
```

### 12. Kısa Tek Cümle
```
Amazing film!
```

### 13. Uzun ve Karmaşık
```
Despite its ambitious scope and impressive visual effects, the movie struggles to maintain narrative coherence. While individual scenes are well-crafted, the overall story arc feels disjointed, leaving viewers with more questions than answers. The performances range from mediocre to excellent, creating an inconsistent viewing experience that ultimately undermines the film's potential impact.
```

## Kullanım Notları

- Bu örnekleri Streamlit uygulamasındaki text area'ya kopyalayıp yapıştırabilirsiniz
- Her örnek farklı bir senaryoyu test eder:
  - **Pozitif/Negatif tahminleri** test edin
  - **Integrated Gradients** ile token önemini görün
  - **Causal Analysis** ile hangi kelimelerin gerçekten etkili olduğunu keşfedin
  - **Rationale Generation** ile doğal dil açıklamaları üretin

## Test Senaryoları

1. **Basit Pozitif**: Örnek 1 veya 12
2. **Basit Negatif**: Örnek 5
3. **Karmaşık Pozitif**: Örnek 2 veya 13
4. **Karmaşık Negatif**: Örnek 6 veya 13
5. **Sınır Durumları**: Örnek 9, 10, 11

