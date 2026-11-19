# Demonstração do EcoTriagem (gerada por `python demo.py --roteiro`)

---

**Imagem:** `dados/bruto/dataset-resized/plastic/plastic1.jpg` — classe real: *plástico*  
**Visão:** plástico (83%); top-3: plastico 83%, vidro 16%, metal 0%  
**OCR:** `(nada legível)`

- **Usuário:** que material é esse?  
  *intenção:* `identificar_material` (0.98); *regras:* R04, R10, R20  
  **EcoTriagem:** Confiança moderada: parece plástico (83%) ou vidro (16%). Confirme o material se puder. O material é **plástico** (fonte: imagem (83%, confiança moderada)). Probabilidades: plástico 83%, vidro 16%, metal 0%.

- **Usuário:** onde eu jogo isso?  
  *intenção:* `como_descartar` (0.97); *regras:* R04, R10, R21  
  **EcoTriagem:** Descarte **plástico** na lixeira **vermelha** da coleta seletiva.

- **Usuário:** precisa tirar a tampa?  
  *intenção:* `preparo_descarte` (0.97); *regras:* R04, R10, R23  
  **EcoTriagem:** Antes de descartar plástico: enxágue embalagens com resto de alimento; retire a tampa (ela também é reciclável) e amasse garrafas PET; embalagens metalizadas (salgadinho) e plástico muito sujo geralmente não são aceitos.

- **Usuário:** tem ponto de coleta perto?  
  *intenção:* `ponto_coleta` (0.98); *regras:* R04, R10, R25a  
  **EcoTriagem:** Em qual bairro de Maceió você está? Assim calculo a rota até o ponto de coleta mais próximo que aceita esse material.

- **Usuário:** moro na Ponta Verde  
  *intenção:* `ponto_coleta` (0.64); *regras:* R04, R10, R25c  
  **EcoTriagem:** Há ponto de coleta no seu próprio bairro (Ponta Verde): PEV Orla Ponta Verde (fictício). Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).

---

**Imagem:** `dados/bruto/dataset-resized/glass/glass101.jpg` — classe real: *vidro*  
**Visão:** vidro (84%); top-3: vidro 84%, plastico 14%, metal 2%  
**OCR:** `(nada legível)`

- **Usuário:** isso é reciclável?  
  *intenção:* `reciclavel` (0.98); *regras:* R02, R10, R22  
  **EcoTriagem:** Sim, **vidro** é reciclável: separe para a coleta seletiva (lixeira verde).

- **Usuário:** quanto tempo leva pra se decompor?  
  *intenção:* `tempo_decomposicao` (0.97); *regras:* R02, R10, R24  
  **EcoTriagem:** Tempo de decomposição estimado de vidro: tempo indeterminado (milhares de anos). (Valores aproximados; dependem muito do ambiente.)

- **Usuário:** onde fica o ponto de coleta mais próximo do Benedito Bentes?  
  *intenção:* `ponto_coleta` (0.98); *regras:* R02, R10, R25c  
  **EcoTriagem:** Ponto de coleta mais próximo para vidro saindo de Benedito Bentes: **Ecoponto Tabuleiro (fictício)** (Tabuleiro do Martins), cerca de 5,9 km. Trajeto: Benedito Bentes → Tabuleiro do Martins. (Busca A*, 2 bairros expandidos.) Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).

---

**Imagem:** `dados/bruto/dataset-resized/cardboard/cardboard110.jpg` — classe real: *papelão*  
**Visão:** papelão (93%); top-3: papelao 93%, vidro 3%, plastico 2%  
**OCR:** `te Ta pe at`

- **Usuário:** qual lixeira uso pra isso?  
  *intenção:* `como_descartar` (0.95); *regras:* R02, R10, R21  
  **EcoTriagem:** Descarte **papelão** na lixeira **azul** da coleta seletiva.

- **Usuário:** devo desmontar a caixa?  
  *intenção:* `preparo_descarte` (0.96); *regras:* R00, R10, R23  
  **EcoTriagem:** Antes de descartar papelão: desmonte e achate as caixas para ocupar menos espaço; remova restos de comida, isopor e excesso de fita adesiva; mantenha seco: papelão molhado perde valor de reciclagem.

---

**Imagem:** `dados/bruto/dataset-resized/metal/metal107.jpg` — classe real: *metal*  
**Visão:** metal (94%); top-3: metal 94%, vidro 4%, plastico 1%  
**OCR:** `NZ`

- **Usuário:** o que é isso na foto?  
  *intenção:* `identificar_material` (0.96); *regras:* R02, R10, R20  
  **EcoTriagem:** O material é **metal** (fonte: imagem (94% de confiança)). Probabilidades: metal 94%, vidro 4%, plástico 1%.

- **Usuário:** dá pra reciclar?  
  *intenção:* `reciclavel` (0.97); *regras:* R02, R10, R22  
  **EcoTriagem:** Sim, **metal** é reciclável: separe para a coleta seletiva (lixeira amarela).

---

**Imagem:** `dados/bruto/dataset-resized/trash/trash101.jpg` — classe real: *rejeito*  
**Visão:** rejeito (não reciclável) (92%); top-3: rejeito 92%, papel 5%, papelao 3%  
**OCR:** `(nada legível)`

- **Usuário:** onde descarto?  
  *intenção:* `como_descartar` (0.59); *regras:* R02, R10, R21  
  **EcoTriagem:** **Rejeito (não reciclável)** vai na lixeira **cinza** e segue pela coleta regular, separado dos recicláveis.

- **Usuário:** tem ecoponto pra isso no Farol?  
  *intenção:* `ponto_coleta` (0.86); *regras:* R02, R10, R25b  
  **EcoTriagem:** Rejeito (não reciclável) não precisa ir a ecoponto: entregue na coleta regular de lixo, em saco fechado.

---

**Imagem:** `dados/bruto/dataset-resized/paper/paper122.jpg` — classe real: *papel*  
**Visão:** papel (85%); top-3: papel 85%, rejeito 8%, plastico 7%  
**OCR:** `4, tt, to,`

- **Usuário:** e se fosse uma pilha velha, onde levo? estou no Pinheiro  
  *intenção:* `ponto_coleta` (0.94); *regras:* R00, R01, R10, R25c  
  **EcoTriagem:** Atenção: na imagem eu reconheci papel (85%), mas você mencionou resíduo perigoso (pilha, bateria, lâmpada, eletrônico). Vou responder sobre resíduo perigoso (pilha, bateria, lâmpada, eletrônico). Ponto de coleta mais próximo para resíduo perigoso (pilha, bateria, lâmpada, eletrônico) saindo de Pinheiro: **Ponto de logística reversa Centro (fictício)** (Centro), cerca de 4,7 km. Trajeto: Pinheiro → Farol → Centro. (Busca A*, 4 bairros expandidos.) Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).

---

**Sem imagem** (apenas texto)

- **Usuário:** oi  
  *intenção:* `saudacao_ajuda` (0.94); *regras:* R27  
  **EcoTriagem:** Olá! Sou o EcoTriagem. Envie a foto de um resíduo e pergunte, por exemplo: 'que material é esse?', 'onde descarto?', 'precisa lavar?', 'é reciclável?' ou 'qual o ponto de coleta mais perto do Farol?'.

- **Usuário:** o que é logística reversa?  
  *intenção:* `duvida_geral` (0.95); *regras:* R26  
  **EcoTriagem:** Logística reversa é o retorno de produtos após o uso ao fabricante ou comerciante para reaproveitamento ou destinação adequada. É obrigatória para pilhas, baterias, lâmpadas, eletroeletrônicos, pneus, óleos lubrificantes e agrotóxicos (Lei 12.305/2010, art. 33).

- **Usuário:** qual a previsão do tempo amanhã?  
  *intenção:* `fora_escopo` (0.96); *regras:* R28  
  **EcoTriagem:** Desculpe, só consigo ajudar com identificação e descarte de resíduos.

- **Usuário:** onde fica o ecoponto mais perto do Jacintinho?  
  *intenção:* `ponto_coleta` (0.98); *regras:* R25d  
  **EcoTriagem:** Há ponto de coleta no seu próprio bairro (Jacintinho): **Ecoponto Jacintinho (fictício)**. Envie a foto do resíduo para eu conferir se o ponto aceita o material. Obs.: pontos de coleta do mapa são fictícios (cenário de demonstração).
